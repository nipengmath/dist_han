# -*- coding: utf-8 -*-

import tensorflow as tf
from base.model import BaseModel
from typing import Dict
import numpy as np
import json

from models import conf
from models.func import cudnn_gru, native_gru, dot_attention, summ, dropout, ptr_net, dense, bi_shortcut_stacked_lstm_return_sequences

class HanModel(BaseModel):
    def __init__(self, config: dict) -> None:
        """
        Create a model used to classify hand written images using the MNIST dataset
        :param config: global configuration
        """
        super().__init__(config)

    def model(
        self, features: Dict[str, tf.Tensor], labels: tf.Tensor, mode: str
    ) -> tf.Tensor:
        """
        Define your model metrics and architecture, the logic is dependent on the mode.
        :param features: A dictionary of potential inputs for your model
        :param labels: Input label set
        :param mode: Current training mode (train, test, predict)
        :return: An estimator spec used by the higher level API
        """
        # set flag if the model is currently training
        is_training = mode == tf.estimator.ModeKeys.TRAIN

        # get input data
        _input = features["input"]
        # initialise model architecture
        logits = _create_model(_input, self.config["keep_prob"], is_training, self.config)

        # define model predictions
        predictions = {
            "class": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits),
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            # define what to output during serving
            export_outputs = {
                "labels": tf.estimator.export.PredictOutput(
                    ## {"id": features["id"], "label": predictions["class"]}
                    {"label": predictions["class"]}
                )
            }
            return tf.estimator.EstimatorSpec(
                mode, predictions=predictions, export_outputs=export_outputs
            )

        # calculate loss
        labels = tf.reshape(tf.decode_raw(labels, tf.float32), [-1, conf.num_classes])
        ## labels = tf.reshape(labels, [-1, 1])
        print("==3", labels)
        print("==4", logits)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

        # add summaries for tensorboard
        tf.summary.scalar("loss", loss)
        ## tf.summary.image("input", tf.reshape(image, [-1, 28, 28, 1]))

        if mode == tf.estimator.ModeKeys.EVAL:
            # create a evaluation metric

            print("==5", labels)
            print("==6", predictions["class"])
            summaries_dict = {
                "val_accuracy": tf.metrics.accuracy(
                    tf.argmax(labels, 1), predictions=predictions["class"]
                )
            }
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=summaries_dict
            )

        # assert only reach this point during training mode
        assert mode == tf.estimator.ModeKeys.TRAIN

        # collect operations which need updating before back-prob e.g. Batch norm
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # create learning rate variable for hyper-parameter tuning
        lr = tf.Variable(
            initial_value=self.config["learning_rate"], name="learning-rate"
        )

        # initialise optimiser
        optimizer = tf.train.AdamOptimizer(lr)

        # Do these operations after updating the extra ops due to BatchNorm
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.minimize(
                loss,
                global_step=tf.train.get_global_step(),
                colocate_gradients_with_ops=True,
            )

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


# def _create_model(x: tf.Tensor, drop: float, is_training: bool) -> tf.Tensor:
#     """
#     A basic deep CNN used to train the MNIST classifier
#     :param x: input data
#     :param drop: percentage of data to drop during dropout
#     :param is_training: flag if currently training
#     :return: completely constructed model
#     """
#     x = tf.reshape(x, [-1, 28, 28, 1])
#     _layers = [1, 1]
#     _filters = [32, 64]

#     # create the residual blocks
#     for i, l in enumerate(_layers):
#         x = _conv_block(x, l, _filters[i], is_training)

#     x = tf.layers.Flatten()(x)
#     _fc_size = [1024]

#     # create the fully connected blocks
#     for s in _fc_size:
#         x = _fc_block(x, s, is_training, drop)
#     # add an output layer (10 classes, one output for each)
#     return tf.layers.Dense(10)(x)


def load_word_mat(path):
    with open(path, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    return word_mat


def _create_model(x: tf.Tensor, drop: float, is_training: bool, config: dict) -> tf.Tensor:
    N, PN, PL, HS = config['train_batch_size'], conf.c_maxnum, conf.c_maxlen, conf.hidden_size
    c = tf.reshape(tf.decode_raw(x, tf.int32), [N, PN, PL])
    gru = native_gru

    print("c", c)
    c_mask = tf.cast(tf.reshape(c, [N*PN, PL]), tf.bool)
    print("c_mask", c_mask)
    c_len = tf.reduce_sum(tf.cast(c_mask, tf.int32), axis=1)
    print("c_len", c_len)

    word_mat = load_word_mat(conf.word_emb_file)

    with tf.variable_scope("emb"):
        c_emb = tf.nn.embedding_lookup(word_mat, c)
        print("c_emb", c_emb)
        c_emb = tf.reshape(c_emb, [N*PN, PL, c_emb.get_shape().as_list()[-1]])
        print("c_emb reshape", c_emb)

    print("===2", is_training)

    with tf.variable_scope("word_encoder"):
        rnn = gru(num_layers=1, num_units=HS, batch_size=N*PN, input_size=c_emb.get_shape(
        ).as_list()[-1], keep_prob=config['keep_prob'], is_train=is_training, scope="word")
        c_encoder = rnn(c_emb, seq_len=c_len)
        print("c_encoder", c_encoder)

    with tf.variable_scope("word_attention"):
        dim = c_encoder.get_shape().as_list()[-1]
        u = tf.nn.tanh(dense(c_encoder, dim, use_bias=True, scope="dense"))
        print("u", u)
        u2 = tf.reshape(u, [N*PN*PL, dim])
        print("u2", u2)
        uw = tf.get_variable("uw", [dim, 1])
        alpha = tf.matmul(u2, uw)
        print("alpha", alpha)
        alpha = tf.reshape(alpha, [N*PN, PL])
        print("alpha", alpha)
        alpha = tf.nn.softmax(alpha, axis=1)
        print("alpha", alpha)
        s = tf.matmul(tf.expand_dims(alpha, axis=1), c_encoder)
        print("s", s)

    with tf.variable_scope("sent_encoder"):
        dim = s.get_shape().as_list()[-1]
        s = tf.reshape(s, [N, PN, dim])
        print("s", s)
        s_len = tf.constant([PN for _ in range(N)],shape=[N,], name='s_len')
        print("s_len", s_len)
        rnn = gru(num_layers=1, num_units=HS, batch_size=N, input_size=dim,
                  keep_prob=config['keep_prob'], is_train=is_training, scope="sent")
        h = rnn(s, seq_len=s_len)
        print("h", s)

    with tf.variable_scope("sent_attention"):
        dim = s.get_shape().as_list()[-1]
        u = tf.nn.tanh(dense(s, dim, use_bias=True, scope="dense"))
        print("u", u)
        u2 = tf.reshape(u, [N*PN, dim])
        print("u2", u2)
        us = tf.get_variable("us", [dim, 1])
        print("us", us)
        alpha2 = tf.matmul(u2, us)
        print("alpha2", alpha2)
        alpha2 = tf.reshape(alpha2, [N, PN])
        print("alpha2", alpha2)
        alpha2 = tf.nn.softmax(alpha2, axis=1)
        print("alpha2", alpha2)
        print(tf.expand_dims(alpha2, axis=1))
        print(s)
        v = tf.matmul(tf.expand_dims(alpha2, axis=1), s)
        print("v", v)
        v = tf.reshape(v, [N, dim])
        print("v", v)

    with tf.variable_scope("output"):
        logits = dense(v, conf.num_classes,
                       use_bias=True, scope="output")
        print("logits", logits)
        scores = tf.nn.softmax(logits)
        print("scores", scores)
    return logits
