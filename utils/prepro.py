# -*- coding: utf-8 -*-

import tensorflow as tf
print(tf.__version__)
import random, re, codecs

import json
from collections import Counter, defaultdict
import numpy as np
import os.path
import jieba

## jieba.load_userdict("newdict.txt")


def word_tokenize(sent):
    ## return list(jieba.cut(sent))
    return list(sent.split(" "))


def process_file(filename, word_counter):
    print("process file {}".format(filename))
    examples = []
    eval_examples = {}

    total = 0
    ## 读取数据
    label_samples = defaultdict(list)
    with codecs.open(filename) as f:
        for line in f:
            total += 1
            row = json.loads(line)
            text = row["x"]
            label = row["z"]
            lst = re.split(r",|\?|!|。|，|？|！", text)

            tokens = []
            for x in lst:
                para_tokens = word_tokenize(x)
                tokens.append(para_tokens)
                for token in para_tokens:
                    word_counter[token] += 1

            example = {"context_tokens": tokens,
                       "y": label,
                       "id": total}

            examples.append(example)
            eval_examples[str(total)] = example

    random.shuffle(examples)

    print("{} examples in {}".format(len(examples), filename))
    return examples, eval_examples


def get_embedding(counter, data_type, limit=-1, emb_file=None,
                  vec_size=None, token2idx_dict=None):
    print("Generating {} embedding...{}".format(data_type, emb_file))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert vec_size is not None
        with codecs.open(emb_file, "r", encoding="utf-8") as fh:
            for line in fh:
                array = line.strip().split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.01) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(
        embedding_dict.keys(), 2)} if token2idx_dict is None else token2idx_dict
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def build_features(examples, out_file, word2idx_dict):
    print("build_features...{}".format(out_file))
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    total_ = 0
    meta = {}
    for example in examples:
        total_ += 1

        #if filter_func(example):
        #    continue

        total += 1
        context_idxs = np.zeros([para_max_num, para_max_length], dtype=np.int32)
        y = np.zeros([num_classes], dtype=np.float32)

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        for i, tokens in enumerate(example["context_tokens"]):
            if i < para_max_num:
                for j, token in enumerate(tokens):
                    if j < para_max_length:
                        context_idxs[i,j] = _get_word(token)

        label = example["y"]
        if label < 0:
            print("label error", label)
            continue

        y[int(label)] = 1.0

        record = tf.train.Example(features=tf.train.Features(feature={
                                  "context_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
                                  "y": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y.tostring()])),
                                  "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))
                                  }))
        writer.write(record.SerializeToString())
    print("Build {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    writer.close()
    return meta


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def prepro():
    word_counter = Counter()
    train_examples, train_eval = process_file(
        train_file, word_counter)
    dev_examples, dev_eval = process_file(
        dev_file, word_counter)

    word2idx_dict = None
    word_emb_mat, word2idx_dict = get_embedding(word_counter, "word",
                                                emb_file=glove_word_file,
                                                vec_size=glove_dim,
                                                token2idx_dict=word2idx_dict)

    train_meta = build_features(train_examples,
                                train_record_file,
                                word2idx_dict)
    dev_meta = build_features(dev_examples,
                              dev_record_file,
                              word2idx_dict)

    save(word_emb_file, word_emb_mat, message="word embedding")
    save(train_meta_file, train_meta, message="train meta")
    save(dev_meta_file, dev_meta, message="dev meta")
    save(word2idx_file, word2idx_dict, message="word2idx")
    print("done...")


if __name__ == "__main__":
    print("ok")

    ## 输入文件
    train_file = "../data/forum/train.dat"
    dev_file = "../data/forum/dev.dat"
    glove_word_file = "../data/forum/glove.forum.char.128.txt"

    ## 输出文件
    train_record_file = "../data/forum/train.tfrecords"
    dev_record_file = "../data/forum/dev.tfrecords"
    train_meta_file = "../data/forum/train_meta.json"
    dev_meta_file = "../data/forum/dev_meta.json"
    word_emb_file = "../data/forum/word_emb.json"
    word2idx_file = "../data/forum/word2idx.json"

    ## 参数
    para_max_num = 7
    para_max_length = 18
    num_classes = 2
    glove_dim = 128

    prepro()
