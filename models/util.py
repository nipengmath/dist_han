import tensorflow as tf
import numpy as np
import re
from collections import Counter
import string


def to_unicode(s):
    return s.decode("utf-8") if isinstance(s, str) else s


def get_record_parser(config, is_test=False):
    def parse(example):
        para_max_num = config.para_max_num
        para_max_length = config.para_max_length
        features = tf.parse_single_example(example,
                                           features={
                                               "context_idxs": tf.FixedLenFeature([], tf.string),
                                               "y": tf.FixedLenFeature([], tf.string),
                                               "id": tf.FixedLenFeature([], tf.int64)
                                           })
        context_idxs = tf.reshape(tf.decode_raw(
            features["context_idxs"], tf.int32), [para_max_num, para_max_length])
        y = tf.reshape(tf.decode_raw(
            features["y"], tf.float32), [config.num_classes])
        _id = features["id"]
        return context_idxs, y, _id
    return parse


def get_batch_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).shuffle(config.capacity).repeat()
    if config.is_bucket:
        buckets = [tf.constant(num) for num in range(*config.bucket_range)]

        def key_func(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id):
            c_len = tf.reduce_sum(
                tf.cast(tf.cast(context_idxs, tf.bool), tf.int32))
            buckets_min = [np.iinfo(np.int32).min] + buckets
            buckets_max = buckets + [np.iinfo(np.int32).max]
            conditions_c = tf.logical_and(
                tf.less(buckets_min, c_len), tf.less_equal(c_len, buckets_max))
            bucket_id = tf.reduce_min(tf.where(conditions_c))
            return bucket_id

        def reduce_func(key, elements):
            return elements.batch(config.batch_size)

        dataset = dataset.apply(tf.contrib.data.group_by_window(
            key_func, reduce_func, window_size=5 * config.batch_size)).shuffle(len(buckets) * 25)
    else:
        dataset = dataset.batch(config.batch_size)

    return dataset


def get_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).repeat().batch(config.batch_size)
    return dataset


# def evaluate(eval_file, answer_dict):
#     f1 = exact_match = total = 0
#     for key, value in answer_dict.items():
#         total += 1
#         ground_truths = eval_file[key]["answers"]
#         prediction = value
#         exact_match += metric_max_over_ground_truths(
#             exact_match_score, prediction, ground_truths)
#         f1 += metric_max_over_ground_truths(f1_score,
#                                             prediction, ground_truths)
#     exact_match = 100.0 * exact_match / total
#     f1 = 100.0 * f1 / total
#     return {'exact_match': exact_match, 'f1': f1}


# def normalize_answer(s):

#     def remove_articles(text):
#         return re.sub(r'\b(a|an|the)\b', ' ', text)

#     def white_space_fix(text):
#         return ' '.join(text.split())

#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return ''.join(ch for ch in text if ch not in exclude)

#     def lower(text):
#         return text.lower()

#     return white_space_fix(remove_articles(remove_punc(lower(s))))


# def f1_score(prediction, ground_truth):
#     prediction_tokens = normalize_answer(prediction).split()
#     ground_truth_tokens = normalize_answer(ground_truth).split()
#     common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
#     num_same = sum(common.values())
#     if num_same == 0:
#         return 0
#     precision = 1.0 * num_same / len(prediction_tokens)
#     recall = 1.0 * num_same / len(ground_truth_tokens)
#     f1 = (2 * precision * recall) / (precision + recall)
#     return f1


# def exact_match_score(prediction, ground_truth):
#     return (normalize_answer(prediction) == normalize_answer(ground_truth))


# def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
#     scores_for_ground_truths = []
#     for ground_truth in ground_truths:
#         score = metric_fn(prediction, ground_truth)
#         scores_for_ground_truths.append(score)
#     return max(scores_for_ground_truths)
