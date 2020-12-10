import time
import datetime
import numpy as np
import scipy
import tensorflow as tf
from sklearn import preprocessing as prep
import random
import scipy.sparse as sp
from sklearn.metrics import roc_curve, auc


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


class timer(object):
    def __init__(self, name='default'):
        """
        timer object to record running time of functions, not for micro-benchmarking
        usage is:
            $ timer = utils.timer('name').tic()
            $ timer.toc('process A').tic()

        :param name: label for the timer
        """
        self._start_time = None
        self._name = name
        self.tic()

    def tic(self):
        self._start_time = time.time()
        return self

    def toc(self, message):
        elapsed = time.time() - self._start_time
        message = '' if message is None else message
        print('[{0:s}] {1:s} elapsed [{2:s}]'.format(self._name, message, timer._format(elapsed)))
        return self

    def reset(self):
        self._start_time = None
        return self

    @staticmethod
    def _format(s):
        delta = datetime.timedelta(seconds=s)
        d = datetime.datetime(1, 1, 1) + delta
        s = ''
        if (d.day - 1) > 0:
            s = s + '{:d} days'.format(d.day - 1)
        if d.hour > 0:
            s = s + '{:d} hr'.format(d.hour)
        if d.minute > 0:
            s = s + '{:d} min'.format(d.minute)
        s = s + '{:d} s'.format(d.second)
        return s


def batch(iterable, _n=1, drop=True):
    """
    returns batched version of some iterable
    :param iterable: iterable object as input
    :param _n: batch size
    :param drop: if true, drop extra if batch size does not divide evenly,
        otherwise keep them (last batch might be shorter)
    :return: batched version of iterable
    """
    it_len = len(iterable)
    for ndx in range(0, it_len, _n):
        if ndx + _n < it_len:
            yield iterable[ndx:ndx + _n]
        elif drop is False:
            yield iterable[ndx:it_len]


def tfidf(R):
    row = R.shape[0]
    col = R.shape[1]
    Rbin = R.copy()
    Rbin[Rbin != 0] = 1.0
    R = R + Rbin
    tf = R.copy()
    tf.data = np.log(tf.data)
    idf = np.sum(Rbin, 0)
    idf = np.log(row / (1 + idf))
    idf = scipy.sparse.spdiags(idf, 0, col, col)
    return tf * idf


def standardize(x):
    """
    takes sparse input and compute standardized version

    Note:
        cap at 5 std

    :param x: 2D scipy sparse data array to standardize (column-wise), must support row indexing
    :return: the object to perform scale (stores mean/std) for inference, as well as the scaled x
    """
    x_nzrow = x.any(axis=1)
    scaler = prep.StandardScaler().fit(x[x_nzrow, :])
    x_scaled = np.copy(x)
    x_scaled[x_nzrow, :] = scaler.transform(x_scaled[x_nzrow, :])
    x_scaled[x_scaled > 5] = 5
    x_scaled[x_scaled < -5] = -5
    x_scaled[np.absolute(x_scaled) < 1e-5] = 0
    return scaler, x_scaled


def standardize_2(x):
    """
    takes sparse input and compute standardized version

    Note:
        cap at 1 std

    :param x: 2D scipy sparse data array to standardize (column-wise), must support row indexing
    :return: the object to perform scale (stores mean/std) for inference, as well as the scaled x
    """
    x_nzrow = x.any(axis=1)
    scaler = prep.StandardScaler().fit(x[x_nzrow, :])
    x_scaled = np.copy(x)
    x_scaled[x_nzrow, :] = scaler.transform(x_scaled[x_nzrow, :])
    x_scaled[x_scaled > 1] = 1
    x_scaled[x_scaled < -1] = -1
    x_scaled[np.absolute(x_scaled) < 1e-5] = 0
    return scaler, x_scaled


def standardize_3(x):
    """
    takes sparse input and compute standardized version

    Note:
        cap at 2 std

    :param x: 2D scipy sparse data array to standardize (column-wise), must support row indexing
    :return: the object to perform scale (stores mean/std) for inference, as well as the scaled x
    """
    x_nzrow = x.any(axis=1)
    scaler = prep.StandardScaler().fit(x[x_nzrow, :])
    x_scaled = np.copy(x)
    x_scaled[x_nzrow, :] = scaler.transform(x_scaled[x_nzrow, :])
    x_scaled[x_scaled > 2] = 2
    x_scaled[x_scaled < -2] = -2
    x_scaled[np.absolute(x_scaled) < 1e-5] = 0
    return scaler, x_scaled


# prepare idcg
rank = 20000
idcg_array = np.arange(rank) + 1
idcg_array = 1 / np.log2(idcg_array + 1)


def map_to_test(ndcg_group, eval_data):
    """
    param:
        interaction: array(@n+1, 3)

    return:
        user - uid
        pos_item - leave-one-out pos item
        neg_items - left items except pos item
    """
    user = ndcg_group[:, 0]
    target_user = user[0]
    assert len(user[user == target_user]) == len(user)
    user = eval_data.test_user_ids_map[target_user]

    items = list(map(eval_data.test_item_ids_map.get, ndcg_group[:, 1]))
    pos_item = items[0]
    neg_items = items[1:]
    return user, pos_item, neg_items


def sigmoid(array):
    return 1 / (1 + np.exp(-array))


def batch_eval_auc(_sess, tf_eval, eval_feed_dict, eval_data, warm=False):
    eval_preds = []
    for (bh, (eval_start, eval_stop)) in enumerate(eval_data.eval_batch):
        eval_preds_batch = _sess.run(tf_eval, feed_dict=eval_feed_dict(bh, eval_start, eval_stop, eval_data, warm))
        eval_preds.append(eval_preds_batch)
    eval_preds = np.concatenate(eval_preds)
    tf.local_variables_initializer().run()  # 为啥这里要加这一句？

    # auc
    auc_list = []
    for row in range(eval_preds.shape[0]):
        y_true = eval_data.R_test_inf[row, :].todense().A.flatten()
        preds = sigmoid(eval_preds[row])
        fpr, tpr, thresholds = roc_curve(y_true, preds, pos_label=1)
        a = auc(fpr, tpr)
        auc_list.append(a)
    return np.mean(auc_list)


def batch_eval(_sess, tf_eval, eval_feed_dict, eval_data, metric, warm=False):
    """
    given EvalData and DropoutNet compute graph in TensorFlow, runs batch evaluation
    param:
        _sess: tf session
        tf_eval: the evaluate output symbol in tf
        eval_feed_dict: method to parse tf, pick from EvalData method
        recall_k: list of thresholds to compute recall at (information retrieval recall)
        eval_data: EvalData instance
        recall array at thresholds matching recall_k
    """

    # 在测试集上得到预测结果 user-item
    eval_preds, arg_eval_preds = [], []
    for (bh, (eval_start, eval_stop)) in enumerate(eval_data.eval_batch):
        eval_preds_batch = _sess.run(tf_eval, feed_dict=eval_feed_dict(bh, eval_start, eval_stop, eval_data, warm))
        eval_preds.append(eval_preds_batch)
        arg_eval_preds_batch = np.argsort(-eval_preds_batch)  # sort
        arg_eval_preds.append(arg_eval_preds_batch)

    eval_preds = np.concatenate(eval_preds)
    arg_eval_preds = np.concatenate(arg_eval_preds)
    tf.local_variables_initializer().run()  # 为啥这里要加这一句？
    ret = []

    # auc
    auc_list = []
    for row in range(eval_preds.shape[0]):
        y_true = eval_data.R_test_inf[row, :].todense().A.flatten()
        preds = sigmoid(eval_preds[row])
        fpr, tpr, thresholds = roc_curve(y_true, preds, pos_label=1)
        a = auc(fpr, tpr)
        auc_list.append(a)
    ret.append(np.mean(auc_list))

    at_k = 10  # at_k
    hr_k_list, ndcg_k_list = [], []
    for ndcg_group in metric:  # metric (n_nei * n_user, @k+1, 3)
        # mapped
        user, pos_item, neg_items = map_to_test(ndcg_group, eval_data)
        # prediction
        preds_k = arg_eval_preds[user]
        # items rank before pos_item
        end = preds_k.tolist().index(pos_item)
        x = np.zeros_like(preds_k)
        x[preds_k[:end]] = 1
        # interactions
        y = np.zeros_like(preds_k)
        y[neg_items] = 1
        # leave-one-out @k - how many neg-sample items rank before pos_item
        z = np.multiply(y, x)
        n_items_rank_before = np.sum(z)
        # hr
        hr = 0 if n_items_rank_before >= at_k else 1
        hr_k_list.append(hr)
        # ndcg
        ndcg = 0 if n_items_rank_before >= at_k else idcg_array[n_items_rank_before]
        ndcg_k_list.append(ndcg)
    ret.append(np.mean(hr_k_list))
    ret.append(np.mean(ndcg_k_list))

    return ret


def negative_sampling(pos_user_array, pos_item_array, neg, item_warm):
    """
    Args:
        pos_user_array: users in train interactions
        pos_item_array: items in train interactions
        neg: num of negative samples
        item_warm: train item set

    Returns:
        user: concat pos users and neg ones
        item: concat pos item and neg ones
        target: scores of both pos interactions and neg ones
    """
    user_pos = pos_user_array.reshape((-1))
    # np.tile(seq, n): repeat seq for n times
    user_neg = np.tile(pos_user_array, neg).reshape((-1))
    item_pos = pos_item_array.reshape((-1))
    # replace: whether element can be chosen more than once
    # ？？为什么 neg item 是直接在 warm item 里面随机抽取，是因为 adj 足够稀疏吗
    item_neg = np.random.choice(item_warm, size=neg * pos_user_array.shape[0], replace=True).reshape((-1))
    target_pos = np.ones_like(item_pos)
    target_neg = np.zeros_like(item_neg)
    return np.concatenate((user_pos, user_neg)), np.concatenate((item_pos, item_neg)), np.concatenate(
        (target_pos, target_neg))
