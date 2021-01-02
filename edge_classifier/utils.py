import datetime
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import time
import random
import os
from sklearn.metrics import roc_auc_score, accuracy_score


# =========== set seed ==============
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.set_random_seed(seed)

set_seed(42)


def label_neg_samp(uni_users, support_dict, item_array, neg_rate, pos_dict=None, bpr=False):
    """
    Generate pos and neg samples for user.

    param:
        uid - uid is from whole user set.
        dict - {uid: array[items]}
        item_array - sample item in this array.
        neg_rate - n_neg_items every pos_items

    return:
        ret_array - [n_samples, [uid, iid, label]] or [n_samples, [uid, pos, neg]]
    """
    ret_array = []
    for uid in uni_users:
        # pos sampling
        pos_neigh = support_dict[uid]
        pos_num = len(pos_neigh)
        # neg sampling
        if pos_dict:
            pos_set = np.hstack([pos_dict[uid], pos_neigh])
        else:
            pos_set = pos_neigh
        neg_item = []
        neg_num = pos_num * neg_rate
        while len(neg_item) < neg_num:
            neg_item = np.random.choice(item_array, 2 * neg_num)
            neg_item = neg_item[[i not in pos_set for i in neg_item]]  # "i not in pos_set" returns a bool value.

        if not bpr:
            ret_arr = np.zeros((pos_num * (neg_rate + 1), 3)).astype(np.int)
            ret_arr[:, 0] = uid
            ret_arr[:pos_num, 1] = pos_neigh
            ret_arr[pos_num:, 1] = neg_item[:pos_num * neg_rate]
            ret_arr[:pos_num, 2] = 1
            ret_arr[pos_num:, 2] = 0
        else:
            ret_arr = np.zeros(neg_num, 3)
            ret_arr[:, 0] = uid
            ret_arr[:, 1] = np.tile(pos_neigh.reshape(-1, 1), [1, neg_rate]).reshape(-1)
            ret_arr[:, 2] = neg_item[:neg_num]

        ret_array.append(ret_arr)
    return np.vstack(ret_array)


def sigmoid(array):
    return 1 / (1 + np.exp(-array))


def batch_eval(_sess, tf_eval, eval_feed_dict, metric,
               cold_features, warm_embeddings, has_warm_feature=False):
    """
    given EvalData and DropoutNet compute graph in TensorFlow, runs batch evaluation
    param:
        _sess: tf session
        tf_eval: the evaluate output symbol in tf
        eval_feed_dict: method to parse tf, pick from EvalData method
        eval_data: EvalData instance
    """

    # 在测试集上得到预测结果 user-item
    tf.local_variables_initializer().run()
    root_nodes = metric[:, 0]
    warm_nodes = metric[:, 1]
    eval_root_features = cold_features[root_nodes, :]
    eval_warm_embeddings = warm_embeddings[warm_nodes, :]
    eval_warm_features = cold_features[warm_nodes, :] if has_warm_feature else None
    eval_preds = _sess.run(tf_eval, feed_dict=eval_feed_dict(root_feature=eval_root_features,
                                                             warm_embedding=eval_warm_embeddings,
                                                             warm_feature=eval_warm_features))
    y_true = metric[:, 2]
    eval_preds = sigmoid(eval_preds)
    auc = roc_auc_score(y_true, eval_preds)
    eval_preds = np.where(eval_preds > 0.5, 1, 0).astype(np.int)
    acc = accuracy_score(y_true, eval_preds)

    return [auc, acc]


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
        print('[{0:s}] {1:s} [{2:s}]'.format(self._name, message, timer._format(elapsed)))
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