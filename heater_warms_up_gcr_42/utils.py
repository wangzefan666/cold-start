import time
import datetime
import numpy as np
import tensorflow as tf
import random
from sklearn.metrics import roc_auc_score
import os


# =========== set seed ==============
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.set_random_seed(seed)


set_seed(42)


# =========== utils ===================
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


# ==============================================
# =============== metric =======================
# ==============================================
def sigmoid(array):
    return 1 / (1 + np.exp(-array))


def batch_eval(_sess, tf_eval, eval_feed_dict, eval_data, U_pref, V_pref, u_fake_pref=None, v_fake_pref=None,
               excluded_dict=None, U_content=None, V_content=None, metric=None, warm=False, val=False):
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
    V_content = V_content.todense() if V_content is not None else None
    v_fake_pref = v_fake_pref if v_fake_pref is not None else None
    eval_preds, arg_eval_preds = [], []
    for (start, end) in eval_data.eval_batch:
        batch_user = eval_data.test_users[start:end]
        batch_u_pref = U_pref[batch_user]
        batch_u_cont = U_content[batch_user].todense() if U_content is not None else None
        batch_u_fake_pref = u_fake_pref[batch_user] if u_fake_pref is not None else None
        # scores
        eval_preds_batch = _sess.run(tf_eval, feed_dict=eval_feed_dict(batch_u_pref, V_pref, eval_data,
                                                                       batch_u_cont, V_content,
                                                                       batch_u_fake_pref, v_fake_pref,
                                                                       warm))
        eval_preds.append(eval_preds_batch)
    eval_preds = np.concatenate(eval_preds)

    # mask pos in train
    exclude_index = []
    exclude_items = []
    for uid in eval_data.test_users:
        iid_array = excluded_dict[uid]
        uid = eval_data.test_user_ids_map[uid]
        iid_array = np.setdiff1d(iid_array, np.array(eval_data.R_test_inf.rows[uid], dtype=np.int))
        exclude_index.extend([uid] * len(iid_array))
        exclude_items.extend(iid_array)
    eval_preds = sigmoid(eval_preds)
    eval_preds[exclude_index, exclude_items] = -(1 << 10)

    # auc
    auc_list, auc_weight = [], []
    for row in range(eval_preds.shape[0]):
        eval_p = eval_preds[row]
        y_true = eval_data.R_test_inf[row, :].todense().A.flatten()
        y_true = y_true[eval_p >= 0]
        preds = eval_p[eval_p >= 0]
        a = roc_auc_score(y_true, preds)
        auc_list.append(a)
        auc_weight.append(len(y_true))
    auc_list = np.array(auc_list)
    auc_weight = np.array(auc_weight)
    ret = np.multiply(auc_list, auc_weight).sum() / auc_weight.sum()
    if val:
        return ret

    at_k = 10  # at_k
    arg_eval_preds = np.argsort(-eval_preds)
    hr_ndcg = list(map(lambda x: hr_ndcg_at_k(x, arg_eval_preds, at_k), metric))
    hr_ndcg = np.array(hr_ndcg, dtype=np.float).mean(axis=0)
    ret = np.hstack([ret, hr_ndcg])

    return ret


# prepare idcg
idcg_array = np.arange(30000) + 1
idcg_array = 1 / np.log2(idcg_array + 1)


def hr_ndcg_at_k(sample, rating, at_k):
    """
    leave-one-out metric in cold start
    input:
        sample - (uid, one pos, k neg)
        rating - rank k item index
    """
    pos_item = sample[1]
    neg_items = sample[2:]
    # items rank before pos_item
    sorted_items = rating[sample[0]]
    end = sorted_items.tolist().index(pos_item)
    x = np.zeros_like(sorted_items)
    x[sorted_items[:end]] = 1
    # interactions
    y = np.zeros_like(sorted_items)
    y[neg_items] = 1
    # leave-one-out @k - how many neg-sample items rank before pos_item
    z = np.multiply(y, x)
    n_items_rank_before = np.sum(z)
    # hr
    hr = 0 if n_items_rank_before >= at_k else 1
    # ndcg
    ndcg = 0 if n_items_rank_before >= at_k else idcg_array[n_items_rank_before]

    return [hr, ndcg]


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
    user_neg = np.tile(pos_user_array, neg).reshape((-1))
    item_pos = pos_item_array.reshape((-1))
    item_neg = np.random.choice(item_warm, size=neg * pos_user_array.shape[0], replace=True).reshape((-1))
    target_pos = np.ones_like(item_pos)
    target_neg = np.zeros_like(item_neg)
    return np.concatenate((user_pos, user_neg)), np.concatenate((item_pos, item_neg)), np.concatenate(
        (target_pos, target_neg))


def at_k_sampling(uni_users, support_dict, at_k, item_array, complement_dict=None):
    """
    leave-one-out metric.
    param:
        uni_users - unique users in training data

    return:
        (uid, pos_item, k_neg_items) * n_interactions
    """
    at_k_list = []
    for uid in uni_users:
        pos_neigh = support_dict[uid]  # pos sample

        if complement_dict is not None:
            pos_set = np.hstack([complement_dict[uid], pos_neigh])  # excluded item
        else:
            pos_set = pos_neigh

        for n in pos_neigh:
            at_k_samp = np.zeros(2 + at_k)
            # a pos sample
            at_k_samp[0] = uid
            at_k_samp[1] = n
            # n neg samples
            neg_item = []
            while len(neg_item) < at_k:
                neg_item = np.random.choice(item_array, 2 * at_k)
                neg_item = neg_item[[i not in pos_set for i in neg_item]]  # "i not in pos_set" returns a bool value.
            neg_item = neg_item[:at_k]
            at_k_samp[2:] = neg_item
            at_k_list.append(at_k_samp)

    return np.stack(at_k_list).astype(np.int)


def label_neg_samp(uni_users, support_dict, neg_rate, item_array, cold_items, exclude_dict=None):
    """
    Generate pos and neg samples for user.

    param:
        uid - uid is from whole user set.
        dict - {uid: array[items]}
        item_array - sample item in this array.
        neg_rate - n_neg_items every pos_items

    return:
        ret_array - [n_samples, 4]  4 refers to uid, iid, label, is_cold_item
    """
    ret_array = []
    for uid in uni_users:
        # pos sampling
        pos_neigh = support_dict[uid]
        pos_num = len(pos_neigh)
        # neg sampling
        if exclude_dict is not None:
            pos_set = np.hstack([exclude_dict[uid], pos_neigh])  # excluded item
        else:
            pos_set = pos_neigh

        neg_item = []
        neg_num = pos_num * neg_rate
        while len(neg_item) < neg_num:
            neg_item = np.random.choice(item_array, 2 * neg_num, replace=False)  # 不放回采样 为了进行集合运算
            neg_item = neg_item[[i not in pos_set for i in neg_item]]  # "i not in pos_set" returns a bool value.
        neg_item = neg_item[:neg_num]

        all_num = pos_num + neg_num
        all_item = np.hstack([pos_neigh, neg_item])
        ret_part = np.zeros((all_num, 4)).astype(np.int)  # user, item, label, is_cold_item
        ret_part[:, 0] = np.tile(uid, all_num)
        ret_part[:, 1] = all_item
        ret_part[:pos_num, 2] = 1
        ret_part[pos_num:, 2] = 0
        ret_part[:, 3] = np.array([1 if item in cold_items else 0 for item in all_item], dtype=np.int)
        ret_array.append(ret_part)

    return np.vstack(ret_array)
