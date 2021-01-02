import scipy.sparse as sp
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import random
import os
from warm.lgn.parse import args


# ====================Sample==============================
# =========================================================
def bpr_neg_samp(uni_users, n_users, support_dict, item_array, complement_dict=None):
    """
    param:
        uni_users - unique users in training data
        dict - {uid: array[items]}
        n_users - sample n users
        neg_num - n of sample pairs for a user.
        item_array - sample item in this array.

    return:
        ret_array - [uid pos_iid neg_iid] * n_records
    """
    users = np.random.choice(uni_users, n_users)
    ret = []
    for user in users:
        # pos
        pos_items = support_dict[user]
        pos_item = np.random.choice(pos_items, 1)
        # neg
        if complement_dict is not None:
            pos_set = np.hstack([complement_dict[user], pos_items])  # excluded item
        else:
            pos_set = pos_items
        while True:
            neg_item = np.random.choice(item_array, 1)
            if neg_item not in pos_set:
                break
        # samples
        samp_arr = [user, pos_item, neg_item]
        ret.append(samp_arr)

    return np.array(ret, dtype=np.int)


# ====================Utils==============================
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def getFileName():
    weight_dir = './model_save/'
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir, exist_ok=True)
    file = f"lgn-{args.dataset}-{args.n_layers}-{args.embed_dim}.pt"
    return os.path.join(weight_dir, file)


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', args.batch_size)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


# ====================Metrics==============================
# =========================================================

def AUC(all_item_scores, test_data):
    """
        design for a single user
    """
    # y_true
    r_all = np.zeros((len(all_item_scores),))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]  # exclude pos item in train
    # y_predict
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)


def test_batch(sorted_items, groundTrue):
    """
    Global metric in recommendation including pre, rec and ndcg at top k.
    input:
        groundTrue - [batch [groundTrue items]]. [groundTrue items]'s length is mutative.
        sorted_items - array[batch [sorted top k items index]]
    return:
        sum of batch metric
    """
    rank_label = get_rank_label(groundTrue, sorted_items)
    ret = recal_precision_k(groundTrue, rank_label)
    pre = ret['precision']
    recall = ret['recall']
    ndcg = ndcg_k(groundTrue, rank_label)
    return pre, recall, ndcg


def get_rank_label(groundTrue, sorted_items):
    """
    Get rank k position label.

    test_data - [batch [groundTrue items]]
    pred_data - [batch [sorted top k items index]]
    """
    r = []
    for i in range(len(groundTrue)):
        gt = groundTrue[i]
        si = sorted_items[i]
        pred = list(map(lambda x: x in gt, si))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def recal_precision_k(groundTrue, rank_label):
    """
    test_data should be a list? Cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = rank_label.sum(axis=1)
    precis_n = rank_label.shape[1]
    recall_n = np.array([len(groundTrue[i]) for i in range(len(groundTrue))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def ndcg_k(groundTrue, rank_label):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(rank_label) == len(groundTrue)
    k = rank_label.shape[1]
    # idcg
    idcg = np.zeros((len(rank_label), k))
    for i, items in enumerate(groundTrue):
        length = k if k <= len(items) else len(items)
        idcg[i, :length] = 1
    idcg = np.sum(idcg * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    # dcg
    dcg = rank_label * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    # ndcg
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)
