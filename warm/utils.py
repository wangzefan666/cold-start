import scipy.sparse as sp
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import random
import os
from warm.lgn.parse import args


# ====================Sample==============================
# =========================================================

def label_neg_samp(uni_users, support_dict, item_array, neg_rate, complement_dict=None):
    """
    Generate pos and neg samples for user.

    param:
        uid - uid is from whole user set.
        dict - {uid: array[items]}
        item_array - sample item in this array.
        neg_rate - n_neg_items every pos_items

    return:
        ret_array - [n_samples, [uid, iid, label]]
    """
    ret_array = []
    for uid in uni_users:
        # pos sampling
        pos_neigh = support_dict.get(uid, [])
        pos_num = len(pos_neigh)
        if pos_num < 1:
            continue

        # neg sampling
        if complement_dict is not None:
            pos_set = np.hstack([complement_dict[uid], pos_neigh])  # excluded item
        else:
            pos_set = pos_neigh
        neg_item = []
        while len(neg_item) < pos_num * neg_rate:
            neg_item = np.random.choice(item_array, 2 * neg_rate * pos_num)
            neg_item = neg_item[[i not in pos_set for i in neg_item]]  # "i not in pos_set" returns a bool value.

        ret_arr = np.zeros((pos_num * (neg_rate + 1), 3)).astype(np.int)
        ret_arr[:, 0] = uid
        ret_arr[:pos_num, 1] = pos_neigh
        ret_arr[pos_num:, 1] = neg_item[:pos_num * neg_rate]
        ret_arr[:pos_num, 2] = 1
        ret_arr[pos_num:, 2] = 0

        ret_array.append(ret_arr)
    return np.vstack(ret_array)


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


def ndcg_sampling(uid, support_dict, neg_num, item_array, complement_dict=None):
    """
    Generate ndcg samples for neighboring an item
    param:
        uid - a user in testing set

    return:
        ndcg_array - array(n_neighbors, ndcg_k + 1, 3)
    """
    pos_neigh = support_dict[uid]  # pos sample
    if complement_dict is not None:
        pos_set = np.hstack([complement_dict[uid], pos_neigh])  # excluded item
    else:
        pos_set = pos_neigh
    ndcg_list = []
    for n in pos_neigh:
        # a pos sample
        pos_data = np.array([uid, n, 1]).reshape(1, 3)
        # n neg samples
        neg_item = []
        while len(neg_item) < neg_num:
            neg_item = np.random.choice(item_array, 2 * neg_num)
            neg_item = neg_item[[i not in pos_set for i in neg_item]]  # "i not in pos_set" returns a bool value.
        neg_item = neg_item[:neg_num]
        neg_data = np.hstack([
            np.array([uid] * len(neg_item)).reshape([-1, 1]),
            neg_item.reshape([-1, 1]),
            np.array([0] * len(neg_item)).reshape([-1, 1])
        ])
        # ndcg samples of a neighboring item
        label_data = np.vstack([pos_data, neg_data])
        ndcg_list.append(label_data)

    ndcg_array = np.stack(ndcg_list)
    return ndcg_array


# ====================Utils==============================
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def comput_bi_adj(records, n_user, n_item):
    """
    param:
        records - array, [n_records, user-item]

    """
    cols = records[:, 0]
    rows = records[:, 1] + n_user
    values = np.ones(len(cols))
    n_node = n_user + n_item

    adj = sp.csr_matrix((values, (rows, cols)), shape=(n_node, n_node))
    adj = adj + adj.T
    return adj

