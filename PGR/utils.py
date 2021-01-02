"""

"""
import scipy.sparse as sp
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import random
import os


def set_seed(seed, cuda=True):
    print('Set Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(42)

# ====================Sample==============================
# =========================================================

def label_auc(uni_users, support_dict, item_array, exclude_dict):
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
            
        # excluded samples
        excluded_neigh = exclude_dict[uid]
        excluded_num = len(excluded_neigh)

        # neg sampling
        neg_item = np.copy(item_array)
        neg_item = np.setdiff1d(neg_item, excluded_neigh)

        ret_arr = np.zeros((len(item_array) - excluded_num + pos_num, 3)).astype(np.int)
        ret_arr[:, 0] = uid
        ret_arr[:pos_num, 1] = pos_neigh
        ret_arr[pos_num:, 1] = neg_item[:len(item_array) - excluded_num]
        ret_arr[:pos_num, 2] = 1
        ret_arr[pos_num:, 2] = 0

        ret_array.append(ret_arr)
    return np.vstack(ret_array)


def label_neg_samp(uni_users, support_dict, item_array, neg_rate, exclude_dict=None):
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
        pos_neigh = support_dict[uid]
        pos_num = len(pos_neigh)
        # neg sampling
        if exclude_dict is not None:
            pos_set = exclude_dict[uid]  # excluded item
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


def ndcg_sampling(uid, support_dict, neg_num, item_array, exclude_dict=None):
    """
    Generate ndcg samples for neighboring an item
    param:
        uid - a user in testing set

    return:
        ndcg_array - array(n_neighbors, ndcg_k + 1, 3)
    """
    pos_neigh = support_dict[uid]  # pos sample
    if exclude_dict is not None:
        pos_set = exclude_dict[uid]  # excluded item
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

