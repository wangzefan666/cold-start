import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score


def sigmoid(array):
    return 1 / (1 + np.exp(-array))


def binary_search(nums, target):
    start = 0
    end = len(nums) - 1
    while start <= end:
        mid = (start + end) // 2
        if nums[mid] == target:
            return 1
        elif nums[mid] < target:
            start = mid + 1
        else:
            end = mid - 1
    return 0


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


def get_sparse_UINet(rows, cols, n_users, m_items):
    values = np.ones((len(rows, )))
    adj = sp.csr_matrix((values, (rows, cols)), shape=(n_users, m_items), dtype=np.int)
    return adj


def get_sparse_graph(UserItemNet, n_users, m_items):
    """
    param:
        UserItemNet - sparse matrix

    return:
        norm_adj - sparse Laplacian matrix
    """
    # build graph
    n_nodes = n_users + m_items
    adj_mat = sp.dok_matrix((n_nodes, n_nodes), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = UserItemNet.tolil()
    adj_mat[:n_users, n_users:] = R
    adj_mat[n_users:, :n_users] = R.T
    adj_mat = adj_mat.todok()
    # build hat(A)
    rowsum = np.array(adj_mat.sum(axis=1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    # build Laplacian matrix
    norm_adj = d_mat.dot(adj_mat)
    norm_adj = norm_adj.dot(d_mat)
    norm_adj = norm_adj.tocsr()
    return norm_adj


def df_get_neighbors(input_df, obj='user', max_nei=0):
    """
    Get users' neighboring items.
    return:
        d - {user: array[items]}
    """
    group = input_df.groupby(obj)
    opp_dict = {'user': 'item', 'item': 'user'}
    if max_nei:
        d = {}
        for g in group:
            value = g[1][opp_dict[obj]].values
            d[g[0]] = value if len(value) <= max_nei else np.random.choice(value, max_nei, replace=False)
    else:
        d = {g[0]: g[1][opp_dict[obj]].values for g in group}

    return d


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


def bpr_neg_samp_large_scale(uni_users, n_users, support_dict, item_array, complement_dict=None):
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
        # samples
        samp_arr = [user, pos_item]
        ret.append(samp_arr)
    ret = np.array(ret, dtype=np.int)  # (n, 2)
    # **neg**
    neg_items = np.random.choice(item_array, n_users).reshape(-1, 1)
    ret = np.hstack([ret, neg_items])
    return ret


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
        pos_neigh = support_dict[uid]
        pos_num = len(pos_neigh)
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


def label_neg_samp_large_scale(user_list, item_list, item_array, neg_rate):
    """
    Generate pos and neg samples.

    return:
        ret_array - [n_samples, [uid, iid, label]]
    """
    users = np.tile(user_list, neg_rate + 1).reshape(-1, 1)
    pos_items = item_list
    neg_items = np.random.choice(item_array, neg_rate * len(pos_items))
    items = np.hstack([pos_items, neg_items]).reshape(-1, 1)
    pos_targets = np.ones(len(pos_items))
    neg_targets = np.zeros(len(neg_items))
    targets = np.hstack([pos_targets, neg_targets]).reshape(-1, 1)
    all_samples = np.hstack(users, items, targets).astype(np.int)
    return all_samples


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
            pos_set = np.hstack(complement_dict[uid], pos_neigh)  # excluded item
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
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
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


# prepare idcg
rank = 20000
idcg_array = np.arange(rank) + 1
idcg_array = 1 / np.log2(idcg_array + 1)


def hr_ndcg_at_k(sample, sorted_items, at_k):
    """
    leave-one-out metric in cold start
    input:
        sample - (uid, one pos, k neg)
        sorted_items - rank k item index
    """
    pos_item = sample[1]
    neg_items = sample[2:]
    # items rank before pos_item
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
