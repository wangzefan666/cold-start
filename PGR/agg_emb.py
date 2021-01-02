"""
Aggregate the stacked embedding together to constrain the space
"""
import os
import time
import random
import pickle
import argparse
import warnings
import numpy as np
import scipy.sparse as sp
from collections import Counter
from sklearn import cluster
from scipy.sparse.linalg import norm
from multiprocessing import Pool
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
import pandas as pd

# from ipdb import launch_ipdb_on_exception

warnings.filterwarnings('ignore')


def compute_adj(input_df):
    """
    generate user-item adjacent matrix.
    | 0      user-item |
    | item-user      0 |
    """
    cols = input_df.user.values
    rows = input_df.item.values + user_num
    if 'weight' in input_df.columns:
        values = input_df.weight.values
    else:
        values = np.ones(len(cols))

    adj = sp.csr_matrix((values, (rows, cols)), shape=(num_nodes, num_nodes))
    adj = adj + adj.T
    return adj


def set_seed(seed):
    print('Unfolder Set Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = 1 / rowsum
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def compute_adj_element(l):
    adj_map = NUM_NODES + np.zeros((l[1] - l[0], MAX_DEGREE), dtype=np.int)
    sub_adj = RAW_ADJ[l[0]: l[1]]
    for v in range(l[0], l[1]):
        neighbors = np.nonzero(sub_adj[v - l[0], :])[1]
        len_neighbors = len(neighbors)
        if len_neighbors > MAX_DEGREE:
            if SORTED:
                weight_sort = np.argsort(-ADJ_SUM[neighbors])
                neighbors = neighbors[weight_sort[:MAX_DEGREE]]
            else:
                neighbors = np.random.choice(neighbors, MAX_DEGREE, replace=False)
            adj_map[v - l[0]] = neighbors
        else:
            adj_map[v - l[0], :len_neighbors] = neighbors
    return adj_map


def compute_adjlist_parallel(sp_adj, max_degree, batch=50):
    global RAW_ADJ, MAX_DEGREE
    RAW_ADJ = sp_adj
    MAX_DEGREE = max_degree
    index_list = []
    for ind in range(0, NUM_NODES, batch):
        index_list.append([ind, min(ind + batch, NUM_NODES)])
    with Pool(N_JOBS) as pool:
        adj_list = pool.map(compute_adj_element, index_list)
    adj_list.append(NUM_NODES + np.zeros((1, MAX_DEGREE), dtype=np.int))
    adj_map = np.vstack(adj_list)
    return adj_map


def init(adj, feature_data, samp_pre_num, samp_num, samp_times, degree=2, max_degree=32,
         max_samp_nei=3, if_normalized=False, degree_normalized=False, if_self_loop=True, if_bagging=True, if_sort=True,
         weight='same', n_jobs=2, seed=42):
    """ Init the shared data for multiprocessing
        Max_neighbor denotes the max allowed sampling neighbor
        number for a single node in traj sampling.
        Preventing the large degree neighbors have higher effects.
    """
    t = time.time()
    global NUM_NODES, FEATURES, ADJ_TRAIN, NUM_FEATURES, MAX_SAMP_NEI, WEIGHT
    global IF_BAGGING, SAMP_PRE_NUM, SAMP_NUM, SAMP_TIMES, N_JOBS, DEGREE
    global SORTED, ADJ_SUM
    if if_self_loop:
        adj = adj + sp.eye(adj.shape[0])
    set_seed(seed)
    NUM_FEATURES = feature_data.shape[1]
    NUM_NODES = feature_data.shape[0]
    SAMP_PRE_NUM = samp_pre_num
    SAMP_NUM = samp_num
    SAMP_TIMES = samp_times
    DEGREE = degree
    MAX_SAMP_NEI = max_samp_nei
    IF_BAGGING = if_bagging
    N_JOBS = n_jobs
    WEIGHT = weight
    if degree_normalized:
        adj_sum = np.sum(adj, axis=-1)
        feature_data = feature_data / np.array(adj_sum)
    null_feature = np.zeros((1, NUM_FEATURES))
    feature_data = np.vstack([feature_data, null_feature])

    if if_normalized:
        feature_data = normalize(feature_data, axis=1, norm='l2')
    FEATURES = feature_data
    if if_sort:
        SORTED = True
        ADJ_SUM = np.array(np.sum(adj, axis=1)).reshape([-1])
    else:
        SORTED = False
    ADJ_TRAIN = compute_adjlist_parallel(adj, max_degree)
    if not IF_BAGGING:
        SAMP_TIMES = 1
    print('Init in %.2f s' % (time.time() - t))


def get_traj_child(parent, sample_num=0):
    '''
    If sample_num == 0 return all the neighbors
    '''

    traj_list = []
    for p in parent:
        neigh = np.unique(ADJ_TRAIN[p].reshape([-1]))
        if len(neigh) > 1:
            neigh = neigh[neigh != NUM_NODES]
        neigh = np.random.choice(neigh, min(MAX_SAMP_NEI, len(neigh)), replace=False)
        t_array = np.hstack(
            [p * np.ones((len(neigh), 1)).astype(np.int), neigh.reshape([-1, 1])])
        traj_list.append(t_array)
    traj_array = np.unique(np.vstack(traj_list), axis=0)
    if traj_array.shape[0] > 1:
        traj_array = traj_array[traj_array[:, -1] != NUM_NODES]
    if sample_num:
        traj_array = traj_array[
            np.random.choice(
                traj_array.shape[0], min(sample_num, traj_array.shape[0]), replace=False)]
    return traj_array


def get_gun_traj(idx):
    '''
    Get the trajectory set of a given node under the naive gun setting.
    '''
    traj_list = [np.array(idx), []]
    whole_trajs = np.unique(ADJ_TRAIN[idx])
    for _ in range(DEGREE - 1):
        whole_trajs = get_traj_child(whole_trajs, 0)
    traj_list[1] = [whole_trajs]
    return traj_list


def get_traj_emb(traj):
    '''
    Get and stack the embedding of a given trajset
    '''
    emb_center = FEATURES[traj[0]]
    if WEIGHT == 'rw':
        emb_traj = np.stack(
            list(map(lambda x: np.mean(FEATURES[x].reshape([-1, DEGREE * NUM_FEATURES]), axis=0), traj[1]))).reshape(
            [DEGREE, -1])
    elif WEIGHT == 'same':
        unique_traj = [
            [np.unique(_[:, __]) for __ in range(_.shape[1])]
            for _ in traj[1]]
        emb_traj = np.mean(
            [np.concatenate([np.mean(FEATURES[__], axis=0) for __ in _])
             for _ in unique_traj], axis=0).reshape(DEGREE, -1)
    else:
        raise ('Weight Paremeter %s not defined' % WEIGHT)

    emb = np.vstack(
        [emb_center,
         emb_traj])
    return emb


def rand_samp_traj(traj):
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    idxes = np.unique(traj_idx[:, 0].reshape([-1]))
    if len(idxes) > 1:
        idxes = idxes[idxes != cen_node]
    if len(idxes) >= int(SAMP_NUM ** 0.5):
        idxes = np.random.choice(idxes, int(SAMP_NUM ** 0.5), replace=False)
    else:
        extra_idxes = np.random.choice(idxes, int(SAMP_NUM ** 0.5) - len(idxes))
        idxes = np.hstack([idxes, extra_idxes])
    traj_list = []
    for idx in idxes:
        sub_traj = traj_idx[traj_idx[:, 0] == idx]
        if sub_traj.shape[0] < int(SAMP_NUM ** 0.5):
            sub_idx = np.array(list(range(sub_traj.shape[0])))
            extra_idx = np.random.choice(
                list(range(sub_traj.shape[0])), int(SAMP_NUM ** 0.5) - len(sub_idx))
            sub_idx = np.hstack([sub_idx, extra_idx])
            traj_list.append(sub_traj[sub_idx])
        else:
            sub_idx = np.random.choice(
                list(range(sub_traj.shape[0])),
                int(SAMP_NUM ** 0.5), replace=False)
            traj_list.append(sub_traj[sub_idx])
    #             print(sub_traj[traj_rank[-int(SAMP_NUM**0.5):]])
    traj_idx = np.vstack(traj_list)
    traj[1] = [traj_idx]
    return traj


def ord1_samp_traj(traj):
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    idxes = np.unique(traj_idx[:, 0].reshape([-1]))
    if len(idxes) > 1:
        idxes = idxes[idxes != cen_node]
    if len(idxes) >= int(SAMP_NUM ** 0.5):
        traj_sim = np.array(RAW_ADJ[cen_node, idxes].toarray())
        traj_sim = traj_sim.reshape([-1])
        traj_rank = np.argsort(traj_sim)
        idxes = idxes[traj_rank[-int(SAMP_NUM ** 0.5):]]
    else:
        extra_idxes = np.random.choice(idxes, int(SAMP_NUM ** 0.5) - len(idxes))
        idxes = np.hstack([idxes, extra_idxes])
    traj_list = []
    for idx in idxes:
        #         print(idx)
        sub_traj = traj_idx[traj_idx[:, 0] == idx]
        if sub_traj.shape[0] < int(SAMP_NUM ** 0.5):
            sub_idx = np.array(list(range(sub_traj.shape[0])))
            extra_idx = np.random.choice(
                list(range(sub_traj.shape[0])), int(SAMP_NUM ** 0.5) - len(sub_idx))
            sub_idx = np.hstack([sub_idx, extra_idx])
            traj_list.append(sub_traj[sub_idx])
        else:
            weight = np.array(RAW_ADJ[sub_traj[:, 0], sub_traj[:, 1]])
            weight = weight.reshape([-1])
            traj_rank = np.argsort(weight)
            traj_list.append(sub_traj[traj_rank[-int(SAMP_NUM ** 0.5):]])
    #             print(sub_traj[traj_rank[-int(SAMP_NUM**0.5):]])
    traj_idx = np.vstack(traj_list)
    traj[1] = [traj_idx]
    return traj


def ord2_samp_traj(traj):
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    idxes = np.unique(traj_idx[:, -1].reshape([-1]))
    if len(idxes) > 1:
        idxes = idxes[idxes != cen_node]
    if len(idxes) <= int(SAMP_NUM):
        extra_idxes = np.random.choice(idxes, SAMP_NUM - len(idxes))
        idxes = np.hstack([idxes, extra_idxes])
    else:
        edges = np.hstack([
            (cen_node * np.ones((idxes.shape[0], 1))).astype(np.int),
            idxes.reshape([-1, 1]).astype(np.int)
        ])
        traj_sim = np.dot(RAW_ADJ[cen_node], RAW_ADJ[idxes].T).toarray().reshape(-1)
        traj_rank = np.argsort(traj_sim)
        idxes = idxes[traj_rank[-SAMP_NUM:]]
    traj_list = []
    for idx in idxes:
        sub_traj = traj_idx[traj_idx[:, -1] == idx]
        if sub_traj.shape[0] == 1:
            traj_list.append(sub_traj)
        else:
            weight = np.array(
                RAW_ADJ[cen_node, sub_traj[:, 0]].toarray()) * np.array(RAW_ADJ[sub_traj[:, 0], sub_traj[:, 1]])
            weight = weight.reshape([-1])
            traj_rank = np.argsort(weight)
            traj_list.append(sub_traj[traj_rank[-1]])
    traj_idx = np.vstack(traj_list)
    traj[1] = [traj_idx]
    return traj


def mix2_samp_traj(traj):
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    idxes2 = np.unique(traj_idx[:, -1].reshape([-1]))
    if len(idxes2) > 1:
        idxes2 = idxes2[idxes2 != cen_node]
    if len(idxes2) <= int(SAMP_NUM):
        extra_idxes = np.random.choice(idxes2, SAMP_NUM - len(idxes2))
        idxes2 = np.hstack([idxes2, extra_idxes])
    else:
        edges = np.hstack([
            (cen_node * np.ones((idxes2.shape[0], 1))).astype(np.int),
            idxes2.reshape([-1, 1]).astype(np.int)
        ])
        traj_sim = np.dot(RAW_ADJ[cen_node], RAW_ADJ[idxes2].T).toarray().reshape(-1)
        traj_rank = np.argsort(traj_sim)
        idxes2 = idxes2[traj_rank[-SAMP_NUM:]]
    idxes1 = np.unique(traj_idx[:, 0].reshape([-1]))
    if len(idxes1) > 1:
        idxes1 = idxes1[idxes1 != cen_node]
    if len(idxes1) >= SAMP_NUM:
        traj_sim = np.array(RAW_ADJ[cen_node, idxes1].toarray())
        traj_sim = traj_sim.reshape([-1])
        traj_rank = np.argsort(traj_sim)
        idxes1 = idxes1[traj_rank[-SAMP_NUM:]]
    else:
        extra_idxes = np.random.choice(idxes1, SAMP_NUM - len(idxes1))
        idxes1 = np.hstack([idxes1, extra_idxes])
    traj_idx = np.hstack([idxes1.reshape([-1, 1]), idxes2.reshape([-1, 1])])
    traj[1] = [traj_idx]
    return traj


def min2_samp_traj(traj):
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    idxes2 = np.unique(traj_idx[:, -1].reshape([-1]))
    if len(idxes2) > 1:
        idxes2 = idxes2[idxes2 != cen_node]
    if len(idxes2) <= int(SAMP_NUM):
        extra_idxes = np.random.choice(idxes2, SAMP_NUM - len(idxes2))
        idxes2 = np.hstack([idxes2, extra_idxes])
    else:
        P = NORM_ADJ[(cen_node * np.ones_like(idxes2)).astype(np.int)]
        Q = NORM_ADJ[idxes2]
        t = P - Q
        traj_sim = np.array(np.sum(t.multiply(t), axis=-1)).reshape(-1)
        traj_rank = np.argsort(-traj_sim)
        idxes2 = idxes2[traj_rank[-SAMP_NUM:]]
    idxes1 = np.unique(traj_idx[:, 0].reshape([-1]))
    if len(idxes1) > 1:
        idxes1 = idxes1[idxes1 != cen_node]
    if len(idxes1) >= SAMP_NUM:
        traj_sim = np.array(RAW_ADJ[cen_node, idxes1].toarray())
        traj_sim = traj_sim.reshape([-1])
        traj_rank = np.argsort(traj_sim)
        idxes1 = idxes1[traj_rank[-SAMP_NUM:]]
    else:
        extra_idxes = np.random.choice(idxes1, SAMP_NUM - len(idxes1))
        idxes1 = np.hstack([idxes1, extra_idxes])
    traj_idx = np.hstack([idxes1.reshape([-1, 1]), idxes2.reshape([-1, 1])])
    traj[1] = [traj_idx]
    return traj


def sin2_samp_traj(traj):
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    idxes = np.unique(traj_idx[:, -1].reshape([-1]))
    if len(idxes) > 1:
        idxes = idxes[idxes != cen_node]
    if len(idxes) <= int(SAMP_NUM):
        extra_idxes = np.random.choice(idxes, SAMP_NUM - len(idxes))
        idxes = np.hstack([idxes, extra_idxes])
    else:
        P = NORM_ADJ[(cen_node * np.ones_like(idxes)).astype(np.int)]
        Q = NORM_ADJ[idxes]
        t = P - Q
        traj_sim = np.array(np.sum(t.multiply(t), axis=-1)).reshape(-1)
        traj_rank = np.argsort(-traj_sim)
        idxes = idxes[traj_rank[-SAMP_NUM:]]
    traj_list = []
    for idx in idxes:
        sub_traj = traj_idx[traj_idx[:, -1] == idx]
        if sub_traj.shape[0] == 1:
            traj_list.append(sub_traj)
        else:
            weight = np.array(
                RAW_ADJ[cen_node, sub_traj[:, 0]].toarray()) * np.array(RAW_ADJ[sub_traj[:, 0], sub_traj[:, 1]])
            weight = weight.reshape([-1])
            traj_rank = np.argsort(weight)
            traj_list.append(sub_traj[traj_rank[-1]])
    traj_idx = np.vstack(traj_list)
    traj[1] = [traj_idx]
    return traj


def sin1_samp_traj(traj):
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    idxes = np.unique(traj_idx[:, -1].reshape([-1]))
    if len(idxes) > 1:
        idxes = idxes[idxes != cen_node]
    if len(idxes) <= int(SAMP_NUM):
        extra_idxes = np.random.choice(idxes, SAMP_NUM - len(idxes))
        idxes = np.hstack([idxes, extra_idxes])
    else:
        P = NORM_ADJ[(cen_node * np.ones_like(idxes)).astype(np.int)]
        Q = NORM_ADJ[idxes]
        t = P - Q
        traj_sim = norm(t, ord=2, axis=-1).reshape(-1)
        traj_rank = np.argsort(-traj_sim)
        idxes = idxes[traj_rank[-SAMP_NUM:]]
    traj_list = []
    for idx in idxes:
        sub_traj = traj_idx[traj_idx[:, -1] == idx]
        if sub_traj.shape[0] == 1:
            traj_list.append(sub_traj)
        else:
            weight = np.array(
                RAW_ADJ[cen_node, sub_traj[:, 0]].toarray()) * np.array(RAW_ADJ[sub_traj[:, 0], sub_traj[:, 1]])
            weight = weight.reshape([-1])
            traj_rank = np.argsort(weight)
            traj_list.append(sub_traj[traj_rank[-1]])
    traj_idx = np.vstack(traj_list)
    traj[1] = [traj_idx]
    return traj


def sikl_samp_traj(traj):
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    idxes = np.unique(traj_idx[:, -1].reshape([-1]))
    if len(idxes) > 1:
        idxes = idxes[idxes != cen_node]
    if len(idxes) <= int(SAMP_NUM):
        extra_idxes = np.random.choice(idxes, SAMP_NUM - len(idxes))
        idxes = np.hstack([idxes, extra_idxes])
    else:
        nz = RAW_ADJ[cen_node].toarray() * (RAW_ADJ[idxes].toarray())
        nz_list = [nz[_].nonzero()[0] for _ in range(nz.shape[0])]
        P = NORM_ADJ[cen_node].toarray()
        Q = NORM_ADJ[idxes].toarray()
        Ps = [P[0, _] for i, _ in enumerate(nz_list)]
        Qs = [Q[i, _] for i, _ in enumerate(nz_list)]
        traj_sim = np.array([np.sum(Ps[i] * np.log(Ps[i] / Qs[i])) for i in range(len(nz_list))])
        traj_rank = np.argsort(-traj_sim)
        idxes = idxes[traj_rank[-SAMP_NUM:]]
    traj_list = []
    for idx in idxes:
        sub_traj = traj_idx[traj_idx[:, -1] == idx]
        if sub_traj.shape[0] == 1:
            traj_list.append(sub_traj)
        else:
            weight = np.array(
                RAW_ADJ[cen_node, sub_traj[:, 0]].toarray()) * np.array(RAW_ADJ[sub_traj[:, 0], sub_traj[:, 1]])
            weight = weight.reshape([-1])
            traj_rank = np.argsort(weight)
            traj_list.append(sub_traj[traj_rank[-1]])
    traj_idx = np.vstack(traj_list)
    traj[1] = [traj_idx]
    return traj


def vdot_samp_traj(traj):
    # node2vec dot distance
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    idxes = np.unique(traj_idx[:, -1].reshape([-1]))
    if len(idxes) > 1:
        idxes = idxes[idxes != cen_node]
    if len(idxes) <= int(SAMP_NUM):
        extra_idxes = np.random.choice(idxes, SAMP_NUM - len(idxes))
        idxes = np.hstack([idxes, extra_idxes])
    else:
        P = NVS[cen_node]
        Q = NVS[idxes]
        traj_sim = np.sum(P * Q, axis=1)
        traj_rank = np.argsort(-traj_sim)
        idxes = idxes[traj_rank[:SAMP_NUM]]
    traj_list = []
    for idx in idxes:
        sub_traj = traj_idx[traj_idx[:, -1] == idx]
        if sub_traj.shape[0] == 1:
            traj_list.append(sub_traj)
        else:
            weight = np.array(
                RAW_ADJ[cen_node, sub_traj[:, 0]].toarray()) * np.array(RAW_ADJ[sub_traj[:, 0], sub_traj[:, 1]])
            weight = weight.reshape([-1])
            traj_rank = np.argsort(weight)
            traj_list.append(sub_traj[traj_rank[-1]])
    traj_idx = np.vstack(traj_list)
    traj[1] = [traj_idx]
    return traj


def vmin_samp_traj(traj):
    # node2vec dot distance
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    idxes = np.unique(traj_idx[:, -1].reshape([-1]))
    if len(idxes) > 1:
        idxes = idxes[idxes != cen_node]
    if len(idxes) <= int(SAMP_NUM):
        extra_idxes = np.random.choice(idxes, SAMP_NUM - len(idxes))
        idxes = np.hstack([idxes, extra_idxes])
    else:
        P = NVS[cen_node]
        Q = NVS[idxes]
        traj_sim = np.sum(np.abs(P - Q), axis=1)
        traj_rank = np.argsort(traj_sim)
        idxes = idxes[traj_rank[:SAMP_NUM]]
    traj_list = []
    for idx in idxes:
        sub_traj = traj_idx[traj_idx[:, -1] == idx]
        if sub_traj.shape[0] == 1:
            traj_list.append(sub_traj)
        else:
            weight = np.array(
                RAW_ADJ[cen_node, sub_traj[:, 0]].toarray()) * np.array(RAW_ADJ[sub_traj[:, 0], sub_traj[:, 1]])
            weight = weight.reshape([-1])
            traj_rank = np.argsort(weight)
            traj_list.append(sub_traj[traj_rank[-1]])
    traj_idx = np.vstack(traj_list)
    traj[1] = [traj_idx]
    return traj


def nor1_samp_traj(traj):
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    traj_idx_bak = traj_idx.copy()
    if len(traj_idx) > 1:
        traj_idx = traj_idx[traj_idx[:, -1] != cen_node]
    if len(traj_idx) > SAMP_NUM:
        P = RAW_ADJ[cen_node, traj_idx[:, 0]].toarray()
        Q = np.array(RAW_ADJ[traj_idx[:, 0], traj_idx[:, 1]])
        traj_sim = (P + Q).reshape(-1)
        traj_rank = np.argsort(-traj_sim)
        traj_idx = traj_idx[traj_rank[:SAMP_NUM]]
    else:
        extra_idxes = np.random.choice(np.arange(len(traj_idx_bak)), SAMP_NUM - len(traj_idx))
        traj_idx = np.vstack([traj_idx, traj_idx_bak[extra_idxes]])
    traj[1] = [traj_idx]
    return traj


def nran_samp_traj(traj):
    # Previous Implementation
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    traj_idx_bak = traj_idx.copy()
    if len(traj_idx) > 1:
        traj_idx = traj_idx[traj_idx[:, -1] != cen_node]
    if len(traj_idx) > SAMP_NUM:
        idxes = np.random.choice(np.arange(len(traj_idx)), SAMP_NUM)
        traj_idx = traj_idx[idxes]
    else:
        extra_idxes = np.random.choice(np.arange(len(traj_idx_bak)), SAMP_NUM - len(traj_idx))
        traj_idx = np.vstack([traj_idx, traj_idx_bak[extra_idxes]])
    traj[1] = [traj_idx]
    return traj


def vmin1_samp_traj(traj):
    # Ord1 cf sampling trajectory
    # node2vec Euclidean distance
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    idxes = np.unique(traj_idx[:, 0].reshape([-1]))
    if len(idxes) > 1:
        idxes = idxes[idxes != cen_node]
    if len(idxes) >= int(SAMP_NUM ** 0.5):
        P = NVS[cen_node]
        Q = NVS[idxes]
        traj_sim = np.sum(np.abs(P - Q), axis=1)
        traj_rank = np.argsort(traj_sim)
        idxes = idxes[traj_rank[:int(SAMP_NUM ** 0.5)]]
    else:
        extra_idxes = np.random.choice(idxes, int(SAMP_NUM ** 0.5) - len(idxes))
        idxes = np.hstack([idxes, extra_idxes])
    traj_list = []
    for idx in idxes:
        #         print(idx)
        sub_traj = traj_idx[traj_idx[:, 0] == idx]
        if sub_traj.shape[0] < int(SAMP_NUM ** 0.5):
            sub_idx = np.array(list(range(sub_traj.shape[0])))
            extra_idx = np.random.choice(
                list(range(sub_traj.shape[0])), int(SAMP_NUM ** 0.5) - len(sub_idx))
            sub_idx = np.hstack([sub_idx, extra_idx])
            traj_list.append(sub_traj[sub_idx])
        else:
            P = NVS[sub_traj[:, 0]]
            Q = NVS[sub_traj[:, 1]]
            traj_sim = np.sum(np.abs(P - Q), axis=1)
            traj_rank = np.argsort(traj_sim)
            traj_list.append(sub_traj[traj_rank[:int(SAMP_NUM ** 0.5)]])
    traj_idx = np.vstack(traj_list)
    traj[1] = [traj_idx]
    return traj


def disrand_samp_traj(traj):
    # Using distance to sample 2-ord neigh
    # Using rand to sample 1-ord neigh
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]

    ### Sample of second-ord neis
    idxes = np.unique(traj_idx[:, -1].reshape([-1]))
    if len(idxes) > 1:
        idxes = idxes[idxes != cen_node]
    if len(idxes) <= int(SAMP_NUM):
        extra_idxes = np.random.choice(idxes, SAMP_NUM - len(idxes))
        idxes2 = np.hstack([idxes, extra_idxes])
    else:
        P = NVS[cen_node]
        Q = NVS[idxes]
        traj_sim = np.sum(np.abs(P - Q), axis=1)
        traj_rank = np.argsort(traj_sim)
        idxes2 = idxes[traj_rank[:SAMP_NUM]]
    ### Sample of first-ord neis
    idxes = np.unique(traj_idx[:, 0].reshape([-1]))
    if len(idxes) <= int(SAMP_NUM):
        extra_idxes = np.random.choice(idxes, SAMP_NUM - len(idxes))
        idxes1 = np.hstack([idxes, extra_idxes])
    else:
        idxes1 = np.random.choice(idxes, SAMP_NUM, replace=False)
    traj_idx = np.hstack([idxes1.reshape([-1, 1]), idxes2.reshape([-1, 1])])
    traj[1] = [traj_idx]
    return traj


def cfplus_samp_traj(traj):
    # Ord1 cf sampling trajectory
    # node2vec Euclidean distance
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    traj_idx_bak = traj_idx.copy()
    if len(traj_idx) > 1:
        traj_idx = traj_idx[traj_idx[:, -1] != cen_node]
    if len(traj_idx) > SAMP_NUM:
        P = NVS[cen_node]
        Q = NVS[traj_idx[:, 0]]
        R = NVS[traj_idx[:, 1]]
        p_q = np.sum(np.abs(P - Q), axis=1)
        q_r = np.sum(np.abs(Q - R), axis=1)
        traj_sim = p_q + q_r
        traj_rank = np.argsort(traj_sim)
        traj_idx = traj_idx[traj_rank[:SAMP_NUM]]
    else:
        extra_idxes = np.random.choice(np.arange(len(traj_idx_bak)), SAMP_NUM - len(traj_idx))
        traj_idx = np.vstack([traj_idx, traj_idx_bak[extra_idxes]])
    traj[1] = [traj_idx]
    return traj


def cfmul_samp_traj(traj):
    # Ord1 cf sampling trajectory
    # node2vec Euclidean distance
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    traj_idx_bak = traj_idx.copy()
    if len(traj_idx) > 1:
        traj_idx = traj_idx[traj_idx[:, -1] != cen_node]
    if len(traj_idx) > SAMP_NUM:
        P = NVS[cen_node]
        Q = NVS[traj_idx[:, 0]]
        R = NVS[traj_idx[:, 1]]
        p_q = np.sum(np.abs(P - Q), axis=1)
        q_r = np.sum(np.abs(Q - R), axis=1)
        traj_sim = p_q * q_r
        traj_rank = np.argsort(traj_sim)
        traj_idx = traj_idx[traj_rank[:SAMP_NUM]]
    else:
        extra_idxes = np.random.choice(np.arange(len(traj_idx_bak)), SAMP_NUM - len(traj_idx))
        traj_idx = np.vstack([traj_idx, traj_idx_bak[extra_idxes]])
    traj[1] = [traj_idx]
    return traj


def cfmin_samp_traj(traj):
    # Ord1 cf sampling trajectory
    # node2vec Euclidean distance
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    traj_idx_bak = traj_idx.copy()
    if len(traj_idx) > 1:
        traj_idx = traj_idx[traj_idx[:, -1] != cen_node]
    if len(traj_idx) > SAMP_NUM:
        P = NVS[cen_node]
        Q = NVS[traj_idx[:, 0]]
        R = NVS[traj_idx[:, 1]]
        p_q = np.sum(np.abs(P - Q), axis=1).reshape([-1, 1])
        q_r = np.sum(np.abs(Q - R), axis=1).reshape([-1, 1])
        traj_sim = np.min(np.hstack([p_q, q_r]), axis=-1)
        traj_rank = np.argsort(traj_sim)
        traj_idx = traj_idx[traj_rank[:SAMP_NUM]]
    else:
        extra_idxes = np.random.choice(np.arange(len(traj_idx_bak)), SAMP_NUM - len(traj_idx))
        traj_idx = np.vstack([traj_idx, traj_idx_bak[extra_idxes]])
    traj[1] = [traj_idx]
    return traj


def cfmax_samp_traj(traj):
    # Ord1 cf sampling trajectory
    # node2vec Euclidean distance
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    traj_idx_bak = traj_idx.copy()
    if len(traj_idx) > 1:
        traj_idx = traj_idx[traj_idx[:, -1] != cen_node]
    if len(traj_idx) > SAMP_NUM:
        P = NVS[cen_node]
        Q = NVS[traj_idx[:, 0]]
        R = NVS[traj_idx[:, 1]]
        p_q = np.sum(np.abs(P - Q), axis=1).reshape([-1, 1])
        q_r = np.sum(np.abs(Q - R), axis=1).reshape([-1, 1])
        traj_sim = np.max(np.hstack([p_q, q_r]), axis=-1)
        traj_rank = np.argsort(traj_sim)
        traj_idx = traj_idx[traj_rank[:SAMP_NUM]]
    else:
        extra_idxes = np.random.choice(np.arange(len(traj_idx_bak)), SAMP_NUM - len(traj_idx))
        traj_idx = np.vstack([traj_idx, traj_idx_bak[extra_idxes]])
    traj[1] = [traj_idx]
    return traj


def vdot1_samp_traj(traj):
    # Ord1 cf sampling trajectory
    # node2vec dot distance
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    idxes = np.unique(traj_idx[:, 0].reshape([-1]))
    if len(idxes) > 1:
        idxes = idxes[idxes != cen_node]
    if len(idxes) >= int(SAMP_NUM ** 0.5):
        P = NVS[cen_node]
        Q = NVS[idxes]
        traj_sim = np.sum(P * Q, axis=1)
        traj_rank = np.argsort(traj_sim)
        idxes = idxes[traj_rank[-int(SAMP_NUM ** 0.5):]]
    else:
        extra_idxes = np.random.choice(idxes, int(SAMP_NUM ** 0.5) - len(idxes))
        idxes = np.hstack([idxes, extra_idxes])
    traj_list = []
    for idx in idxes:
        #         print(idx)
        sub_traj = traj_idx[traj_idx[:, 0] == idx]
        if sub_traj.shape[0] < int(SAMP_NUM ** 0.5):
            sub_idx = np.array(list(range(sub_traj.shape[0])))
            extra_idx = np.random.choice(
                list(range(sub_traj.shape[0])), int(SAMP_NUM ** 0.5) - len(sub_idx))
            sub_idx = np.hstack([sub_idx, extra_idx])
            traj_list.append(sub_traj[sub_idx])
        else:
            P = NVS[sub_traj[:, 0]]
            Q = NVS[sub_traj[:, 1]]
            traj_sim = np.sum(P * Q, axis=1)
            traj_rank = np.argsort(traj_sim)
            traj_list.append(sub_traj[traj_rank[-int(SAMP_NUM ** 0.5):]])
    traj_idx = np.vstack(traj_list)
    traj[1] = [traj_idx]
    return traj


def ndot1_samp_traj(traj):
    # Ord1 cf sampling trajectory
    # node2vec normalized dot distance
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    idxes = np.unique(traj_idx[:, 0].reshape([-1]))
    if len(idxes) > 1:
        idxes = idxes[idxes != cen_node]
    if len(idxes) >= int(SAMP_NUM ** 0.5):
        P = normalize(NVS[cen_node], axis=1, norm='l2')
        Q = normalize(NVS[idxes], axis=1, norm='l2')
        traj_sim = np.sum(P * Q, axis=1)
        traj_rank = np.argsort(traj_sim)
        idxes = idxes[traj_rank[-int(SAMP_NUM ** 0.5):]]
    else:
        extra_idxes = np.random.choice(idxes, int(SAMP_NUM ** 0.5) - len(idxes))
        idxes = np.hstack([idxes, extra_idxes])
    traj_list = []
    for idx in idxes:
        #         print(idx)
        sub_traj = traj_idx[traj_idx[:, 0] == idx]
        if sub_traj.shape[0] < int(SAMP_NUM ** 0.5):
            sub_idx = np.array(list(range(sub_traj.shape[0])))
            extra_idx = np.random.choice(
                list(range(sub_traj.shape[0])), int(SAMP_NUM ** 0.5) - len(sub_idx))
            sub_idx = np.hstack([sub_idx, extra_idx])
            traj_list.append(sub_traj[sub_idx])
        else:
            P = normalize(NVS[sub_traj[:, 0]], axis=1, norm='l2')
            Q = normalize(NVS[sub_traj[:, 1]], axis=1, norm='l2')
            traj_sim = np.sum(P * Q, axis=1)
            traj_rank = np.argsort(traj_sim)
            traj_list.append(sub_traj[traj_rank[-int(SAMP_NUM ** 0.5):]])
    traj_idx = np.vstack(traj_list)
    traj[1] = [traj_idx]
    return traj


def dotplus_samp_traj(traj):
    # Ord1 cf sampling trajectory
    # node2vec Euclidean distance
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    traj_idx_bak = traj_idx.copy()
    if len(traj_idx) > 1:
        traj_idx = traj_idx[traj_idx[:, -1] != cen_node]
    if len(traj_idx) > SAMP_NUM:
        P = NVS[cen_node]
        Q = NVS[traj_idx[:, 0]]
        R = NVS[traj_idx[:, 1]]
        p_q = np.sum(P * Q, axis=1)
        q_r = np.sum(Q * R, axis=1)
        traj_sim = p_q + q_r
        traj_rank = np.argsort(-traj_sim)
        traj_idx = traj_idx[traj_rank[:SAMP_NUM]]
    else:
        extra_idxes = np.random.choice(np.arange(len(traj_idx_bak)), SAMP_NUM - len(traj_idx))
        traj_idx = np.vstack([traj_idx, traj_idx_bak[extra_idxes]])
    traj[1] = [traj_idx]
    return traj


def dotmul_samp_traj(traj):
    # Ord1 cf sampling trajectory
    # node2vec Euclidean distance
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    traj_idx_bak = traj_idx.copy()
    if len(traj_idx) > 1:
        traj_idx = traj_idx[traj_idx[:, -1] != cen_node]
    if len(traj_idx) > SAMP_NUM:
        P = NVS[cen_node]
        Q = NVS[traj_idx[:, 0]]
        R = NVS[traj_idx[:, 1]]
        p_q = np.sum(P * Q, axis=1)
        q_r = np.sum(Q * R, axis=1)
        traj_sim = p_q * q_r
        traj_rank = np.argsort(-traj_sim)
        traj_idx = traj_idx[traj_rank[:SAMP_NUM]]
    else:
        extra_idxes = np.random.choice(np.arange(len(traj_idx_bak)), SAMP_NUM - len(traj_idx))
        traj_idx = np.vstack([traj_idx, traj_idx_bak[extra_idxes]])
    traj[1] = [traj_idx]
    return traj


def dotmin_samp_traj(traj):
    # Ord1 cf sampling trajectory
    # node2vec Euclidean distance
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    traj_idx_bak = traj_idx.copy()
    if len(traj_idx) > 1:
        traj_idx = traj_idx[traj_idx[:, -1] != cen_node]
    if len(traj_idx) > SAMP_NUM:
        P = NVS[cen_node]
        Q = NVS[traj_idx[:, 0]]
        R = NVS[traj_idx[:, 1]]
        p_q = np.sum(P * Q, axis=1)
        q_r = np.sum(Q * R, axis=1)
        traj_sim = np.min(np.hstack([p_q, q_r]), axis=-1)
        traj_rank = np.argsort(-traj_sim)
        traj_idx = traj_idx[traj_rank[:SAMP_NUM]]
    else:
        extra_idxes = np.random.choice(np.arange(len(traj_idx_bak)), SAMP_NUM - len(traj_idx))
        traj_idx = np.vstack([traj_idx, traj_idx_bak[extra_idxes]])
    traj[1] = [traj_idx]
    return traj


def dotmax_samp_traj(traj):
    # Ord1 cf sampling trajectory
    # node2vec Euclidean distance
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    traj_idx_bak = traj_idx.copy()
    if len(traj_idx) > 1:
        traj_idx = traj_idx[traj_idx[:, -1] != cen_node]
    if len(traj_idx) > SAMP_NUM:
        P = NVS[cen_node]
        Q = NVS[traj_idx[:, 0]]
        R = NVS[traj_idx[:, 1]]
        p_q = np.sum(P * Q, axis=1)
        q_r = np.sum(Q * R, axis=1)
        traj_sim = np.max(np.hstack([p_q, q_r]), axis=-1)
        traj_rank = np.argsort(-traj_sim)
        traj_idx = traj_idx[traj_rank[:SAMP_NUM]]
    else:
        extra_idxes = np.random.choice(np.arange(len(traj_idx_bak)), SAMP_NUM - len(traj_idx))
        traj_idx = np.vstack([traj_idx, traj_idx_bak[extra_idxes]])
    traj[1] = [traj_idx]
    return traj


def dis1rand_samp_traj(traj):
    # Ord1 cf sampling trajectory
    # node2vec Euclidean distance
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    idxes = np.unique(traj_idx[:, 0].reshape([-1]))
    if len(idxes) > 1:
        idxes = idxes[idxes != cen_node]
    if len(idxes) >= int(SAMP_NUM ** 0.5):
        P = NVS[cen_node]
        Q = NVS[idxes]
        traj_sim = np.sum(np.abs(P - Q), axis=1)
        traj_rank = np.argsort(traj_sim)
        idxes = idxes[traj_rank[:int(SAMP_NUM ** 0.5)]]
    else:
        extra_idxes = np.random.choice(idxes, int(SAMP_NUM ** 0.5) - len(idxes))
        idxes = np.hstack([idxes, extra_idxes])
    traj_list = []
    for idx in idxes:
        #         print(idx)
        sub_traj = traj_idx[traj_idx[:, 0] == idx]
        if sub_traj.shape[0] < int(SAMP_NUM ** 0.5):
            sub_idx = np.array(list(range(sub_traj.shape[0])))
            extra_idx = np.random.choice(
                list(range(sub_traj.shape[0])), int(SAMP_NUM ** 0.5) - len(sub_idx))
            sub_idx = np.hstack([sub_idx, extra_idx])
            traj_list.append(sub_traj[sub_idx])
        else:
            P = NVS[sub_traj[:, 0]]
            Q = NVS[sub_traj[:, 1]]
            traj_sim = np.sum(np.abs(P - Q), axis=1)
            traj_rank = np.argsort(traj_sim)
            traj_list.append(sub_traj[traj_rank[:int(SAMP_NUM ** 0.5)]])
    traj_idx = np.vstack(traj_list)
    idxes = np.unique(traj_idx[:, 0].reshape([-1]))
    if len(idxes) <= int(SAMP_NUM):
        extra_idxes = np.random.choice(idxes, SAMP_NUM - len(idxes))
        idxes1 = np.hstack([idxes, extra_idxes])
    else:
        idxes1 = np.random.choice(idxes, SAMP_NUM, replace=False)
    traj_idx[:, 0] = idxes1
    traj[1] = [traj_idx]
    return traj


def sepdot_samp_traj(traj):
    # Ord1 cf sampling trajectory
    # node2vec dot distance
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    ### Sample of first-ord neis
    idxes = np.unique(traj_idx[:, 0].reshape([-1]))
    if len(idxes) > 1:
        idxes = idxes[idxes != cen_node]
    if len(idxes) >= int(SAMP_NUM):
        P = NVS[cen_node]
        Q = NVS[idxes]
        traj_sim = np.sum(P * Q, axis=1)
        traj_rank = np.argsort(traj_sim)
        idxes1 = idxes[traj_rank[-int(SAMP_NUM):]]
    else:
        extra_idxes = np.random.choice(idxes, int(SAMP_NUM) - len(idxes))
        idxes1 = np.hstack([idxes, extra_idxes])

    ### Sample of second-ord neis
    idxes = np.unique(traj_idx[:, -1].reshape([-1]))
    if len(idxes) > 1:
        idxes = idxes[idxes != cen_node]
    if len(idxes) <= int(SAMP_NUM):
        extra_idxes = np.random.choice(idxes, SAMP_NUM - len(idxes))
        idxes2 = np.hstack([idxes, extra_idxes])
    else:
        P = NVS[cen_node]
        Q = NVS[idxes]
        traj_sim = np.sum(P * Q, axis=1)
        traj_rank = np.argsort(traj_sim)
        idxes2 = idxes[traj_rank[-int(SAMP_NUM):]]
    traj_idx = np.hstack([idxes1.reshape([-1, 1]), idxes2.reshape([-1, 1])])
    traj[1] = [traj_idx]
    return traj


def sepdis_samp_traj(traj):
    # Ord1 cf sampling trajectory
    # node2vec dot distance
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    ### Sample of first-ord neis
    idxes = np.unique(traj_idx[:, 0].reshape([-1]))
    if len(idxes) > 1:
        idxes = idxes[idxes != cen_node]
    if len(idxes) >= int(SAMP_NUM):
        P = NVS[cen_node]
        Q = NVS[idxes]
        traj_sim = np.sum(np.abs(P - Q), axis=1)
        traj_rank = np.argsort(traj_sim)
        idxes1 = idxes[traj_rank[:SAMP_NUM]]
    else:
        extra_idxes = np.random.choice(idxes, int(SAMP_NUM) - len(idxes))
        idxes1 = np.hstack([idxes, extra_idxes])

    ### Sample of second-ord neis
    idxes = np.unique(traj_idx[:, -1].reshape([-1]))
    if len(idxes) > 1:
        idxes = idxes[idxes != cen_node]
    if len(idxes) <= int(SAMP_NUM):
        extra_idxes = np.random.choice(idxes, SAMP_NUM - len(idxes))
        idxes2 = np.hstack([idxes, extra_idxes])
    else:
        P = NVS[cen_node]
        Q = NVS[idxes]
        traj_sim = np.sum(np.abs(P - Q), axis=1)
        traj_rank = np.argsort(traj_sim)
        idxes2 = idxes[traj_rank[:SAMP_NUM]]
    traj_idx = np.hstack([idxes1.reshape([-1, 1]), idxes2.reshape([-1, 1])])
    traj[1] = [traj_idx]
    return traj


def dis1dot_samp_traj(traj):
    # Ord1 cf sampling trajectory
    # node2vec Euclidean distance
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    idxes = np.unique(traj_idx[:, 0].reshape([-1]))
    if len(idxes) > 1:
        idxes = idxes[idxes != cen_node]
    if len(idxes) >= int(SAMP_NUM):
        P = NVS[cen_node]
        Q = NVS[idxes]
        traj_sim = np.sum(np.abs(P - Q), axis=1)
        traj_rank = np.argsort(traj_sim)
        idxes1 = idxes[traj_rank[:SAMP_NUM]]
    else:
        extra_idxes = np.random.choice(idxes, int(SAMP_NUM) - len(idxes))
        idxes1 = np.hstack([idxes, extra_idxes])
    idxes = np.unique(traj_idx[:, 0].reshape([-1]))
    if len(idxes) > 1:
        idxes = idxes[idxes != cen_node]
    if len(idxes) >= int(SAMP_NUM ** 0.5):
        P = NVS[cen_node]
        Q = NVS[idxes]
        traj_sim = np.sum(np.abs(P - Q), axis=1)
        traj_rank = np.argsort(traj_sim)
        idxes = idxes[traj_rank[:int(SAMP_NUM ** 0.5)]]
    else:
        extra_idxes = np.random.choice(idxes, int(SAMP_NUM ** 0.5) - len(idxes))
        idxes = np.hstack([idxes, extra_idxes])
    traj_list = []
    for idx in idxes:
        #         print(idx)
        sub_traj = traj_idx[traj_idx[:, 0] == idx]
        if sub_traj.shape[0] < int(SAMP_NUM ** 0.5):
            sub_idx = np.array(list(range(sub_traj.shape[0])))
            extra_idx = np.random.choice(
                list(range(sub_traj.shape[0])), int(SAMP_NUM ** 0.5) - len(sub_idx))
            sub_idx = np.hstack([sub_idx, extra_idx])
            traj_list.append(sub_traj[sub_idx])
        else:
            P = NVS[sub_traj[:, 0]]
            Q = NVS[sub_traj[:, 1]]
            traj_sim = np.sum(np.abs(P - Q), axis=1)
            traj_rank = np.argsort(traj_sim)
            traj_list.append(sub_traj[traj_rank[:int(SAMP_NUM ** 0.5)]])
    traj_idx = np.vstack(traj_list)
    traj_idx[:, 0] = idxes1
    traj[1] = [traj_idx]
    return traj


def rsin2_samp_traj(traj):
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    idxes = np.unique(traj_idx[:, 0].reshape([-1]))
    if len(idxes) > 1:
        idxes = idxes[idxes != cen_node]
    if len(idxes) >= int(10):
        idxes = np.random.choice(idxes, int(10), replace=False)
    else:
        extra_idxes = np.random.choice(idxes, int(10) - len(idxes))
        idxes = np.hstack([idxes, extra_idxes])
    traj_list = []
    for idx in idxes:
        sub_traj = traj_idx[traj_idx[:, 0] == idx]
        if sub_traj.shape[0] < int(10):
            sub_idx = np.array(list(range(sub_traj.shape[0])))
            extra_idx = np.random.choice(
                list(range(sub_traj.shape[0])), int(10) - len(sub_idx))
            sub_idx = np.hstack([sub_idx, extra_idx])
            traj_list.append(sub_traj[sub_idx])
        else:
            sub_idx = np.random.choice(
                list(range(sub_traj.shape[0])),
                int(10), replace=False)
            traj_list.append(sub_traj[sub_idx])
    #             print(sub_traj[traj_rank[-int(SAMP_NUM**0.5):]])
    traj_idx = np.vstack(traj_list)
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    idxes = np.unique(traj_idx[:, -1].reshape([-1]))
    if len(idxes) > 1:
        idxes = idxes[idxes != cen_node]
    if len(idxes) <= int(SAMP_NUM):
        extra_idxes = np.random.choice(idxes, SAMP_NUM - len(idxes))
        idxes = np.hstack([idxes, extra_idxes])
    else:
        P = NORM_ADJ[(cen_node * np.ones_like(idxes)).astype(np.int)]
        Q = NORM_ADJ[idxes]
        t = P - Q
        traj_sim = np.array(np.sum(t.multiply(t), axis=-1)).reshape(-1)
        traj_rank = np.argsort(-traj_sim)
        idxes = idxes[traj_rank[-SAMP_NUM:]]
    traj_list = []
    for idx in idxes:
        sub_traj = traj_idx[traj_idx[:, -1] == idx]
        if sub_traj.shape[0] == 1:
            traj_list.append(sub_traj)
        else:
            weight = np.array(
                RAW_ADJ[cen_node, sub_traj[:, 0]].toarray()) * np.array(RAW_ADJ[sub_traj[:, 0], sub_traj[:, 1]])
            weight = weight.reshape([-1])
            traj_rank = np.argsort(weight)
            traj_list.append(sub_traj[traj_rank[-1]])
    traj_idx = np.vstack(traj_list)
    traj[1] = [traj_idx]
    return traj


def ord1sin2_samp_traj(traj):
    cen_node = traj[0]
    traj_idx = traj[1][0]
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        return_traj = cen_node * np.ones((int(SAMP_NUM), 2)).astype(np.int)
        traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
        return_traj[:len(traj_idx)] = traj_idx
        traj[1] = [return_traj]
        return (traj)
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    idxes = np.unique(traj_idx[:, 0].reshape([-1]))
    if len(idxes) > 1:
        idxes = idxes[idxes != cen_node]
    if len(idxes) >= int(SAMP_NUM ** 0.5):
        traj_sim = np.array(RAW_ADJ[cen_node, idxes].toarray())
        traj_sim = traj_sim.reshape([-1])
        traj_rank = np.argsort(traj_sim)
        idxes = idxes[traj_rank[-int(SAMP_NUM ** 0.5):]]
    else:
        extra_idxes = np.random.choice(idxes, int(SAMP_NUM ** 0.5) - len(idxes))
        idxes = np.hstack([idxes, extra_idxes])
    traj_list = []
    for idx in idxes:
        #         print(idx)
        sub_traj = traj_idx[traj_idx[:, 0] == idx]
        if sub_traj.shape[0] < int(SAMP_NUM ** 0.5):
            sub_idx = np.array(list(range(sub_traj.shape[0])))
            extra_idx = np.random.choice(
                list(range(sub_traj.shape[0])), int(SAMP_NUM ** 0.5) - len(sub_idx))
            sub_idx = np.hstack([sub_idx, extra_idx])
            traj_list.append(sub_traj[sub_idx])
        else:
            P = NORM_ADJ[(cen_node * np.ones(sub_traj.shape[0])).astype(np.int)]
            Q = NORM_ADJ[sub_traj[:, 1]]
            t = P - Q
            traj_sim = np.array(np.sum(t.multiply(t), axis=-1)).reshape(-1)
            traj_rank = np.argsort(-traj_sim)
            traj_list.append(sub_traj[traj_rank[-int(SAMP_NUM ** 0.5):]])
    #             print(sub_traj[traj_rank[-int(SAMP_NUM**0.5):]])
    traj_idx = np.vstack(traj_list)
    traj[1] = [traj_idx]
    return traj


def get_stack_emb(idx):
    emb_node = FEATURES[idx]
    emb_traj = FEATURES[traj_list[idx][1][0][:, -1]].reshape(-1)
    #     if idx %10000 ==0:
    #         print("Get stack embedding for %d nodes"%idx)
    return np.hstack([emb_node, emb_traj])


def get_samp_traj(idx):
    traj = SAMP_FUNC(TRAJS[idx])
    return traj


def get_agg_emb(idx):
    embs = []
    for _ in LAYERS:
        if _ == 0:
            embs.append(FEATURES[idx])
            continue
        if _ > 0:
            embs.append(np.mean(FEATURES[traj_list[idx][1][0][:, _ - 1]], axis=0))
    if len(LAYERS) > 1:
        return np.hstack([embs])
    else:
        return embs[0].reshape([1, 1, embs[0].shape[0]])


def get_cf_score(idx):
    traj = traj_list[idx]
    cen_node = traj[0]
    traj_idx = traj[1][0]
    P = NVS[cen_node]
    Q = NVS[traj_idx[:, 0]]
    R = NVS[traj_idx[:, 1]]
    p_q = np.mean(np.abs(P - Q), axis=1)
    q_r = np.mean(np.abs(Q - R), axis=1)
    traj_sim = p_q + q_r
    return np.array([np.mean(p_q), np.mean(q_r), np.mean(traj_sim)])


NUM_NODES, FEATURES, ADJ_TRAIN, NUM_FEATURES, MAX_SAMP_NEI = 0, 0, 0, 0, 0
IF_BAGGING, SAMP_PRE_NUM, SAMP_NUM, SAMP_TIMES, DEGREE = 0, 0, 0, 0, 0
SORTED, WEIGHT, SAMP_FUNC = 0, 0, 0
RAW_ADJ, ADJ_SUM, MAX_DEGREE = 0, 0, 0
N_JOBS = 0

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="CiteULike",
                    help='Dataset to use.')
parser.add_argument('--datadir', type=str, default="./data/process/",
                    help='Director of the dataset.')
parser.add_argument('--user_samp', type=str, default='sepdot',
                    help='Sampling method for user.')
parser.add_argument('--item_samp', type=str, default='sepdot',
                    help='Sampling method for item.')
parser.add_argument('--embed_meth', type=str, default='grmf',
                    help='Embedding_method.')
parser.add_argument('--if_samp', action='store_true', default=False,
                    help='Whether use Sampling.')
parser.add_argument('--if_stat', action='store_true', default=False,
                    help='Whether statistic the dataset.')
parser.add_argument('--if_raw', action='store_true', default=False,
                    help='Whether use raw adj matrix.')
parser.add_argument('--if_norm', action='store_true', default=False,
                    help='Whether normalized the .')
parser.add_argument('--recompute', action='store_true', default=False,
                    help='Whether recompute the trajectory list.')
parser.add_argument('--samp_size', type=int, default=25,
                    help='Sampling size.')
parser.add_argument('--layers', nargs='?', default='[0,1,2]',
                    help='Output layers')
parser.add_argument('--n_jobs', type=int, default=12,
                    help='Multiprocessing number.')
parser.add_argument('--if_stack', action='store_true', default=False,
                    help='Whether output stack embeddings.')
args, _ = parser.parse_known_args()
if not args.if_stack:
    args.if_raw = True
print('#' * 70)
print('\n'.join([(str(_) + ':' + str(vars(args)[_])) for _ in vars(args).keys()]))

if not eval(args.layers):
    layers = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
else:
    layers = [eval(args.layers)]
print('Computing Layers', layers)

map_dict = pickle.load(open(args.datadir + args.dataset + '/warm_dict.pkl', 'rb'))
user_num = map_dict['user_num']
item_num = map_dict['item_num']
num_nodes = user_num + item_num
df_train = pd.read_csv(args.datadir + args.dataset + '/warm_emb.csv', dtype=np.int)
adj_train = compute_adj(df_train)
# adj_train = sp.load_npz(args.datadir + args.dataset + '_adj_train.npz')
features = np.load(args.datadir + args.dataset + '/{}.npy'.format(args.embed_meth))
print('Loading:' + args.datadir + args.dataset + '/{}.npy'.format(args.embed_meth))
# NVS = normalize(features, axis=1, norm='l2')
NVS = features
NORM_ADJ = normalize(adj_train, axis=1, norm='l1')

init(adj_train, features, 1, args.samp_size, 1, 2, max_degree=256,
     max_samp_nei=256, if_normalized=args.if_norm, degree_normalized=False,
     if_self_loop=False, if_bagging=False, if_sort=False,
     weight='same', n_jobs=args.n_jobs, seed=42)

user_list = list(range(user_num))
item_list = list(range(user_num, user_num + item_num))
node_list = user_list + item_list
traj_file = args.datadir + args.dataset + '/traj.pkl'
t0 = time.time()
with Pool(args.n_jobs) as pool:
    user_traj = pool.map(get_gun_traj, user_list)
    item_traj = pool.map(get_gun_traj, item_list)
t1 = time.time()
print('Get Trajs in %.2f second, saving to %s' % (t1 - t0, traj_file))
TRAJS = user_traj + item_traj
pickle.dump(TRAJS, open(args.datadir + args.dataset + '_traj.pkl', 'wb'))

t0 = time.time()
traj_list = []
if args.user_samp == 'rand':
    SAMP_FUNC = rand_samp_traj
elif args.user_samp == 'ord2':
    SAMP_FUNC = ord2_samp_traj
elif args.user_samp == 'ord1':
    SAMP_FUNC = ord1_samp_traj
elif args.user_samp == 'sin1':
    SAMP_FUNC = sin1_samp_traj
elif args.user_samp == 'sin2':
    SAMP_FUNC = sin2_samp_traj
elif args.user_samp == 'rsin2':
    SAMP_FUNC = rsin2_samp_traj
elif args.user_samp == 'ord1sin2':
    SAMP_FUNC = ord1sin2_samp_traj
elif args.user_samp == 'vdot':
    SAMP_FUNC = vdot_samp_traj
elif args.user_samp == 'vmin':
    SAMP_FUNC = vmin_samp_traj
elif args.user_samp == 'nran':
    SAMP_FUNC = nran_samp_traj
elif args.user_samp == 'nor1':
    SAMP_FUNC = nor1_samp_traj
elif args.user_samp == 'gdot':
    SAMP_FUNC = vdot_samp_traj
    NVS = np.load(args.datadir + args.dataset + '_gcn_emb.npy')
elif args.user_samp == 'gmin':
    SAMP_FUNC = vmin_samp_traj
    NVS = np.load(args.datadir + args.dataset + '_gcn_emb.npy')
elif args.user_samp == 'vmin1':
    SAMP_FUNC = vmin1_samp_traj
elif args.user_samp == 'vdot1':
    SAMP_FUNC = vdot1_samp_traj
elif args.user_samp == 'ndot1':
    SAMP_FUNC = vdot1_samp_traj
elif args.user_samp == 'cfplus':
    SAMP_FUNC = cfplus_samp_traj
elif args.user_samp == 'cfmul':
    SAMP_FUNC = cfmul_samp_traj
elif args.user_samp == 'cfmin':
    SAMP_FUNC = cfmin_samp_traj
elif args.user_samp == 'cfmax':
    SAMP_FUNC = cfmax_samp_traj
elif args.user_samp == 'dotplus':
    SAMP_FUNC = dotplus_samp_traj
elif args.user_samp == 'dotmul':
    SAMP_FUNC = dotmul_samp_traj
elif args.user_samp == 'dotmin':
    SAMP_FUNC = dotmin_samp_traj
elif args.user_samp == 'dotmax':
    SAMP_FUNC = dotmax_samp_traj
elif args.user_samp == 'disrand':
    SAMP_FUNC = disrand_samp_traj
elif args.user_samp == 'dis1rand':
    SAMP_FUNC = dis1rand_samp_traj
elif args.user_samp == 'sepdot':
    SAMP_FUNC = sepdot_samp_traj
elif args.user_samp == 'sepdis':
    SAMP_FUNC = sepdis_samp_traj
elif args.user_samp == 'dis1dot':
    SAMP_FUNC = sepdis_samp_traj
elif args.user_samp == 'mix2':
    SAMP_FUNC = mix2_samp_traj
elif args.user_samp == 'min2':
    SAMP_FUNC = min2_samp_traj
else:
    raise 'SAMP_FUNC %s not defined' % (args.user_samp)
with Pool(args.n_jobs) as pool:
    user_traj = pool.map(get_samp_traj, user_list)
if args.item_samp == 'rand':
    SAMP_FUNC = rand_samp_traj
elif args.item_samp == 'ord2':
    SAMP_FUNC = ord2_samp_traj
elif args.item_samp == 'ord1':
    SAMP_FUNC = ord1_samp_traj
elif args.item_samp == 'sin1':
    SAMP_FUNC = sin1_samp_traj
elif args.item_samp == 'sin2':
    SAMP_FUNC = sin2_samp_traj
elif args.item_samp == 'rsin2':
    SAMP_FUNC = rsin2_samp_traj
elif args.item_samp == 'ord1sin2':
    SAMP_FUNC = ord1sin2_samp_traj
elif args.item_samp == 'vdot':
    SAMP_FUNC = vdot_samp_traj
elif args.item_samp == 'vmin':
    SAMP_FUNC = vmin_samp_traj
elif args.item_samp == 'nran':
    SAMP_FUNC = nran_samp_traj
elif args.item_samp == 'nor1':
    SAMP_FUNC = nor1_samp_traj
elif args.item_samp == 'gdot':
    SAMP_FUNC = vdot_samp_traj
    NVS = np.load(args.datadir + args.dataset + '_gcn_emb.npy')
elif args.item_samp == 'gmin':
    SAMP_FUNC = vmin_samp_traj
    NVS = np.load(args.datadir + args.dataset + '_gcn_emb.npy')
elif args.item_samp == 'vmin1':
    SAMP_FUNC = vmin1_samp_traj
elif args.item_samp == 'vdot1':
    SAMP_FUNC = vdot1_samp_traj
elif args.item_samp == 'ndot1':
    SAMP_FUNC = vdot1_samp_traj
elif args.item_samp == 'cfplus':
    SAMP_FUNC = cfplus_samp_traj
elif args.item_samp == 'cfmul':
    SAMP_FUNC = cfmul_samp_traj
elif args.item_samp == 'cfmin':
    SAMP_FUNC = cfmin_samp_traj
elif args.item_samp == 'cfmax':
    SAMP_FUNC = cfmax_samp_traj
elif args.item_samp == 'dotplus':
    SAMP_FUNC = dotplus_samp_traj
elif args.item_samp == 'dotmul':
    SAMP_FUNC = dotmul_samp_traj
elif args.item_samp == 'dotmin':
    SAMP_FUNC = dotmin_samp_traj
elif args.item_samp == 'dotmax':
    SAMP_FUNC = dotmax_samp_traj
elif args.item_samp == 'disrand':
    SAMP_FUNC = disrand_samp_traj
elif args.item_samp == 'dis1rand':
    SAMP_FUNC = dis1rand_samp_traj
elif args.item_samp == 'sepdot':
    SAMP_FUNC = sepdot_samp_traj
elif args.item_samp == 'sepdis':
    SAMP_FUNC = sepdis_samp_traj
elif args.item_samp == 'dis1dot':
    SAMP_FUNC = sepdis_samp_traj
elif args.item_samp == 'mix2':
    SAMP_FUNC = mix2_samp_traj
elif args.item_samp == 'min2':
    SAMP_FUNC = min2_samp_traj
else:
    raise 'SAMP_FUNC %s not defined' % args.item_samp
with Pool(args.n_jobs) as pool:
    t0 = time.time()
    item_traj = pool.map(get_samp_traj, item_list)
    t1 = time.time()
    print('Get Sampled Trajs in %.2f second' % (t1 - t0))
traj_list = user_traj + item_traj
for layer in layers:
    LAYERS = layer
    if args.if_stack:
        with Pool(args.n_jobs) as pool:
            stack_emb = pool.map(get_stack_emb, node_list)
            t2 = time.time()
            print('Get Stacked Emb in %.2f second' % (t2 - t1))
        stack_emb = np.stack(stack_emb)
        print('Shape of Emb', stack_emb.shape)
        if args.if_raw:
            raw_str = 'raw'
        else:
            raw_str = ''

        print(args.datadir + args.dataset + '/' + args.embed_meth + '_' +
              str(args.samp_size) + '_' + ''.join([str(_) for _ in LAYERS])
              + '_u(' + args.user_samp + ')_i(' + args.item_samp + ')_stack.npy')
        np.save(args.datadir + args.dataset + '/' + args.embed_meth + '_' +
                str(args.samp_size) + '_' + ''.join([str(_) for _ in LAYERS])
                + '_u(' + args.user_samp + ')_i(' + args.item_samp + ')_stack.npy', np.stack(stack_emb))
    else:
        with Pool(args.n_jobs) as pool:
            stack_emb = pool.map(get_agg_emb, node_list)
            t2 = time.time()
            print('Get Stacked Emb in %.2f second' % (t2 - t1))
        stack_emb = np.stack(stack_emb)
        print(stack_emb.shape)
        print(
            args.datadir + args.dataset + '/' + args.embed_meth + '_' +
            str(args.samp_size) + '_' + ''.join([str(_) for _ in LAYERS])
            + '_u(' + args.user_samp + ')_i(' + args.item_samp + ')_agg.npy')
        np.save(args.datadir + args.dataset + '/' + args.embed_meth + '_' +
                str(args.samp_size) + '_' + ''.join([str(_) for _ in LAYERS])
                + '_u(' + args.user_samp + ')_i(' + args.item_samp + ')_agg.npy', np.stack(stack_emb))

