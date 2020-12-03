import time
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from pprint import pprint
import os
import scipy.sparse as sp

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="CiteULike", help='Dataset to use.')
parser.add_argument('--data_dir', type=str, default="./data/process/", help='Director of the dataset.')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--neg_rate', type=int, default=5, help='The negative sampling rate.')
args = parser.parse_args()
pprint(vars(args))

random.seed(args.seed)
np.random.seed(args.seed)
t0 = time.time()

store_path = args.data_dir + args.dataset
if not os.path.exists(store_path):
    os.makedirs(store_path)


"""read data from file"""
df_emb = pd.read_csv(store_path + '/warm_emb.csv', dtype=np.int)
df_map = pd.read_csv(store_path + '/warm_map.csv', dtype=np.int)
df_warm_val = pd.read_csv(store_path + '/warm_val.csv', dtype=np.int)
df_warm_test = pd.read_csv(store_path + '/warm_test.csv', dtype=np.int)

i2user = np.unique(df_emb['user'])  # emb set 里包含 warm set 的所有 user 和 item.
i2item = np.unique(df_emb['item'])

n_warm_user = len(i2user)
n_warm_item = len(i2item)
n_warm_node = n_warm_user + n_warm_item
print('[warm]\nUser: %d\tItem: %d' % (n_warm_user, n_warm_item))


"""user/item map for creating adj matrix"""
user2i = {user: i for i, user in enumerate(i2user)}
item2i = {item: i for i, item in enumerate(i2item)}

df_emb.iloc[:, 0] = np.array(list(map(user2i.get, df_emb.iloc[:, 0])))
df_emb.iloc[:, 1] = np.array(list(map(item2i.get, df_emb.iloc[:, 1])))
df_map.iloc[:, 0] = np.array(list(map(user2i.get, df_map.iloc[:, 0])))
df_map.iloc[:, 1] = np.array(list(map(item2i.get, df_map.iloc[:, 1])))
df_warm_val.iloc[:, 0] = np.array(list(map(user2i.get, df_warm_val.iloc[:, 0])))
df_warm_val.iloc[:, 1] = np.array(list(map(item2i.get, df_warm_val.iloc[:, 1])))
df_warm_test.iloc[:, 0] = np.array(list(map(user2i.get, df_warm_test.iloc[:, 0])))
df_warm_test.iloc[:, 1] = np.array(list(map(item2i.get, df_warm_test.iloc[:, 1])))

# store map file
user_map_file = [[org_id, remap_id] for remap_id, org_id in enumerate(i2user)]
user_map_file = np.vstack(user_map_file).astype(np.int)
item_map_file = [[org_id, remap_id] for remap_id, org_id in enumerate(i2item)]
item_map_file = np.vstack(item_map_file).astype(np.int)

pd.DataFrame(user_map_file).to_csv(store_path + '/warm_user_mapped.csv', header=['org_id', 'remap_id'], index=False)
pd.DataFrame(item_map_file).to_csv(store_path + '/warm_item_mapped.csv', header=['org_id', 'remap_id'], index=False)


def get_neighbors(input_df):
    """
    Get users' neighboring items.
    return:
        d - {user: [items]}
    """
    group = input_df.groupby('user')
    d = {g[0]: g[1].item.values for g in group}

    return d


def comput_adj(input_df):
    cols = input_df.user.values
    rows = input_df.item.values + n_warm_user
    values = np.ones(len(cols))

    adj = sp.csr_matrix((values, (rows, cols)), shape=(n_warm_node, n_warm_node))
    adj = adj + adj.T
    return adj


def _neg_samp(uid, main_d):
    """
    Generate pos and neg samples for user.
    n_neg = 5 * n_pos
    负采样是针对训练集加上当前集合的

    param:
        uid - uid is from whole user set.
        status - It refers to which subset we are sampling.

    return:
        ret_array - [[user_id item_id label] for samples of a user]
    """
    # pos sampling
    pos_neigh = main_d.get(uid, [])
    pos_num = len(pos_neigh)

    # excluded item
    pos_set = emb_nb.get(uid, [])
    pos_set = np.union1d(pos_neigh, pos_set)

    # neg sampling
    neg_item = []
    while len(neg_item) < pos_num * args.neg_rate:
        neg_item = np.random.choice(np.array(list(range(n_warm_item))), 2 * args.neg_rate * pos_num)
        neg_item = neg_item[[i not in pos_set for i in neg_item]]  # "i not in pos_set" returns a bool value.

    ret_array = np.zeros((pos_num * (args.neg_rate + 1), 3)).astype(np.int32)
    ret_array[:, 0] = uid
    ret_array[:pos_num, 1] = pos_neigh
    ret_array[pos_num:, 1] = neg_item[:pos_num * args.neg_rate]
    ret_array[:pos_num, 2] = 1
    ret_array[pos_num:, 2] = 0
    return ret_array


def ndcg_sampling(uid, test_dict, neg_num, item_array):
    """
    Generate ndcg samples for neighboring an item
    param:
        uid - a user in testing set

    return:
        ndcg_array - array(n_neighbors, ndcg_k + 1, 3)
    """
    # pos sampling
    pos_neigh = test_dict.get(uid, [])

    # excluded item
    pos_set = emb_nb.get(uid, [])
    pos_set = np.union1d(pos_neigh, pos_set)

    ndcg_list = []
    for n in pos_neigh:
        # a pos sample
        pos_data = np.hstack([
            np.array([uid]).reshape([-1, 1]),
            np.array([n]).reshape([-1, 1]),
            np.array([1]).reshape([-1, 1])
        ])
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


"""Generate train graph"""
adj_tr = comput_adj(df_emb)


"""Generate users' neighboring items."""
emb_nb = get_neighbors(df_emb)  # {user: item_array}
map_nb = get_neighbors(df_map)
val_nb = get_neighbors(df_warm_val)
test_nb = get_neighbors(df_warm_test)

all_item = {'val': np.unique(df_warm_val['item']),
            'test': np.unique(df_warm_test['item']), }

print('User sparse rate in val: %.4f' % (np.mean([len(v) for v in val_nb.values()]) / len(all_item['val'])))
print('User sparse rate in test: %.4f' % (np.mean([len(v) for v in val_nb.values()]) / len(all_item['test'])))


"""Negative sampling for training."""
neg_sampling = {}

t1 = time.time()
neg_sampling['VAL'] = np.vstack(list(map(lambda x: _neg_samp(x, val_nb), list(val_nb.keys())))).astype(np.int32)
print('Negative sampling for %s val data in %.2f s' % (args.dataset, time.time() - t1))

t1 = time.time()
neg_sampling['TEST'] = np.vstack(list(map(lambda x: _neg_samp(x, test_nb), list(test_nb.keys())))).astype(np.int32)
print('Negative sampling for %s test data in %.2f s' % (args.dataset, time.time() - t1))


"""Get ndcg testing samples"""
t1 = time.time()
test_ndcg_50 = list(map(lambda x: ndcg_sampling(x, test_nb, 50, all_item['test']), list(test_nb.keys())))
print('Build NDGC for %s test@50 data in %.2f s' % (args.dataset, time.time() - t1))


"""Save results"""
para_dict = {}
para_dict['user_num'] = n_warm_user
para_dict['item_num'] = n_warm_item
para_dict['adj_train'] = adj_tr
para_dict['emb_data'] = df_emb[['user', 'item']].values  # mapped data
para_dict['map_data'] = df_map[['user', 'item']].values
para_dict['val_data'] = df_warm_val[['user', 'item']].values
para_dict['test_data'] = df_warm_test[['user', 'item']].values
para_dict['emb_nb'] = emb_nb  # {user: item_array}
para_dict['map_nb'] = map_nb
para_dict['val_nb'] = val_nb
para_dict['test_nb'] = test_nb
para_dict['val_sampling'] = neg_sampling['VAL']  # [[user_id item_id label] for samples of a user]
para_dict['test_sampling'] = neg_sampling['TEST']
para_dict['test_ndcg@50'] = test_ndcg_50  # [array(n_neighbors, ndcg_k + 1, 3)] * n_users
pickle.dump(para_dict, open(store_path + '/warm_dict.pkl', 'wb'))
print('Process %s in %.2f s' % (args.dataset, time.time() - t0))

