import copy
import time
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from pprint import pprint
import os
import utils
import scipy.sparse as sp

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="CiteULike", help='Dataset to use.')
parser.add_argument('--data_dir', type=str, default="./data/process/", help='Director of the dataset.')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
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
df_pos = pd.concat([df_emb, df_map, df_warm_val, df_warm_test])
info_dict = pickle.load(open(store_path + '/info.pkl', 'rb'))
user_num = info_dict['user_num']
item_num = info_dict['item_num']


"""Generate users' neighboring items."""
emb_nb = utils.df_get_neighbors(df_emb)  # {user: item_array}
map_nb = utils.df_get_neighbors(df_map)
pos_nb = utils.df_get_neighbors(df_pos)
val_nb = utils.df_get_neighbors(df_warm_val)
test_nb = utils.df_get_neighbors(df_warm_test)

emb_nb_reverse = utils.df_get_neighbors(df_emb, 'item')
map_nb_reverse = utils.df_get_neighbors(df_map, 'item')
pos_nb_reverse = utils.df_get_neighbors(df_pos, 'item')
val_nb_reverse = utils.df_get_neighbors(df_warm_val, 'item')
test_nb_reverse = utils.df_get_neighbors(df_warm_test, 'item')


"""build ii nb"""
sparse_emb_UINet = utils.get_sparse_UINet(df_emb['user'].values, df_emb['item'].values, user_num, item_num)
sparse_val_UINet = utils.get_sparse_UINet(df_warm_val['user'].values, df_warm_val['item'].values, user_num, item_num)
sparse_test_UINet = utils.get_sparse_UINet(df_warm_test['user'].values, df_warm_test['item'].values, user_num, item_num)
sparse_pos_UINet = utils.get_sparse_UINet(df_pos['user'].values, df_pos['item'].values, user_num, item_num)

sparse_emb_IUNet = copy.deepcopy(sparse_emb_UINet.T)
sparse_val_IUNet = copy.deepcopy(sparse_val_UINet.T)
sparse_test_IUNet = copy.deepcopy(sparse_test_UINet.T)
sparse_pos_IUNet = copy.deepcopy(sparse_pos_UINet.T)


mask = sp.coo_matrix(np.tril(np.ones((item_num, item_num), dtype=np.int), k=-1))
sparse_emb_IINet = sparse_emb_IUNet.dot(sparse_emb_UINet).multiply(mask).tocoo()

# train/val/test split
warm_item_array = np.array(list(emb_nb_reverse.keys())).astype(np.int)
train_item = np.random.choice(warm_item_array, len(warm_item_array) * 4 // 5, replace=False)
val_test_item = np.setdiff1d(warm_item_array, train_item)
val_item = np.random.choice(val_test_item, (len(warm_item_array)-len(train_item)) // 2, replace=False)
test_item = np.setdiff1d(val_test_item, val_item)

# row/col split
row = sparse_emb_IINet.row
col = sparse_emb_IINet.col
train_row = np.random.choice(train_item, len(train_item) // 2, replace=False)
train_col = np.setdiff1d(train_item, train_row)
val_row = np.random.choice(val_item, len(val_item) // 2, replace=False)
val_col = np.setdiff1d(val_item, val_row)
test_row = np.random.choice(test_item, len(test_item) // 2, replace=False)
test_col = np.setdiff1d(test_item, test_row)

# ii_train
train_row_label = np.array([r in train_row for r in row], dtype=np.int)
train_col_label = np.array([r in train_col for r in col], dtype=np.int)
train_label = np.multiply(train_row_label, train_col_label).astype(np.bool)
train_row = row[train_label]
train_col = col[train_label]
df_ii_train = pd.DataFrame(np.vstack([train_row, train_col]).T, columns=['user', 'item'])
ii_train_nb = utils.df_get_neighbors(df_ii_train, max_nei=item_num // 100)

# ii_val
val_row_label = np.array([r in val_row for r in row], dtype=np.int)
val_col_label = np.array([r in val_col for r in col], dtype=np.int)
val_label = np.multiply(val_row_label, val_col_label).astype(np.bool)
val_row = row[val_label]
val_col = col[val_label]
df_ii_val = pd.DataFrame(np.vstack([val_row, val_col]).T, columns=['user', 'item'])
ii_val_nb = utils.df_get_neighbors(df_ii_val, max_nei=item_num // 100)

# ii_test
test_row_label = np.array([r in test_row for r in row], dtype=np.int)
test_col_label = np.array([r in test_col for r in col], dtype=np.int)
test_label = np.multiply(test_row_label, test_col_label).astype(np.bool)
test_row = row[test_label]
test_col = col[test_label]
df_ii_test = pd.DataFrame(np.vstack([test_row, test_col]).T, columns=['user', 'item'])
ii_test_nb = utils.df_get_neighbors(df_ii_test, max_nei=item_num // 100)


sparse_pos_IINet = sparse_pos_IUNet.dot(sparse_pos_UINet).tocoo()
df_pos_ii = pd.DataFrame(np.vstack([sparse_pos_IINet.row, sparse_pos_IINet.col]).T, columns=['user', 'item'])  # fake user
ii_pos_nb = utils.df_get_neighbors(df_pos_ii)


"""Save results"""
para_dict = {}
para_dict['user_num'] = info_dict['user_num']
para_dict['item_num'] = info_dict['item_num']
para_dict['emb_nb'] = emb_nb  # {user: item_array}
para_dict['map_nb'] = map_nb
para_dict['pos_nb'] = pos_nb
para_dict['val_nb'] = val_nb
para_dict['test_nb'] = test_nb

para_dict['emb_nb_reverse'] = emb_nb_reverse  # {item: user_array}
para_dict['map_nb_reverse'] = map_nb_reverse
para_dict['pos_nb_reverse'] = pos_nb_reverse
para_dict['val_nb_reverse'] = val_nb_reverse
para_dict['test_nb_reverse'] = test_nb_reverse

para_dict['ii_train_nb'] = ii_train_nb
para_dict['ii_val_nb'] = ii_val_nb
para_dict['ii_test_nb'] = ii_test_nb
para_dict['ii_pos_nb'] = ii_pos_nb

pickle.dump(para_dict, open(store_path + '/warm_dict.pkl', 'wb'), protocol=4)
print('Process %s in %.2f s' % (args.dataset, time.time() - t0))
