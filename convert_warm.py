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
from tqdm import tqdm
import multiprocessing as mul

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
t0 = time.time()
df_emb = pd.read_csv(store_path + '/warm_emb.csv', dtype=np.int)
df_map = pd.read_csv(store_path + '/warm_map.csv', dtype=np.int)
df_warm_val = pd.read_csv(store_path + '/warm_val.csv', dtype=np.int)
df_warm_test = pd.read_csv(store_path + '/warm_test.csv', dtype=np.int)
df_pos = pd.concat([df_emb, df_map, df_warm_val, df_warm_test])
info_dict = pickle.load(open(store_path + '/info.pkl', 'rb'))
user_num = info_dict['user_num']
item_num = info_dict['item_num']
print('Finish load data: ', time.time() - t0)
t0 = time.time()


"""Generate users' neighboring items."""
emb_nb = utils.df_get_neighbors(df_emb)  # {user: item_array}
map_nb = utils.df_get_neighbors(df_map)
pos_nb = utils.df_get_neighbors(df_pos)
val_nb = utils.df_get_neighbors(df_warm_val)
test_nb = utils.df_get_neighbors(df_warm_test)
print('Finish getting neighbor: ', time.time() - t0)
t0 = time.time()

emb_nb_reverse = utils.df_get_neighbors(df_emb, 'item')
map_nb_reverse = utils.df_get_neighbors(df_map, 'item')
pos_nb_reverse = utils.df_get_neighbors(df_pos, 'item')
val_nb_reverse = utils.df_get_neighbors(df_warm_val, 'item')
test_nb_reverse = utils.df_get_neighbors(df_warm_test, 'item')
print('Finish getting neighbor reversely: ', time.time() - t0)
t0 = time.time()

"""build ii nb"""
if args.dataset in ['CiteULike', 'XING']:
    sparse_emb_UINet = utils.get_sparse_UINet(df_emb['user'].values, df_emb['item'].values, user_num, item_num)
    sparse_pos_UINet = utils.get_sparse_UINet(df_pos['user'].values, df_pos['item'].values, user_num, item_num)

    sparse_emb_IUNet = copy.deepcopy(sparse_emb_UINet.T)
    sparse_pos_IUNet = copy.deepcopy(sparse_pos_UINet.T)
    sparse_emb_IINet = sparse_emb_IUNet.dot(sparse_emb_UINet).tocoo()
    print('Finish Net: ', time.time() - t0)
    t0 = time.time()

    # train/val/test split
    warm_item_array = np.array(list(emb_nb_reverse.keys())).astype(np.int)
    train_item = np.random.choice(warm_item_array, len(warm_item_array) * 4 // 5, replace=False)
    val_test_item = np.setdiff1d(warm_item_array, train_item)
    val_item = np.random.choice(val_test_item, (len(warm_item_array)-len(train_item)) // 2, replace=False)
    test_item = np.setdiff1d(val_test_item, val_item)
    print('Finish set split: ', time.time() - t0)
    t0 = time.time()

    # row/col split
    row = sparse_emb_IINet.row
    col = sparse_emb_IINet.col
    mask = [r > c for r, c in tqdm(zip(row, col))]  # 只保留下三角矩阵
    row = row[mask]
    col = col[mask]
    train_row = np.random.choice(train_item, len(train_item) // 2, replace=False)
    train_col = np.setdiff1d(train_item, train_row)
    val_row = np.random.choice(val_item, len(val_item) // 2, replace=False)
    val_col = np.setdiff1d(val_item, val_row)
    test_row = np.random.choice(test_item, len(test_item) // 2, replace=False)
    test_col = np.setdiff1d(test_item, test_row)
    print('Finish row/col split: ', time.time() - t0)
    t0 = time.time()

    # ii_train
    # train_row_label = np.array([r in train_row for r in tqdm(row)], dtype=np.int)
    # train_col_label = np.array([r in train_col for r in tqdm(col)], dtype=np.int)
    sorted_train_row = np.sort(train_row)
    sorted_train_col = np.sort(train_col)
    train_row_label = np.array(list(map(lambda x: utils.binary_search(sorted_train_row, x), tqdm(row))), dtype=np.int)
    train_col_label = np.array(list(map(lambda x: utils.binary_search(sorted_train_col, x), tqdm(col))), dtype=np.int)
    train_label = np.multiply(train_row_label, train_col_label).astype(np.bool)
    train_row = row[train_label]
    train_col = col[train_label]
    df_ii_train = pd.DataFrame(np.vstack([train_row, train_col]).T, columns=['user', 'item'])
    ii_train_nb = utils.df_get_neighbors(df_ii_train, max_nei=item_num // 100)
    print('Finish ii train: ', time.time() - t0)
    t0 = time.time()

    # ii_val
    # val_row_label = np.array([r in val_row for r in tqdm(row)], dtype=np.int)
    # val_col_label = np.array([r in val_col for r in tqdm(col)], dtype=np.int)
    sorted_val_row = np.sort(val_row)
    sorted_val_col = np.sort(val_col)
    val_row_label = np.array(list(map(lambda x: utils.binary_search(sorted_val_row, x), tqdm(row))), dtype=np.int)
    val_col_label = np.array(list(map(lambda x: utils.binary_search(sorted_val_col, x), tqdm(col))), dtype=np.int)
    val_label = np.multiply(val_row_label, val_col_label).astype(np.bool)
    val_row = row[val_label]
    val_col = col[val_label]
    df_ii_val = pd.DataFrame(np.vstack([val_row, val_col]).T, columns=['user', 'item'])
    ii_val_nb = utils.df_get_neighbors(df_ii_val, max_nei=item_num // 100)
    print('Finish ii val: ', time.time() - t0)
    t0 = time.time()

    # ii_test
    # test_row_label = np.array([r in test_row for r in tqdm(row)], dtype=np.int)
    # test_col_label = np.array([r in test_col for r in tqdm(col)], dtype=np.int)
    sorted_test_row = np.sort(test_row)
    sorted_test_col = np.sort(test_col)
    test_row_label = np.array(list(map(lambda x: utils.binary_search(sorted_test_row, x), tqdm(row))), dtype=np.int)
    test_col_label = np.array(list(map(lambda x: utils.binary_search(sorted_test_col, x), tqdm(col))), dtype=np.int)
    test_label = np.multiply(test_row_label, test_col_label).astype(np.bool)
    test_row = row[test_label]
    test_col = col[test_label]
    df_ii_test = pd.DataFrame(np.vstack([test_row, test_col]).T, columns=['user', 'item'])
    ii_test_nb = utils.df_get_neighbors(df_ii_test, max_nei=item_num // 100)
    print('Finish ii test: ', time.time() - t0)
    t0 = time.time()

    sparse_pos_IINet = sparse_pos_IUNet.dot(sparse_pos_UINet).tocoo()
    df_pos_ii = pd.DataFrame(np.vstack([sparse_pos_IINet.row, sparse_pos_IINet.col]).T, columns=['user', 'item'])  # fake user
    ii_pos_nb = utils.df_get_neighbors(df_pos_ii)
    print('Finish ii pos: ', time.time() - t0)
    t0 = time.time()


"""build uu nb"""
if args.dataset in ['LastFM', 'XING']:
    sparse_emb_UINet = utils.get_sparse_UINet(df_emb['user'].values, df_emb['item'].values, user_num, item_num)
    sparse_pos_UINet = utils.get_sparse_UINet(df_pos['user'].values, df_pos['item'].values, user_num, item_num)

    sparse_emb_IUNet = copy.deepcopy(sparse_emb_UINet.T)
    sparse_pos_IUNet = copy.deepcopy(sparse_pos_UINet.T)
    sparse_emb_UUNet = sparse_emb_UINet.dot(sparse_emb_IUNet).tocoo()
    print('Finish Net: ', time.time() - t0)
    t0 = time.time()

    # train/val/test split
    warm_user_array = np.array(list(emb_nb.keys())).astype(np.int)
    train_user = np.random.choice(warm_user_array, len(warm_user_array) * 4 // 5, replace=False)
    val_test_user = np.setdiff1d(warm_user_array, train_user)
    val_user = np.random.choice(val_test_user, (len(warm_user_array)-len(train_user)) // 2, replace=False)
    test_user = np.setdiff1d(val_test_user, val_user)
    print('Finish set split: ', time.time() - t0)
    t0 = time.time()

    # row/col split
    row = sparse_emb_UUNet.row
    col = sparse_emb_UUNet.col
    mask = [r > c for r, c in tqdm(zip(row, col))]  # 只保留下三角矩阵
    row = row[mask]
    col = col[mask]
    train_row = np.random.choice(train_user, len(train_user) // 2, replace=False)
    train_col = np.setdiff1d(train_user, train_row)
    val_row = np.random.choice(val_user, len(val_user) // 2, replace=False)
    val_col = np.setdiff1d(val_user, val_row)
    test_row = np.random.choice(test_user, len(test_user) // 2, replace=False)
    test_col = np.setdiff1d(test_user, test_row)
    print('Finish row/col split: ', time.time() - t0)
    t0 = time.time()

    # uu_train
    # train_row_label = np.array([r in train_row for r in tqdm(row)], dtype=np.int)
    # train_col_label = np.array([r in train_col for r in tqdm(col)], dtype=np.int)
    sorted_train_row = np.sort(train_row)
    sorted_train_col = np.sort(train_col)
    train_row_label = np.array(list(map(lambda x: utils.binary_search(sorted_train_row, x), tqdm(row))), dtype=np.int)
    train_col_label = np.array(list(map(lambda x: utils.binary_search(sorted_train_col, x), tqdm(col))), dtype=np.int)
    train_label = np.multiply(train_row_label, train_col_label).astype(np.bool)
    train_row = row[train_label]
    train_col = col[train_label]
    df_uu_train = pd.DataFrame(np.vstack([train_row, train_col]).T, columns=['user', 'item'])
    uu_train_nb = utils.df_get_neighbors(df_uu_train, max_nei=user_num // 100)
    print('Finish uu train: ', time.time() - t0)
    t0 = time.time()

    # uu_val
    # val_row_label = np.array([r in val_row for r in row], dtype=np.int)
    # val_col_label = np.array([r in val_col for r in col], dtype=np.int)
    sorted_val_row = np.sort(val_row)
    sorted_val_col = np.sort(val_col)
    val_row_label = np.array(list(map(lambda x: utils.binary_search(sorted_val_row, x), tqdm(row))), dtype=np.int)
    val_col_label = np.array(list(map(lambda x: utils.binary_search(sorted_val_col, x), tqdm(col))), dtype=np.int)
    val_label = np.multiply(val_row_label, val_col_label).astype(np.bool)
    val_row = row[val_label]
    val_col = col[val_label]
    df_uu_val = pd.DataFrame(np.vstack([val_row, val_col]).T, columns=['user', 'item'])
    uu_val_nb = utils.df_get_neighbors(df_uu_val, max_nei=user_num // 100)
    print('Finish uu val: ', time.time() - t0)
    t0 = time.time()

    # uu_test
    # test_row_label = np.array([r in test_row for r in row], dtype=np.int)
    # test_col_label = np.array([r in test_col for r in col], dtype=np.int)
    sorted_test_row = np.sort(test_row)
    sorted_test_col = np.sort(test_col)
    test_row_label = np.array(list(map(lambda x: utils.binary_search(sorted_test_row, x), tqdm(row))), dtype=np.int)
    test_col_label = np.array(list(map(lambda x: utils.binary_search(sorted_test_col, x), tqdm(col))), dtype=np.int)
    test_label = np.multiply(test_row_label, test_col_label).astype(np.bool)
    test_row = row[test_label]
    test_col = col[test_label]
    df_uu_test = pd.DataFrame(np.vstack([test_row, test_col]).T, columns=['user', 'item'])
    uu_test_nb = utils.df_get_neighbors(df_uu_test, max_nei=user_num // 100)
    print('Finish uu test: ', time.time() - t0)
    t0 = time.time()

    sparse_pos_UUNet = sparse_pos_UINet.dot(sparse_pos_IUNet).tocoo()
    df_pos_uu = pd.DataFrame(np.vstack([sparse_pos_UUNet.row, sparse_pos_UUNet.col]).T, columns=['user', 'item'])  # fake user
    uu_pos_nb = utils.df_get_neighbors(df_pos_uu)
    print('Finish tt pos: ', time.time() - t0)
    t0 = time.time()


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

if args.dataset in ['CiteULike', 'XING']:
    para_dict['ii_train_nb'] = ii_train_nb
    para_dict['ii_val_nb'] = ii_val_nb
    para_dict['ii_test_nb'] = ii_test_nb
    para_dict['ii_pos_nb'] = ii_pos_nb

if args.dataset in ['LastFM', 'XING']:
    para_dict['uu_train_nb'] = uu_train_nb
    para_dict['uu_val_nb'] = uu_val_nb
    para_dict['uu_test_nb'] = uu_test_nb
    para_dict['uu_pos_nb'] = uu_pos_nb

pickle.dump(para_dict, open(store_path + '/warm_dict.pkl', 'wb'), protocol=4)
print('Store %s in %.2f s' % (args.dataset, time.time() - t0))
