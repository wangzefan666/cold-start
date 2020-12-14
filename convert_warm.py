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
info_dict = pickle.load(open(store_path + '/info.pkl', 'rb'))


"""Generate users' neighboring items."""
emb_nb = utils.df_get_neighbors(df_emb)  # {user: item_array}
map_nb = utils.df_get_neighbors(df_map)
pos_nb = utils.df_get_neighbors(pd.concat([df_emb, df_map]))
val_nb = utils.df_get_neighbors(df_warm_val)
test_nb = utils.df_get_neighbors(df_warm_test)


"""Save results"""
para_dict = {}
para_dict['user_num'] = info_dict['user_num']
para_dict['item_num'] = info_dict['item_num']
para_dict['emb_nb'] = emb_nb  # {user: item_array}
para_dict['map_nb'] = map_nb
para_dict['pos_nb'] = pos_nb
para_dict['val_nb'] = val_nb
para_dict['test_nb'] = test_nb

pickle.dump(para_dict, open(store_path + '/warm_dict.pkl', 'wb'), protocol=4)
print('Process %s in %.2f s' % (args.dataset, time.time() - t0))
