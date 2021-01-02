import copy
import time
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from pprint import pprint
import utils
import scipy.sparse as sp

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="CiteULike", help='Dataset to use.')
parser.add_argument('--data_dir', type=str, default="./data/process/", help='Director of the dataset.')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--cold_object', type=str, default='item')
args = parser.parse_args()
pprint(vars(args))

random.seed(args.seed)
np.random.seed(args.seed)
t0 = time.time()

"""read data from file"""
df_val = pd.read_csv(args.data_dir + args.dataset + f'/cold_{args.cold_object}_val.csv', dtype=np.int)
df_test = pd.read_csv(args.data_dir + args.dataset + f'/cold_{args.cold_object}_test.csv', dtype=np.int)
info_dict = pickle.load(open(args.data_dir + args.dataset + '/info.pkl', 'rb'))
user_num = info_dict['user_num']
item_num = info_dict['item_num']


"""Generate users' neighboring items."""
val_nb = utils.df_get_neighbors(df_val)
test_nb = utils.df_get_neighbors(df_test)

val_nb_reverse = utils.df_get_neighbors(df_val, 'item')
test_nb_reverse = utils.df_get_neighbors(df_test, 'item')


"""Save results"""
para_dict = {}
para_dict['user_num'] = info_dict['user_num']
para_dict['item_num'] = info_dict['item_num']
para_dict['val_nb'] = val_nb
para_dict['test_nb'] = test_nb
para_dict['val_nb_reverse'] = val_nb_reverse
para_dict['test_nb_reverse'] = test_nb_reverse


pickle.dump(para_dict, open(args.data_dir + args.dataset + f'/cold_{args.cold_object}_dict.pkl', 'wb'), protocol=4)
print('Process %s in %.2f s' % (args.dataset, time.time() - t0))
