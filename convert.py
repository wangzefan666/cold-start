"""
Using the processing in lightgcn
"""
import time
import random
import pickle
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="ciaoUI",
                    help='Dataset to use.')
parser.add_argument('--datadir', type=str, default="../datanorm/process/",
                    help='Director of the dataset.')
parser.add_argument('--neg_rate', type=int, default=5,
                    help='The negative sampling rate.')
parser.add_argument('--n_jobs', type=int, default=20,
                    help='Multiprocessing number.')
parser.add_argument('--cold', type=float, default=0.2,
                    help='Cold item Ratio')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
args, _ = parser.parse_known_args()
print('\n'.join([(str(_) + ':' + str(vars(args)[_])) for _ in vars(args).keys()]))

print('Set Seed: ', args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

t0 = time.time()
df_tr = pd.read_csv(args.datadir + args.dataset + '_dftr.csv', header=None)
try:
    df_va = pd.read_csv(args.datadir + args.dataset + '_dfva.csv', header=None)
except:
    pass
df_ts = pd.read_csv(args.datadir + args.dataset + '_dfts.csv', header=None)
if df_tr.shape[1]>3:
    df_tr.columns = ['user', 'item', 'weight', 'timestamp']
    try:
        df_va.columns = ['user', 'item', 'weight', 'timestamp']
    except:
        pass
    df_ts.columns = ['user', 'item', 'weight', 'timestamp']
elif df_tr.shape[1]>2:
    df_tr.columns = ['user', 'item', 'weight']
    try:
        df_va.columns = ['user', 'item', 'weight']
    except:
        pass
    df_ts.columns = ['user', 'item', 'weight']
else:
    df_tr.columns = ['user', 'item']
    try:
        df_va.columns = ['user', 'item']
    except:
        pass
    df_ts.columns = ['user', 'item']
try:
    df = pd.concat([df_tr,df_va,df_ts])
except:
    df = pd.concat([df_tr, df_ts])

user_idx = sorted(list(set(df['user'])))
item_idx = sorted(list(set(df['item'])))

USER_NUM = len(user_idx)
ITEM_NUM = len(item_idx)
NODE_NUM = USER_NUM + ITEM_NUM

def comput_adj(input_df):
    cols = input_df.user.values
    rows = input_df.item.values+USER_NUM

    if 'weight' in input_df.columns:
        values = input_df.weight.values
    else:
        values = np.ones(len(cols))
    adj = sp.csr_matrix((values, (rows, cols)), shape=(NODE_NUM, NODE_NUM))
    adj= adj + adj.T
    return adj

adj_tr = comput_adj(df_tr)
# adj_va = comput_adj(df_va)
# adj_ts = comput_adj(df_ts)

def comput_nei(input_df):
    d = {}
    for _ in input_df.values:
        if _[0] not in d:
            d[_[0]] = []
        d[_[0]].append(_[1])
    return d

tr_items = comput_nei(df_tr)
try:
    va_items = comput_nei(df_va)
    for _ in va_items:
        for i in va_items[_]:
            if i in tr_items.get(_, []):
                print(_,i)
except:
    va_items = {}
ts_items = comput_nei(df_ts)
for _ in ts_items:
    for i in ts_items[_]:
        if i in tr_items.get(_, []):
            print(_,i)
pos_items = comput_nei(df)

ITEM_ARRAY = np.array(list(range(ITEM_NUM)))

def _neg_samp(uid):
    if STATUS == 'TR':
        main_d = tr_items
    elif STATUS == 'VA':
        main_d = va_items
    elif STATUS == 'TS':
        main_d = ts_items
    else:
        raise 'STATUS not defined'
    supp_d = pos_items
    pos_item = main_d.get(uid, [])
    pos_num = len(pos_item)
    if pos_num == 0:
        return np.zeros((0,3)).astype(np.int)
    pos_set = set(supp_d.get(uid, []))
    ret_array = np.zeros((pos_num*(args.neg_rate+1),3)).astype(np.int)
    neg_items = np.random.choice(ITEM_ARRAY, 5*args.neg_rate*pos_num)
    samp_neg= []
    t = 0
    while not samp_neg:
        neg_items = np.random.choice(ITEM_ARRAY, 10*args.neg_rate*pos_num)
        samp_neg = list(filter(
            lambda x: x not in pos_set, neg_items))[:pos_num*args.neg_rate]
        t+=1
        if t==3:
            return np.zeros((0,3)).astype(np.int)
    ret_array[:,0] = uid
    ret_array[:pos_num,2] = 1
    ret_array[pos_num:,2] =0
    ret_array[:pos_num,1] = pos_item
    ret_array[pos_num:,1] = samp_neg
    return ret_array

STATUS = 'TR'
with Pool(args.n_jobs) as pool:
    t1 = time.time()
    tr_lbs = np.vstack(
        pool.map(_neg_samp, user_idx)).astype(int)
    print('Negative sampling for %s VA data in %.2f s' % (
        args.dataset, time.time() - t1))
    t1 = time.time()
STATUS = 'VA'
with Pool(args.n_jobs) as pool:
    t1 = time.time()
    va_lbs = np.vstack(
        pool.map(_neg_samp, user_idx)).astype(int)
    print('Negative sampling for %s VA data in %.2f s' % (
        args.dataset, time.time() - t1))
    t1 = time.time()
STATUS = 'TS'
with Pool(args.n_jobs) as pool:
    ts_lbs = np.vstack(
        pool.map(_neg_samp, user_idx)).astype(int)
    print('Negative sampling for %s TS data in %.2f s' % (
        args.dataset, time.time() - t1))

##### Get NDCG Samps #####
def ndcg_samp(idx):
    neg_num = NDCG_K
    neigh_pos = ts_items.get(idx, [])
    if len(neigh_pos) == 0:
        return 0
    ndcg_list = []
    pos_set = set(pos_items.get(idx, []))
    for n in neigh_pos:
        pos_data = np.hstack([
            np.array([idx]).reshape([-1, 1]),
            np.array([n]).reshape([-1, 1]),
            np.array([1]).reshape([-1, 1])
        ])
        neg_item = np.random.choice(ITEM_ARRAY, 3 * neg_num)
        neg_item = neg_item[[_ not in pos_set for _ in neg_item]]
        while len(neg_item)< NDCG_K:
            new_neg = np.random.choice(ITEM_ARRAY, 3 * neg_num)
            new_neg = new_neg[[_ not in pos_set for _ in new_neg]]
            neg_item = np.hstack([new_neg, neg_item])        
        neg_item = neg_item[:neg_num]
        neg_data = np.hstack([
            np.array([idx] * len(neg_item)).reshape([-1, 1]),
            neg_item.reshape([-1, 1]),
            np.array([0] * len(neg_item)).reshape([-1, 1])
        ])
        label_data = np.vstack([pos_data, neg_data])
        ndcg_list.append(label_data)
    ndcg_array = np.stack(ndcg_list)
    return ndcg_array


NDCG_K = 50
t1 = time.time()
with Pool(args.n_jobs) as pool:
    ndcg_50 = pool.map(ndcg_samp, list(ts_items.keys()))
    ndcg_50 = list(filter(lambda x: not isinstance(x, int), ndcg_50))
print('Build NDGCTest for %s testing data in %.2f s' % (
    args.dataset, time.time() - t1))

para_dict = {}
para_dict['user_num'] = USER_NUM
para_dict['item_num'] = ITEM_NUM
para_dict['tr_items'] = tr_items
para_dict['va_items'] = va_items
para_dict['ts_items'] = ts_items
para_dict['pos_items'] = pos_items
para_dict['tr_lbs'] = va_lbs
para_dict['va_lbs'] = va_lbs
para_dict['ts_lbs'] = ts_lbs
para_dict['ts_ndcg'] = ndcg_50
sp.save_npz(args.datadir + args.dataset + '_adj_train.npz', adj_tr)
pickle.dump(para_dict, open(args.datadir + args.dataset + '_map.pkl', 'wb'))

print('Process %s in %.2f s' % (args.dataset, time.time() - t0))