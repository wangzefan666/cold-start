import random
import argparse
import numpy as np
import pandas as pd
from pprint import pprint
from scipy import sparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="XING", help='Dataset to use.')
parser.add_argument('--data_dir', type=str, default="./data/", help='Director of the dataset.')
parser.add_argument('--warm_ratio', type=float, default=0.8, help='Warm ratio of all items')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--warm_split', nargs='?', default='[0.65, 0.15, 0.1, 0.1]')
parser.add_argument('--cold_split', nargs='?', default='[0.5, 0.5]')
args = parser.parse_args()
args.warm_split = eval(args.warm_split)
args.cold_split = eval(args.cold_split)
pprint(vars(args))

# set seed
random.seed(args.seed)
np.random.seed(args.seed)

store_path = args.data_dir + 'process/' + f'{args.dataset}/'
if not os.path.exists(store_path):
    os.makedirs(store_path)


t0 = time.time()
"""读取数据"""
df = pd.read_csv(args.data_dir + args.dataset + '.csv')
origin_len = df.shape[0]
df = df.drop_duplicates(['user', 'item']).reset_index(drop=True)
new_len = df.shape[0]
print('Duplicated :%d -> %d' % (origin_len, new_len))

USER_NUM = len(set(df['user']))
ITEM_NUM = len(set(df['item']))
print('User: %d\tItem: %d' % (USER_NUM, ITEM_NUM))

item_content = sparse.load_npz(args.data_dir + args.dataset + f'_item_content.npz')
item_content = item_content.tolil()
user_content = sparse.load_npz(args.data_dir + args.dataset + f'_user_content.npz')
user_content = user_content.tolil()


"""对原 user 和 item 的 id 进行映射，并存储映射关系"""
i2user = list(set(df.iloc[:, 0]))
i2item = list(set(df.iloc[:, 1]))
user2i = {int(user): i for i, user in enumerate(i2user)}
item2i = {int(item): i for i, item in enumerate(i2item)}

df.iloc[:, 0] = np.array(list(map(user2i.get, df.iloc[:, 0])))
df.iloc[:, 1] = np.array(list(map(item2i.get, df.iloc[:, 1])))

# 调整为 map 后的 content 顺序
item_content = item_content[i2item]
sparse.save_npz(store_path + 'item_content.npz', item_content.tocsr())
user_content = user_content[i2user]
sparse.save_npz(store_path + 'user_content.npz', user_content.tocsr())

user_map_file = [[org_id, remap_id] for remap_id, org_id in enumerate(i2user)]
user_map_file = np.vstack(user_map_file).astype(np.int32)
item_map_file = [[org_id, remap_id] for remap_id, org_id in enumerate(i2item)]
item_map_file = np.vstack(item_map_file).astype(np.int32)

pd.DataFrame(user_map_file).to_csv(store_path + 'raw_user_map.csv', header=['org_id', 'remap_id'], index=False)
pd.DataFrame(item_map_file).to_csv(store_path + 'raw_item_map.csv', header=['org_id', 'remap_id'], index=False)


"""warm/cold splitting"""
item_group = df.groupby(by='item')
item_group = np.array([g[1].index for g in item_group], dtype=object)
print(f'Max sparse rate (of a item): %.4f' % (max([len(g) for g in item_group]) / USER_NUM))

user_group = df.groupby(by='user')
user_group = np.array([g[1].index for g in user_group], dtype=object)
print(f'Max sparse rate (of a user): %.4f' % (max([len(g) for g in user_group]) / ITEM_NUM))
print('=' * 30)

# warm/cold item record splitting
n_warm_item_group = int(args.warm_ratio * len(item_group))
n_cold_item_group = len(item_group) - n_warm_item_group

warm_item_group = item_group[:n_warm_item_group]  # warm
warm_item_records = np.hstack(warm_item_group)

cold_item_group = item_group[n_warm_item_group:]  # cold
cold_item_records = np.hstack(cold_item_group)

# warm/cold user record splitting
n_warm_user_group = int(args.warm_ratio * len(user_group))
n_cold_user_group = len(user_group) - n_warm_user_group

warm_user_group = user_group[:n_warm_user_group]
warm_user_records = np.hstack(warm_user_group)

cold_user_group = user_group[n_warm_user_group:]
cold_user_records = np.hstack(cold_user_group)

# 分出 warm_both_records, 即 warm_item_records 和 warm_user_records 的并集
warm_both_records = np.intersect1d(warm_item_records, warm_user_records)
warm_item_idx = np.unique(df.iloc[warm_both_records]['item'])  # 此处有玄机：过滤后再...因为 warm object 必须经过训练
warm_user_idx = np.unique(df.iloc[warm_both_records]['user'])

# 分出 cold_both_records, 即 cold_item_records 和 cold_user_records 的并集
cold_both_records = np.intersect1d(cold_item_records, cold_user_records)


"""subset splitting"""
# warm -> emb/map/val/test
n_warm_map = int(args.warm_split[1] * len(warm_both_records))
n_warm_val = int(args.warm_split[2] * len(warm_both_records))
n_warm_test = int(args.warm_split[3] * len(warm_both_records))
n_warm_emb = len(warm_both_records) - n_warm_map - n_warm_val - n_warm_test

np.random.shuffle(warm_both_records)
warm_emb_records = warm_both_records[:n_warm_emb]
warm_map_records = warm_both_records[n_warm_emb:(n_warm_emb+n_warm_map)]
warm_val_records = warm_both_records[(n_warm_emb+n_warm_map):(n_warm_emb+n_warm_map+n_warm_val)]
warm_test_records = warm_both_records[(n_warm_emb+n_warm_map+n_warm_val):]

# Move the map records whose user/item don't emerge in emb set into emb set. It hurts map more.
warm_emb_user_set = np.unique(df.iloc[warm_emb_records]['user'])
df_warm_map = df.iloc[warm_map_records]
idx_to_move = df_warm_map[True ^ df_warm_map['user'].isin(warm_emb_user_set)].index
warm_map_records = np.setdiff1d(warm_map_records, idx_to_move)
warm_emb_records = np.hstack([warm_emb_records, idx_to_move])

warm_emb_item_set = np.unique(df.iloc[warm_emb_records]['item'])
df_warm_map = df.iloc[warm_map_records]
idx_to_move = df_warm_map[True ^ df_warm_map['item'].isin(warm_emb_item_set)].index
warm_map_records = np.setdiff1d(warm_map_records, idx_to_move)
warm_emb_records = np.hstack([warm_emb_records, idx_to_move])

# Move the val records whose user/item don't emerge in emb set into emb set.
warm_emb_user_set = np.unique(df.iloc[warm_emb_records]['user'])
df_warm_val = df.iloc[warm_val_records]
idx_to_move = df_warm_val[True ^ df_warm_val['user'].isin(warm_emb_user_set)].index
warm_val_records = np.setdiff1d(warm_val_records, idx_to_move)
warm_emb_records = np.hstack([warm_emb_records, idx_to_move])

warm_emb_item_set = np.unique(df.iloc[warm_emb_records]['item'])
df_warm_val = df.iloc[warm_val_records]
idx_to_move = df_warm_val[True ^ df_warm_val['item'].isin(warm_emb_item_set)].index
warm_val_records = np.setdiff1d(warm_val_records, idx_to_move)
warm_emb_records = np.hstack([warm_emb_records, idx_to_move])

# Move the test records whose user/item don't emerge in emb set into emb set.
warm_emb_user_set = np.unique(df.iloc[warm_emb_records]['user'])
df_warm_test = df.iloc[warm_test_records]
idx_to_move = df_warm_test[True ^ df_warm_test['user'].isin(warm_emb_user_set)].index
warm_test_records = np.setdiff1d(warm_test_records, idx_to_move)
warm_emb_records = np.hstack([warm_emb_records, idx_to_move])

warm_emb_item_set = np.unique(df.iloc[warm_emb_records]['item'])
df_warm_test = df.iloc[warm_test_records]
idx_to_move = df_warm_test[True ^ df_warm_test['item'].isin(warm_emb_item_set)].index
warm_test_records = np.setdiff1d(warm_test_records, idx_to_move)
warm_emb_records = np.hstack([warm_emb_records, idx_to_move])

# store df
df_warm_emb_records = df.iloc[warm_emb_records]
df_warm_map_records = df.iloc[warm_map_records]
df_warm_val_records = df.iloc[warm_val_records]
df_warm_test_records = df.iloc[warm_test_records]

df_warm_emb_records.to_csv(store_path + 'warm_emb.csv', index=False)
df_warm_map_records.to_csv(store_path + 'warm_map.csv', index=False)
df_warm_val_records.to_csv(store_path + 'warm_val.csv', index=False)
df_warm_test_records.to_csv(store_path + 'warm_test.csv', index=False)

print('[warm]\tuser\titem\trecord')
print('emb\t%d\t%d\t%d' %
      (len(np.unique(df_warm_emb_records['user'])), len(np.unique(df_warm_emb_records['item'])), len(warm_emb_records)))
print('map\t%d\t%d\t%d' %
      (len(np.unique(df_warm_map_records['user'])), len(np.unique(df_warm_map_records['item'])), len(warm_map_records)))
print('val\t%d\t%d\t%d' %
      (len(np.unique(df_warm_val_records['user'])), len(np.unique(df_warm_val_records['item'])), len(warm_val_records)))
print('test\t%d\t%d\t%d' %
      (len(np.unique(df_warm_test_records['user'])), len(np.unique(df_warm_test_records['item'])), len(warm_test_records)))
print('=' * 30)

# cold item
# 先分组后过滤，UI需要知道分组信息
np.random.shuffle(cold_item_records)
df_cold_item = df.iloc[cold_item_records]
cold_item_group = df_cold_item.groupby(by='item')
cold_item_group = np.array([g[1].index for g in cold_item_group], dtype=object)

n_cold_item_val_group = int(args.cold_split[0] * len(cold_item_group))
n_cold_item_test_group = len(cold_item_group) - n_cold_item_val_group
cold_item_val_records = np.hstack(cold_item_group[:n_cold_item_val_group].tolist())
cold_item_test_records = np.hstack(cold_item_group[n_cold_item_val_group:].tolist())

# item 分组信息
cold_item_val = np.unique(df.iloc[cold_item_val_records]['item'])
cold_item_test = np.unique(df.iloc[cold_item_test_records]['item'])

# 过滤出 cold_user_records 里含 warm user 的 record
df_cold_item_val = df.iloc[cold_item_val_records, :]
cold_item_val_records = df_cold_item_val[False ^ df_cold_item_val['user'].isin(warm_user_idx)].index
cold_item_val_records = np.array(cold_item_val_records, dtype=np.int32)

df_cold_item_test = df.iloc[cold_item_test_records, :]
cold_item_test_records = df_cold_item_test[False ^ df_cold_item_test['user'].isin(warm_user_idx)].index
cold_item_test_records = np.array(cold_item_test_records, dtype=np.int32)

df_cold_item_val_records = df.iloc[cold_item_val_records]
df_cold_item_test_records = df.iloc[cold_item_test_records]
df_cold_item_val_records.to_csv(store_path + 'cold_item_val.csv', index=False)
df_cold_item_test_records.to_csv(store_path + 'cold_item_test.csv', index=False)

print('[cold]\n[item]\tuser\titem\trecord')
print('val\t%d\t%d\t%d' % (len(np.unique(df_cold_item_val_records['user'])),
                           len(np.unique(df_cold_item_val_records['item'])),
                           len(cold_item_val_records)))
print('test\t%d\t%d\t%d' % (len(np.unique(df_cold_item_test_records['user'])),
                            len(np.unique(df_cold_item_test_records['item'])),
                            len(cold_item_test_records)))
print('=' * 30)

# cold user
# 先分组后过滤，UI需要知道分组信息
np.random.shuffle(cold_user_records)
df_cold_user = df.iloc[cold_user_records]
cold_user_group = df_cold_user.groupby(by='user')
cold_user_group = np.array([g[1].index for g in cold_user_group], dtype=object)

n_cold_user_val_group = int(args.cold_split[0] * len(cold_user_group))
n_cold_user_test_group = len(cold_user_group) - n_cold_user_val_group
cold_user_val_records = np.hstack(cold_user_group[:n_cold_user_val_group].tolist())
cold_user_test_records = np.hstack(cold_user_group[n_cold_user_val_group:].tolist())

# user 分组信息
cold_user_val = np.unique(df.iloc[cold_user_val_records]['user'])
cold_user_test = np.unique(df.iloc[cold_user_test_records]['user'])

# 过滤出 cold_user_records 里含 warm item 的 record
df_cold_user_val = df.iloc[cold_user_val_records, :]
cold_user_val_records = df_cold_user_val[False ^ df_cold_user_val['item'].isin(warm_item_idx)].index
cold_user_val_records = np.array(cold_user_val_records, dtype=np.int32)

df_cold_user_test = df.iloc[cold_user_test_records, :]
cold_user_test_records = df_cold_user_test[False ^ df_cold_user_test['item'].isin(warm_item_idx)].index
cold_user_test_records = np.array(cold_user_test_records, dtype=np.int32)

df_cold_user_val_records = df.iloc[cold_user_val_records, :]
df_cold_user_test_records = df.iloc[cold_user_test_records, :]
df_cold_user_val_records.to_csv(store_path + 'cold_user_val.csv', index=False)
df_cold_user_test_records.to_csv(store_path + 'cold_user_test.csv', index=False)

print('[user]\tuser\titem\trecord')
print('val\t%d\t%d\t%d' % (len(np.unique(df_cold_user_val_records['user'])),
                           len(np.unique(df_cold_user_val_records['item'])),
                           len(cold_user_val_records)))
print('test\t%d\t%d\t%d' % (len(np.unique(df_cold_user_test_records['user'])),
                            len(np.unique(df_cold_user_test_records['item'])),
                            len(cold_user_test_records)))
print('=' * 30)

# cold item cold user
np.random.shuffle(cold_both_records)
df_cold_both = df.iloc[cold_both_records]

cold_both_item_val_records = df_cold_both[False ^ df_cold_both['item'].isin(cold_item_val)].index  # cold item val idx in cold both
cold_both_user_val_records = df_cold_both[False ^ df_cold_both['user'].isin(cold_user_val)].index  # cold user val idx in cold both
cold_both_val_records = np.intersect1d(cold_both_item_val_records, cold_both_user_val_records)

cold_both_item_test_records = df_cold_both[False ^ df_cold_both['item'].isin(cold_item_test)].index
cold_both_user_test_records = df_cold_both[False ^ df_cold_both['user'].isin(cold_user_test)].index
cold_both_test_records = np.intersect1d(cold_both_item_test_records, cold_both_user_test_records)

df_cold_both_val_records = df.iloc[cold_both_val_records]
df_cold_both_test_records = df.iloc[cold_both_test_records]
df_cold_both_val_records.to_csv(store_path + 'cold_both_val.csv', index=False)
df_cold_both_test_records.to_csv(store_path + 'cold_both_test.csv', index=False)

print('[both]\tuser\titem\trecord')
print('val\t%d\t%d\t%d' % (len(np.unique(df_cold_both_val_records['user'])),
                           len(np.unique(df_cold_both_val_records['item'])),
                           len(cold_both_val_records)))
print('test\t%d\t%d\t%d' % (len(np.unique(df_cold_both_test_records['user'])),
                            len(np.unique(df_cold_both_test_records['item'])),
                            len(cold_both_test_records)))
print('Process %s in %.2f s' % (args.dataset, time.time() - t0))
print('=' * 30)

