import pickle
import random
import argparse
import numpy as np
import pandas as pd
from pprint import pprint
from scipy import sparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="LastFM", help='Dataset to use.')
parser.add_argument('--data_dir', type=str, default="./data/", help='Director of the dataset.')
parser.add_argument('--warm_ratio', type=float, default=0.8, help='Warm ratio of all items')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--warm_split', nargs='?', default='[0.65, 0.15, 0.1, 0.1]')
parser.add_argument('--cold_split', nargs='?', default='[0.5, 0.5]')
parser.add_argument('--cold_object', type=str, default='user')
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
info_dict = {'user_num': USER_NUM, 'item_num': ITEM_NUM}
pickle.dump(info_dict, open(store_path + 'info.pkl', 'wb'))
print('User: %d\tItem: %d' % (USER_NUM, ITEM_NUM))

content = sparse.load_npz(args.data_dir + args.dataset + f'_{args.cold_object}_content.npz')
content = content.tolil()


"""对原 user 和 item 的 id 进行映射，并存储映射关系"""
i2user = list(set(df.iloc[:, 0]))
i2item = list(set(df.iloc[:, 1]))
user2i = {int(user): i for i, user in enumerate(i2user)}
item2i = {int(item): i for i, item in enumerate(i2item)}

df.iloc[:, 0] = np.array(list(map(user2i.get, df.iloc[:, 0])))
df.iloc[:, 1] = np.array(list(map(item2i.get, df.iloc[:, 1])))

# 调整为 map 后的 content 顺序
content = content[i2user] if args.cold_object == 'user' else content[i2item]
sparse.save_npz(store_path + f'{args.cold_object}_content.npz', content.tocsr())

user_map_file = [[org_id, remap_id] for remap_id, org_id in enumerate(i2user)]
user_map_file = np.vstack(user_map_file).astype(np.int32)
item_map_file = [[org_id, remap_id] for remap_id, org_id in enumerate(i2item)]
item_map_file = np.vstack(item_map_file).astype(np.int32)

pd.DataFrame(user_map_file).to_csv(store_path + 'raw_user_map.csv', header=['org_id', 'remap_id'], index=False)
pd.DataFrame(item_map_file).to_csv(store_path + 'raw_item_map.csv', header=['org_id', 'remap_id'], index=False)


"""warm/cold splitting"""
group = df.groupby(by=args.cold_object)
# (object_id, record_ids) for every group(user).  _[1].index is [record_ids]
group = np.array([g[1].index for g in group], dtype=object)
np.random.shuffle(group)

n_warm_group = int(args.warm_ratio * len(group))
n_cold_group = len(group) - n_warm_group
warm_idx = np.hstack(group[:n_warm_group].tolist())
cold_idx = np.hstack(group[n_warm_group:].tolist())

print(f'Max sparse rate (of a {args.cold_object}): %.4f' %
      (max([len(g) for g in group]) / (ITEM_NUM if args.cold_object == 'user' else USER_NUM)))
print('=' * 30)

# 分出 cold_idx 里含 warm user(item) 的 record
warm_object = 'user' if args.cold_object == 'item' else 'item'
warm_object_set = np.unique(df.iloc[warm_idx, :][warm_object])
df_cold = df.iloc[cold_idx]
cold_idx = df_cold[False ^ df_cold[warm_object].isin(warm_object_set)].index
cold_idx = np.array(cold_idx, dtype=np.int32)

"""subset splitting"""
# warm(interaction) -> emb/map/val/test
n_warm_map = int(args.warm_split[1] * len(warm_idx))
n_warm_val = int(args.warm_split[2] * len(warm_idx))
n_warm_test = int(args.warm_split[3] * len(warm_idx))
n_warm_emb = len(warm_idx) - n_warm_map - n_warm_val - n_warm_test

np.random.shuffle(warm_idx)
warm_emb_idx = warm_idx[:n_warm_emb]
warm_map_idx = warm_idx[n_warm_emb:(n_warm_emb+n_warm_map)]
warm_val_idx = warm_idx[(n_warm_emb+n_warm_map):(n_warm_emb+n_warm_map+n_warm_val)]
warm_test_idx = warm_idx[(n_warm_emb+n_warm_map+n_warm_val):]

# Move the map records whose user/item don't emerge in emb set into emb set. It hurts map more.
warm_emb_user_set = np.unique(df.iloc[warm_emb_idx]['user'])
df_warm_map = df.iloc[warm_map_idx]
idx_to_move = df_warm_map[True ^ df_warm_map['user'].isin(warm_emb_user_set)].index
warm_map_idx = np.setdiff1d(warm_map_idx, idx_to_move)
warm_emb_idx = np.hstack([warm_emb_idx, idx_to_move])

warm_emb_item_set = np.unique(df.iloc[warm_emb_idx]['item'])
df_warm_map = df.iloc[warm_map_idx]
idx_to_move = df_warm_map[True ^ df_warm_map['item'].isin(warm_emb_item_set)].index
warm_map_idx = np.setdiff1d(warm_map_idx, idx_to_move)
warm_emb_idx = np.hstack([warm_emb_idx, idx_to_move])

# Move the val records whose user/item don't emerge in emb set into emb set.
warm_emb_user_set = np.unique(df.iloc[warm_emb_idx]['user'])
df_warm_val = df.iloc[warm_val_idx]
idx_to_move = df_warm_val[True ^ df_warm_val['user'].isin(warm_emb_user_set)].index
warm_val_idx = np.setdiff1d(warm_val_idx, idx_to_move)
warm_emb_idx = np.hstack([warm_emb_idx, idx_to_move])

warm_emb_item_set = np.unique(df.iloc[warm_emb_idx]['item'])
df_warm_val = df.iloc[warm_val_idx]
idx_to_move = df_warm_val[True ^ df_warm_val['item'].isin(warm_emb_item_set)].index
warm_val_idx = np.setdiff1d(warm_val_idx, idx_to_move)
warm_emb_idx = np.hstack([warm_emb_idx, idx_to_move])

# Move the test records whose user/item don't emerge in emb set into emb set.
warm_emb_user_set = np.unique(df.iloc[warm_emb_idx]['user'])
df_warm_test = df.iloc[warm_test_idx]
idx_to_move = df_warm_test[True ^ df_warm_test['user'].isin(warm_emb_user_set)].index
warm_test_idx = np.setdiff1d(warm_test_idx, idx_to_move)
warm_emb_idx = np.hstack([warm_emb_idx, idx_to_move])

warm_emb_item_set = np.unique(df.iloc[warm_emb_idx]['item'])
df_warm_test = df.iloc[warm_test_idx]
idx_to_move = df_warm_test[True ^ df_warm_test['item'].isin(warm_emb_item_set)].index
warm_test_idx = np.setdiff1d(warm_test_idx, idx_to_move)
warm_emb_idx = np.hstack([warm_emb_idx, idx_to_move])

# store df
df_warm_emb = df.iloc[warm_emb_idx]
df_warm_map = df.iloc[warm_map_idx]
df_warm_val = df.iloc[warm_val_idx]
df_warm_test = df.iloc[warm_test_idx]

df_warm_emb.to_csv(store_path + 'warm_emb.csv', index=False)
df_warm_map.to_csv(store_path + 'warm_map.csv', index=False)
df_warm_val.to_csv(store_path + 'warm_val.csv', index=False)
df_warm_test.to_csv(store_path + 'warm_test.csv', index=False)

print('[warm]\tuser\titem\trecord')
print('emb\t%d\t%d\t%d' %
      (len(np.unique(df_warm_emb['user'])), len(np.unique(df_warm_emb['item'])), len(warm_emb_idx)))
print('map\t%d\t%d\t%d' %
      (len(np.unique(df_warm_map['user'])), len(np.unique(df_warm_map['item'])), len(warm_map_idx)))
print('val\t%d\t%d\t%d' %
      (len(np.unique(df_warm_val['user'])), len(np.unique(df_warm_val['item'])), len(warm_val_idx)))
print('test\t%d\t%d\t%d' %
      (len(np.unique(df_warm_test['user'])), len(np.unique(df_warm_test['item'])), len(warm_test_idx)))
print('=' * 30)

# cold(object) -> val/test
np.random.shuffle(cold_idx)
df_cold = df.iloc[cold_idx, :]
cold_group = df_cold.groupby(by=args.cold_object)
cold_group = np.array([g[1].index for g in cold_group], dtype=object)

n_cold_val_group = int(args.cold_split[0] * len(cold_group))
n_cold_test_group = len(cold_group) - n_cold_val_group
cold_val_idx = np.hstack(cold_group[:n_cold_val_group].tolist())
cold_test_idx = np.hstack(cold_group[n_cold_val_group:].tolist())

df_cold_val = df.iloc[cold_val_idx, :]
df_cold_test = df.iloc[cold_test_idx, :]

df_cold_val.to_csv(store_path + f'cold_{args.cold_object}_val.csv', index=False)
df_cold_test.to_csv(store_path + f'cold_{args.cold_object}_test.csv', index=False)

print('[cold]\tuser\titem\trecord')
print('val\t%d\t%d\t%d' %
      (len(np.unique(df_cold_val['user'])), len(np.unique(df_cold_val['item'])), len(cold_val_idx)))
print('test\t%d\t%d\t%d' %
      (len(np.unique(df_cold_test['user'])), len(np.unique(df_cold_test['item'])), len(cold_test_idx)))
print('=' * 30)

print('Process %s in %.2f s' % (args.dataset, time.time() - t0))

