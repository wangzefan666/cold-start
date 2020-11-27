import random
import argparse
from collections import Counter
import numpy as np
import pandas as pd
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="ciaoUI", help='Dataset to use.')
parser.add_argument('--datadir', type=str, default="./data/", help='Director of the dataset.')
parser.add_argument('--ratio', nargs='?', default='[7, 1.5, 2.3]', help='Output sizes of every layer')
parser.add_argument('--seed', type=int, default=42, help='random seed')
args = parser.parse_args()
args.ratio = eval(args.ratio)
pprint(vars(args))

# set seed
random.seed(args.seed)
np.random.seed(args.seed)


def split(ind):
    """
    Split records of a user into train/val/test three sets.

    param:
        ind - [adjacent items of a user]

    return:
        tr_idx - train items
        val_idx - validation items
        ts_idx - test items
    """
    n_records = len(ind)
    ind = np.array(ind)
    if n_records > np.sum(args.ratio):
        val_end = int(n_records * args.ratio[1] / np.sum(args.ratio))
        ts_end = int(n_records * (args.ratio[1] + args.ratio[2]) / np.sum(args.ratio))
    elif n_records > 3:
        v_len = n_records // 3
        val_end = v_len
        ts_end = 2 * v_len
    elif n_records > 1:
        rand = np.random.rand()
        if rand < args.ratio[1] / (args.ratio[1] + args.ratio[2]):
            val_end, ts_end = 1, 1
        else:
            val_end, ts_end = 0, 1
    else:
        val_end, ts_end = 0, 0

    np.random.shuffle(ind)
    val_idx = ind[:val_end]
    ts_idx = ind[val_end:ts_end]
    tr_idx = ind[ts_end:]
    assert len(val_idx) + len(ts_idx) + len(tr_idx) == n_records
    return tr_idx, val_idx, ts_idx


def sort_by_user(input_df):
    """
    To save records group by users in file.
    return:
        outs - ['user_id record_ids']
    """
    group_df = input_df.groupby(by='user')
    outs = []
    for g in group_df:
        sub_df = g[1]
        out = [g[0]]  # a user
        out.extend(list(sub_df.iloc[:, 1].values))  # records of a user
        out = ' '.join([str(_) for _ in out])
        outs.append(out)
    return outs


def move_cold_item_to_train(train, test):
    """
    Move the test(val) records whose items are not in train set into train set.
    """
    item_count = Counter(train.iloc[:, 1])
    cold_list = []
    warm_list = []
    for i, item in enumerate(test.item.values):
        if item_count.get(item, 0) < 1:  # if item not in Counter, return the second param of get()
            cold_list.append(i)
        else:
            warm_list.append(i)
    train = train.append(test.iloc[cold_list, :])
    test = test.iloc[warm_list, :]
    return train, test


"""读取数据"""
df = pd.read_csv(args.datadir + args.dataset + '.csv')
org_len = df.shape[0]
# drop_duplicates 去除某几列下重复的行数据
# reset_index(drop) 重置索引，drop决定原来的索引是否变成列'index'
df = df.drop_duplicates(['user', 'item']).reset_index(drop=True)
new_len = df.shape[0]
print('Duplicated :%d -> %d' % (org_len, new_len))


"""对原user和item的id进行映射，并存储映射关系"""
if args.dataset != 'wechat':
    i2user = list(set(df.iloc[:, 0]))
    i2item = list(set(df.iloc[:, 1]))
    user2i = {int(user): i for i, user in enumerate(i2user)}
    item2i = {int(item): i for i, item in enumerate(i2item)}

    df.iloc[:, 0] = np.array(list(map(user2i.get, df.iloc[:, 0])))
    df.iloc[:, 1] = np.array(list(map(item2i.get, df.iloc[:, 1])))

    user_map_file = [f'{org_id} {remap_id}' for remap_id, org_id in enumerate(i2user)]
    user_map_file.insert(0, 'org_id remap_id')
    item_map_file = [f'{org_id} {remap_id}' for remap_id, org_id in enumerate(i2item)]
    item_map_file.insert(0, 'org_id remap_id')
    with open(args.datadir + args.dataset + '_raw_user_map.txt', 'w') as f:
        f.write('\n'.join(user_map_file))
    with open(args.datadir + args.dataset + '_raw_item_map.txt', 'w') as f:
        f.write('\n'.join(item_map_file))


"""按每个 user 相关的 records 分割数据集，保证 val / test set 中的 user 一定出现在 train set 中"""
user_df = df.groupby(by='user')
idxes = [_[1].index for _ in user_df]  # (user_id, records) for every group(user). so the _[1].index
sp_idxes = list(map(split, idxes))  # [[tr_id, va_id, ts_id] for every group]

tr_idx = np.hstack([_[0] for _ in sp_idxes])  # all train records' index
va_idx = np.hstack([_[1] for _ in sp_idxes])
ts_idx = np.hstack([_[2] for _ in sp_idxes])
df_tr = df.iloc[tr_idx, :].reset_index(drop=True)  # split the data
df_va = df.iloc[va_idx, :].reset_index(drop=True)
df_ts = df.iloc[ts_idx, :].reset_index(drop=True)
print('[splitting]\nTr:%.2f%%  Va:%.2f%%  Ts:%.2f%%' %
      (100.0*len(df_tr)/len(df), 100.0*len(df_va)/len(df), 100.0*len(df_ts)/len(df)))
assert len(df_tr) + len(df_ts) + len(df_va) == len(df)


"""将 item 没有在 train set 中出现的 record 移到 train set 中，然后以原格式存储三个子集"""
df_tr, df_va = move_cold_item_to_train(df_tr, df_va)
df_tr, df_ts = move_cold_item_to_train(df_tr, df_ts)
df_tr.to_csv(args.datadir + 'process/' + args.dataset + '_tr.csv', header=None, index=False)
df_va.to_csv(args.datadir + 'process/' + args.dataset + '_va.csv', header=None, index=False)
df_ts.to_csv(args.datadir + 'process/' + args.dataset + '_ts.csv', header=None, index=False)
print('[tuning]\nTr:%.2f%%  Va:%.2f%%  Ts:%.2f%%' %
      (100.0*len(df_tr)/len(df), 100.0*len(df_va)/len(df), 100.0*len(df_ts)/len(df)))
assert len(df_tr) + len(df_ts) + len(df_va) == len(df)


"""按 user 存储三个子集，每一行由 user_id 及其相关的 record_ids 组成"""
tr_out = sort_by_user(df_tr)
va_out = sort_by_user(df_va)
ts_out = sort_by_user(df_ts)

with open(args.datadir + 'process/' + args.dataset + '_sort_tr.txt', 'w') as f:
    f.write('\n'.join(tr_out))
with open(args.datadir + 'process/' + args.dataset + '_sort_va.txt', 'w') as f:
    f.write('\n'.join(va_out))
with open(args.datadir + 'process/' + args.dataset + '_sort_ts.txt', 'w') as f:
    f.write('\n'.join(ts_out))

