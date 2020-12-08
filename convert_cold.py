import time
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from pprint import pprint

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


def get_neighbors(input_df):
    """
    Get users' neighboring items.
    return:
        d - {user: [items]}
    """
    group = input_df.groupby('user')
    d = {g[0]: g[1].item.values for g in group}

    return d


def ndcg_sampling(uid, test_dict, neg_num, item_array):
    """
    Generate ndcg samples for neighboring an item
    param:
        uid - a user in testing set

    return:
        ndcg_array - (n_neighbors, ndcg_k + 1, 3)
    """

    # pos sampling
    pos_neigh = test_dict.get(uid, [])
    pos_set = pos_nb.get(uid, [])
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


"""read data from file"""
df_val = pd.read_csv(args.data_dir + args.dataset + f'/cold_{args.cold_object}_val.csv', dtype=np.int)
df_test = pd.read_csv(args.data_dir + args.dataset + f'/cold_{args.cold_object}_test.csv', dtype=np.int)
df = pd.concat([df_val, df_test])


"""Generate users' neighboring items."""
val_nb = get_neighbors(df_val)
test_nb = get_neighbors(df_test)
pos_nb = get_neighbors(df)

val_item_array = np.unique(df_val['item'])
test_item_array = np.unique(df_test['item'])


"""Get ndcg testing samples"""

# metric@n needs n neg samples
t1 = time.time()
val_at_100 = np.vstack(list(map(lambda x: ndcg_sampling(x, val_nb, 100, val_item_array), list(val_nb.keys()))))
print('Build leave-one-out val@100 for %s dataset in %.2f s' % (args.dataset, time.time() - t1))

t1 = time.time()
test_at_100 = np.vstack(list(map(lambda x: ndcg_sampling(x, test_nb, 100, test_item_array), list(test_nb.keys()))))
print('Build leave-one-out test@100 for %s dataset in %.2f s' % (args.dataset, time.time() - t1))


"""Save results"""
para_dict = {}
para_dict['val@100'] = val_at_100  # (users * pos_items, ndcg_k + 1, 3)
para_dict['test@100'] = test_at_100
pickle.dump(para_dict, open(args.data_dir + args.dataset + f'/cold_{args.cold_object}_dict.pkl', 'wb'), protocol=4)
print('Process %s in %.2f s' % (args.dataset, time.time() - t0))
