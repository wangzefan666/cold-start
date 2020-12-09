import time
import random
import pickle
import argparse
import numpy as np
from itertools import chain
from gensim.models.word2vec import Word2Vec
from utils import set_seed

RAW_ADJ, MAX_DEGREE = 0, 0

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="CiteULike", help='Dataset to use.')
parser.add_argument('--datadir', type=str, default="../data/process/", help='Director of the dataset.')
parser.add_argument('--walk_num', type=int, default=100, help='Walk number for each node.')
parser.add_argument('--walk_length', type=int, default=80, help='The length of each walk.')
parser.add_argument('--max_degree', type=int, default=768, help='Max degree number.')
parser.add_argument('--emb_size', type=int, default=200, help='Dimension of the word embedding.')
parser.add_argument('--n_jobs', type=int, default=8, help='Multiprocessing number.')
args, _ = parser.parse_known_args()

seed = 0
set_seed(seed)

map_dict = pickle.load(open(args.datadir + args.dataset + '/warm_dict.pkl', 'rb'))
adj = map_dict['adj_train']
user_num = map_dict['user_num']
item_num = map_dict['item_num']
total_num = user_num + item_num


def compute_adj_element(l):
    adj_map = -1 + np.zeros((l[1] - l[0], MAX_DEGREE + 1), dtype=np.int)
    sub_adj = RAW_ADJ[l[0]: l[1]]
    for v in range(l[0], l[1]):
        neighbors = np.nonzero(sub_adj[v - l[0], :])[1]
        len_neighbors = len(neighbors)
        if len_neighbors == 0:
            neighbors = np.array([v])
            len_neighbors = 1
        adj_map[v - l[0], -1] = len_neighbors
        if len_neighbors > MAX_DEGREE:
            neighbors = np.random.choice(neighbors, MAX_DEGREE, replace=False)
            adj_map[v - l[0], :MAX_DEGREE] = neighbors
        else:
            adj_map[v - l[0], :len_neighbors] = neighbors
    return adj_map


def compute_adjlist_parallel(adj, max_degree, batch=50):
    global RAW_ADJ, MAX_DEGREE
    RAW_ADJ = adj
    MAX_DEGREE = max_degree
    num_nodes = adj.shape[0]
    index_list = []
    for ind in range(0, num_nodes, batch):
        index_list.append([ind, min(ind + batch, num_nodes)])
    adj_list = map(compute_adj_element, index_list)
    adj_map = np.vstack(list(adj_list))
    return adj_map


def _walker(start):
    # Input the id of the start point: such as 'u111'
    # Use Fast adjacent table
    sent_list = []
    for _ in range(args.walk_num):
        idx = start
        sent = [str(idx)]
        for _ in range(args.walk_length - 1):
            neigh = ADJ_TAB[idx]
            idx = np.random.choice(neigh[:neigh[-1]], 1)[0]
            sent.append(str(idx))
        sent_list.append(sent)
    return sent_list


ADJ_TAB = compute_adjlist_parallel(adj, max_degree=args.max_degree)

# Random walk
t = time.time()
user_list = list(range(user_num))
sent_list = list(map(_walker, user_list))
print('Random walk for %s in %.2f second' % (args.dataset, time.time() - t))

# Word2Vec
t = time.time()
corpus = list(chain(*sent_list))
w2v = Word2Vec(corpus, size=args.emb_size, iter=10, min_count=0, window=5)
print('Embedding of %s(%d/%d) computed in %.2f s' % (args.dataset, len(w2v.wv.vocab), total_num, time.time() - t))

# store embeddings
node_emb = np.zeros((total_num, args.emb_size))
for _ in w2v.wv.vocab:  # vocab: {'id': embed_array}
    node_emb[int(_)] = w2v.wv[_]
np.save(args.datadir + args.dataset + '/meta2vec.npy', node_emb)
