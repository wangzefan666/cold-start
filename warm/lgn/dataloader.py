"""

"""
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time
from parse import args
from parse import para_dict
import pandas as pd


class Loader(Dataset):

    def __init__(self, path):
        # train or test
        super().__init__()
        self.path = path
        print(f'loading [{path}]')
        self.n_users, self.m_items = para_dict['user_num'], para_dict['item_num']

        train_nb = para_dict['emb_nb']
        train_data = pd.read_csv(path + '/warm_emb.csv')
        self.trainUniqueUsers = np.array(list(train_nb.keys()), dtype=np.int)
        self.trainUser = train_data['user'].values
        self.trainItem = train_data['item'].values
        self.trainUniqueItems = np.unique(self.trainItem)
        self.trainDataSize = len(self.trainUser)

        print(f"There are {self.n_users} users, {self.m_items} items.")
        print(f"{args.dataset} Sparsity : {self.trainDataSize / self.n_users / self.m_items}")

        # get sparse graph
        UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                 shape=(self.n_users, self.m_items))
        self.graph = self.getSparseGraph(UserItemNet)

        # pos dict
        self.allPos = train_nb
        print(f"{args.dataset} is ready to go")

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self, UserItemNet):
        print("generating adjacency matrix")
        s = time()
        adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = UserItemNet.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        end = time()
        print(f"costing {end - s}s, saved norm_mat...")
        sp.save_npz(self.path + f'/adj_mat.npz', norm_adj)

        graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        # 如果重复执行可以产生重复项的操作(例如, torch.sparse.FloatTensor.add())
        # 应该偶尔将稀疏张量coalesced一起, 以防止它们变得太大.
        graph = graph.coalesce().to(args.device)
        return graph

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.allPos[user])
        return posItems
