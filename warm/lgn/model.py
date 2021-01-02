import torch
from torch import nn
import torch.nn.functional as F
import copy
from parse import args


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, graph, users):
        raise NotImplementedError


class LightGCN(BasicModel):
    def __init__(self, n_users, m_items):
        super(LightGCN, self).__init__()
        self.num_users = n_users
        self.num_items = m_items
        self.latent_dim = args.embed_dim
        self.n_layers = args.n_layers
        self.keep_prob = 1 - args.drop_rate
        self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.__init_weight()

    def __init_weight(self):
        """
        random normal init seems to be a better choice
        when lightGCN actually don't use any non-linear activation function
        """
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        # new characteristic of python 3.6 - f""
        print(f"model lgn(dropout:{args.dropout})")

    def __dropout(self, graph, keep_prob):
        """
        use dropout to make a more sparse graph
        """
        size = graph.size()
        index = graph.indices().t()
        values = graph.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def computer(self, graph):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb], dim=0)
        if args.dropout:
            if self.training:
                print("droping")
                g_droped = self.__dropout(graph, self.keep_prob)
            else:
                g_droped = graph
        else:
            g_droped = graph

        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=2)  # (n_nodes, embed_size, n_layers)
        embs = torch.mean(embs, dim=2)

        users, items = torch.split(embs, [self.num_users, self.num_items], dim=0)
        return users, items

    def getUsersRating(self, graph, users):
        all_users, all_items = self.computer(graph)
        users = all_users[users]  # (n_nodes, embed_size)
        # why sigmoid here? for auc
        rating = torch.sigmoid(torch.matmul(users, all_items.t()))
        return rating  # (n_users, n_items)

    def getEmbedding(self, graph, users, pos_items, neg_items):
        all_users, all_items = self.computer(graph)
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, graph, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(graph, users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        # why softplus here?
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        return loss, reg_loss

    def forward(self, users, items):
        return

