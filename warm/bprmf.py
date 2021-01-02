import os
import time
import pickle
import argparse
import numpy as np
import ndcg
import torch
import torch.nn as nn
import torch.optim as optim
from pprint import pprint
from utils import set_seed
import utils
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="LastFM", help='Dataset to use.')
parser.add_argument('--datadir', type=str, default="../data/process/", help='Director of the dataset.')
parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
parser.add_argument("--lamda", type=float, default=0.001, help="model regularization rate")
parser.add_argument("--batch_size", type=int, default=4096, help="batch size for training")
parser.add_argument("--epochs", type=int, default=10000, help="training epoches")
parser.add_argument("--patience", type=int, default=10, help="Patience number")
parser.add_argument("--factor_num", type=int, default=200, help="predictive factors numbers in the model")
parser.add_argument('--neg_num', type=int, default=4, help='BPR negative sampling number.')
parser.add_argument("--out", default=True, help="save model or not")
parser.add_argument("--out_epoch", type=int, default=10, help="Output Epoch")
parser.add_argument('--Ks', nargs='?', default='[1,5,10]', help='Output sizes of every layer')
args, _ = parser.parse_known_args()
args.Ks = eval(args.Ks)
pprint(vars(args))

seed = 42
set_seed(seed)
ndcg.init(args)
device = torch.device('cuda:0')

para_dict = pickle.load(open(args.datadir + args.dataset + '/warm_dict.pkl', 'rb'))
tr_items = para_dict['emb_nb']
user_num = para_dict['user_num']
item_num = para_dict['item_num']
train_data = pd.read_csv(args.datadir + args.dataset + '/warm_emb.csv')
train_array = np.unique(train_data['item'])

pos_items = para_dict['pos_nb']
val_items = para_dict['val_nb']
val_data = pd.read_csv(args.datadir + args.dataset + '/warm_val.csv')
val_array = np.unique(val_data['item'])
VAL_LBS = utils.label_neg_samp(list(val_items.keys()),
                               support_dict=val_items, complement_dict=pos_items,
                               item_array=val_array, neg_rate=args.neg_num)


def bpr_generate(num=None):
    if not num:
        num = args.batch_size
    bpr_lbs = utils.bpr_neg_samp(list(tr_items.keys()), n_users=num * args.neg_num,
                                 support_dict=tr_items, item_array=train_array)
    return torch.LongTensor(bpr_lbs)


class BPR(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(BPR, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        self.drop = nn.Dropout(0.7)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, user, item_i, item_j):
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        item_j = self.embed_item(item_j)
        user = self.drop(user)
        item_i = self.drop(item_i)
        item_j = self.drop(item_j)

        prediction_i = (user * item_i).sum(dim=-1)
        prediction_j = (user * item_j).sum(dim=-1)
        return prediction_i, prediction_j

    def predict(self, user, item):
        user = self.embed_user(user)
        item = self.embed_item(item)
        return torch.sigmoid((user * item).sum(dim=-1))


def _predict(net, lbs, batch_size=10240):
    net.eval()
    with torch.no_grad():
        out_list = []
        for begin in range(0, lbs.shape[0], batch_size):
            end = min(begin + batch_size, lbs.shape[0])
            bh_lbs = torch.LongTensor(lbs[begin:end, :]).to(device)
            out = net.predict(bh_lbs[:, 0], bh_lbs[:, 1])
            out_list.append(out)
        out = torch.cat(out_list, dim=0)
        out = out.cpu().data.numpy()
    return out


# CREATE MODEL
model = BPR(user_num, item_num, args.factor_num)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lamda)


# TRAINING
count, best_auc = 0, 0
epoch, patient_count = 0, 0
init_time = time.time()
for epoch in range(args.epochs):
    model.train()
    start_time = time.time()
    loss0, loss1 = 0., 0.
    for i in range(int(user_num / args.batch_size) + 1):
        model.zero_grad()
        batch_lbs = bpr_generate().to(device)

        pred_i, pred_j = model(batch_lbs[:, 0], batch_lbs[:, 1], batch_lbs[:, 2])
        loss0 = - ((pred_i - pred_j).sigmoid() + 1e-9).log().sum()
        loss = loss0
        loss.backward()
        optimizer.step()
        count += 1

    if epoch % args.out_epoch == 0:
        model.eval()
        auc_vle = ndcg.AUC(model, _predict, lbs=VAL_LBS)
        elapsed_time = time.time() - start_time
        print("{} Epos({}/{}),{:.2f}min, Loss0:{:.2f}, AUC:{:.4f}.".format(
            epoch, patient_count, args.patience, (time.time() - init_time) / 60,
            float(loss0.cpu().data.numpy()), auc_vle))

        if auc_vle > best_auc:
            patient_count = 0
            best_auc = auc_vle
            if args.out:
                if not os.path.exists('model_save/'):
                    os.mkdir('model_save/')
                if epoch > 60:
                    torch.save(model.state_dict(), '{}BPRMF{}.pt'.format('model_save/', args.dataset))
        else:
            patient_count += 1
            if patient_count == args.patience:
                break

# Test
model.load_state_dict(torch.load('{}BPRMF{}.pt'.format('model_save/', args.dataset)))
print('Success Loading!')
best_res = ndcg.l1out_test(model, _predict)
print('%s, bprmf, %.2fmin,' % (epoch, (time.time() - init_time) / 60))
print('AUC, HR, NDCG: %.4f, %.4f, %.4f' % (best_res['auc'], best_res['hr'][0], best_res['ndcg'][0]))

# store embedding
model.eval()
embedding = np.zeros((user_num + item_num, args.factor_num))
embedding[:user_num, :] = model.embed_user.weight.cpu().data.numpy()
embedding[user_num:, :] = model.embed_item.weight.cpu().data.numpy()
np.save(args.datadir + args.dataset + '/bprmf.npy', embedding)
print('embeddings of bprmf is stored in ' + args.datadir + args.dataset + '/bprmf.npy')

