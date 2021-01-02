#!/usr/bin/env python
# coding: utf-8

import utils
import os
import ndcg
import time
import model
import uuid
import pickle
import argparse
import torch
import numpy as np
import pandas as pd
import multiprocessing as mul

# In[3]:

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default="ICLR",
                    help='Model name')
parser.add_argument('--model', type=str, default="MLP",
                    help='Model name')
parser.add_argument('--dataset', type=str, default="CiteULike",
                    help='Dataset to use.')
parser.add_argument('--datadir', type=str, default="../data/process/",
                    help='Director of the dataset.')
parser.add_argument('--user_samp', type=str, default='sepdot',
                    help='Sampling method for user.')
parser.add_argument('--item_samp', type=str, default='sepdot',
                    help='Sampling method for item.')
parser.add_argument('--loss', type=str, default='norm',
                    help='Loss function.')
parser.add_argument('--embed_meth', type=str, default='grmf',
                    help='Emebdding method')
parser.add_argument('--samp_size', type=int, default=25,
                    help='Sampling size.')
parser.add_argument('--gun_layer', type=int, default=3,
                    help='GUN_layer num.')
parser.add_argument('--if_raw', action='store_true', default=False,
                    help='Whether use raw adj matrix.')
parser.add_argument('--if_dnn', action='store_true', default=False,
                    help='Whether use raw adj matrix.')
parser.add_argument('--if_norm', action='store_true', default=False,
                    help='Whether normalized the features.')
parser.add_argument('--layers', nargs='?', default='[0,1,2]',
                    help='GUN layers')
parser.add_argument('--n_jobs', type=int, default=12,
                    help='Multiprocessing number.')
parser.add_argument('--neg_num', type=int, default=5,
                    help='BPR negative sampling number.')
parser.add_argument('--smlp_size', type=int, default=256,
                    help='MLP_size of stack machine.')
parser.add_argument('--smlp_ly', type=int, default=3,
                    help='MLP layer num of stack machine.')
parser.add_argument('--if_output', action='store_true', default=True,
                    help='Whether output.')
parser.add_argument('--if_stack', action='store_true', default=False,
                    help='Wether useing stack embeddings not aggregated embeddings.')
parser.add_argument('--batch_size', type=int, default=10240,
                    help='Normal batch size.')
parser.add_argument('--warm_batch_size', type=int, default=256,
                    help='Warming up batch size.')
parser.add_argument('--warm_batch_num', type=int, default=100,
                    help='Batchs of the warming up.')
parser.add_argument('--out_epoch', type=int, default=5,
                    help='Validation per training batch.')
parser.add_argument('--patience', type=int, default=10,
                    help='Early stop patience.')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Whether use CUDA.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Multiprocessing number.')
parser.add_argument('--drop_rate', type=float, default=0.8,
                    help='Drop Rate.')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='Multiprocessing number.')
parser.add_argument('--Ks', nargs='?', default='[1,5,10]',
                    help='Output sizes of every layer')
parser.add_argument('--skip', type=int, default=0,
                    help='SKip epochs.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random Seed.')
args, _ = parser.parse_known_args()
args.layers = eval(args.layers)
print('#' * 70)
if not args.if_stack:
    args.if_raw = True
if args.if_output:
    print('\n'.join([(str(_) + ':' + str(vars(args)[_])) for _ in vars(args).keys()]))
args.cuda = not args.no_cuda and torch.cuda.is_available()
utils.set_seed(args.seed, args.cuda)
args.device = torch.device("cuda:0" if args.cuda else "cpu")
print(args.device)
if args.dataset == 'wechat':
    args.out_epoch = 1
args.loss = 'bpr'

ndcg.init(args)

# In[4]:


para_dict = pickle.load(open(args.datadir + args.dataset + '/warm_dict.pkl', 'rb'))
uuid_code = str(uuid.uuid4())[:4]
root_path = os.getcwd() + '/'
save_path = root_path + 'model_save/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_file = save_path + args.dataset + args.model + uuid_code
TR_ITEMS = para_dict['map_nb']
VA_ITEMS = para_dict['val_nb']
TS_ITEMS = para_dict['test_nb']
POS_ITEMS = para_dict['pos_nb']
USER_NUM = para_dict['user_num']
ITEM_NUM = para_dict['item_num']
USER_ARRAY = np.array(list(range(USER_NUM)))
ITEM_ARRAY = np.array(list(range(ITEM_NUM)))
SAMP_POOL = None

# In[5]:


print(args.datadir + args.dataset + '/' + args.embed_meth + '_' +
      str(args.samp_size) + '_' + ''.join([str(_) for _ in args.layers])
      + '_u(' + args.user_samp + ')_i(' + args.item_samp + ')_agg.npy')
emb = np.load(
    args.datadir + args.dataset + '/' + args.embed_meth + '_' +
    str(args.samp_size) + '_' + ''.join([str(_) for _ in args.layers])
    + '_u(' + args.user_samp + ')_i(' + args.item_samp + ')_agg.npy')


# In[6]:


def _vasamp_bpr_pair(udata):
    uid = udata[0]
    np.random.seed(udata[1])
    ret_array = np.zeros((args.neg_num, 3)).astype(np.int)
    pos_train = VA_ITEMS.get(uid, [])
    if len(pos_train) == 0:
        return np.zeros((0, 3)).astype(np.int)
    pos_set = set(POS_ITEMS.get(uid, []))
    samp_pos = np.random.choice(pos_train, 1).astype(np.int)
    neg_items = np.random.choice(ITEM_ARRAY, 5 * args.neg_num)
    samp_neg = np.array(neg_items[[_ not in pos_set for _ in neg_items]])[:args.neg_num].astype(np.int)
    ret_array[:, 0] = uid
    ret_array[:, 1] = samp_pos + USER_NUM
    ret_array[:, 2] = samp_neg + USER_NUM
    return ret_array


def vabpr_generate(num=None):
    global SAMP_POOL
    if not SAMP_POOL:
        SAMP_POOL = mul.Pool(args.n_jobs)
    if not num:
        num = args.batch_size
    samp_user = np.hstack(
        [np.random.choice(USER_ARRAY, num).reshape([-1, 1]), np.random.randint(0, 2 ** 32, num).reshape([-1, 1])])
    bpr_lbs = np.vstack(SAMP_POOL.map(_vasamp_bpr_pair, samp_user))
    return bpr_lbs


def bpr_loss(tr_out):
    tr_out = tr_out.reshape([-1, 2])
    return torch.mean(-torch.log(torch.sigmoid_(tr_out[:, 0] - tr_out[:, 1]) + 1e-9))


def _predict(net, lbs, batch_size=20480):
    net.eval()
    with torch.no_grad():
        out_list = []
        for begin in range(0, lbs.shape[0], batch_size):
            end = min(begin + batch_size, lbs.shape[0])
            batch_lbs = lbs[begin:end, :].copy()
            batch_lbs[:, 1] += USER_NUM
            out = net(batch_lbs)
            out_list.append(out)
        out = torch.cat(out_list, dim=0).cpu().data.numpy().reshape(-1)
    return out


# In[7]:

t0 = time.time()
X = torch.from_numpy(emb).float().to(args.device)

if args.gun_layer == 2:
    layer_pairs = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
else:
    layer_pairs = [
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [1, 2],
        [2, 0],
        [2, 1],
        [2, 2]
    ]

gun = eval('model.' + args.model)(X, layer_pairs, mlp_size=args.smlp_size, mlp_layer=args.smlp_ly, if_xavier=True,
                                  drop_rate=0.7, device=args.device)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, gun.parameters()), lr=args.lr, weight_decay=args.weight_decay)
if args.if_output:
    print(gun)

# In[8]:


loss_func = bpr_loss
patience_count = 0
va_acc_max = 0
ts_score_max = 0
batch = 0
time_list = []
time_plot_list = []
early_stop_flag = False
train_time = 0
opt_time = 0
val_time = 0
pure_train_time = 0
batch_size = args.warm_batch_size
plot_time_list = []

# In[9]:


for epoch in range(50000):
    if early_stop_flag:
        break
    t_epoch_begin = time.time()
    epoch_train_cost = []
    epoch_val_cost = []
    for ind in range(0, max(int(len(USER_ARRAY) / batch_size), 1)):
        t_train_begin = time.time()
        if batch == args.warm_batch_num:
            batch_size = args.batch_size
            print('*' * 25, 'Updating BatchSize to %d' % batch_size, '*' * 25)
            batch += 1
            break
        batch += 1
        batch_lbs = vabpr_generate(batch_size)
        batch_lbs = np.hstack([batch_lbs[:, [0, 1]], batch_lbs[:, [0, 2]]]).reshape([-1, 2])
        gun.train()
        t_opt_begin = time.time()
        optimizer.zero_grad()
        tr_out = gun(batch_lbs)
        loss = loss_func(tr_out)
        loss.backward()
        optimizer.step()
        t_train_end = time.time()
        epoch_train_cost.append(t_train_end - t_train_begin)
        train_time += t_train_end - t_train_begin
        opt_time += t_train_end - t_opt_begin
    if (epoch % args.out_epoch == 0) or (epoch < args.out_epoch):
        gun.eval()
        t_val_begin = time.time()
        va_acc = ndcg._auc(gun, _predict)
        time_plot_list.append([epoch, train_time, va_acc])
        if epoch > args.skip:
            if va_acc > va_acc_max:
                va_acc_max = va_acc
                torch.save(gun.state_dict(), save_file)
                patience_count = 0
            else:
                patience_count += 1
                if patience_count > args.patience:
                    early_stop_flag = True
                    break
        t_val_end = time.time()
        val_time += t_val_end - t_val_begin
        epoch_val_cost.append(t_val_end - t_val_begin)
        plot_time_list.append([batch, train_time, opt_time, va_acc])
        if args.if_output:
            print(
                'Epo%d(%d/%d) loss:%.4f|VA_auc:%.4f|BestVA_auc:%.4f|Train:%.2f,Opt:%.2f,Val:%.2fs' % (
                    epoch + 1, patience_count, args.patience, loss.data, va_acc,
                    va_acc_max, train_time, opt_time, val_time))
    #                 print(gun.mlp.weight.data)
    #             print(gun.w1.weight, gun.w2.weight)
    t_epoch_end = time.time()
    time_list.append([
        t_epoch_end - t_epoch_begin,
        np.sum(epoch_train_cost),
        np.sum(epoch_val_cost),
        train_time
    ])

# In[10]:


time_array = np.array(time_list)
t1 = time.time()
gun.load_state_dict(torch.load(save_file))
gun.eval()
running_cost = t1 - t0
result = [args.dataset + ',' + args.embed_meth, args.user_samp + args.item_samp, epoch]
if args.if_dnn:
    result = [args.dataset + '-' + args.embed_meth, 'dnn', epoch]
result += [
    running_cost,  # Total Running time 3
    train_time,  # 4
    np.mean(time_array[:, 0]),  # Per epoch running time 5
    np.mean(time_array[:, 1]),  # Per epoch training time 6
    np.mean(time_array[:, 2]),  # Per epoch val time 7
]

# In[11]:

res = ndcg._fast_ndcg(gun, _predict)

print('#' * 30, args.dataset, args.model, args.user_samp + args.item_samp, '#' * 30)
print('Final Epoch:%d, Running time:%.2f, Train:%.2f, Opt:%.2f' % (
    result[2], result[3], result[4], opt_time))
print('Per Epoch Run:%.2f, Per Epoch Train:%.2f, Per Epoch Test:%.2f' % (
    result[5], result[6], result[7]))
print('Ts_auc  HR@%d  NDCG@%d: %.4f  :%.4f  :%.4f' % (
    eval(args.Ks)[-1], eval(args.Ks)[-1], res['auc'], res['hr'][-1], res['ndcg'][-1]))

# In[12]:


SAMP_POOL.close()
SAMP_POOL = None
print('Pool shutted')

# In[13]:


plot_file_name = './Time/%s_%s_%d_%d_%s_%s_%s_%s.csv' % (
    args.dataset, args.model, args.smlp_ly, args.seed, args.dataset,
    args.embed_meth, ''.join([str(_) for _ in args.layers]),
    args.user_samp + args.item_samp)
df = pd.DataFrame(plot_time_list, columns=['epochs', 'train_time', 'opt_time', 'tsauc'])
df.to_csv(plot_file_name, index=False)

# In[14]:


weight_flag = False
if (args.model == 'ModStack' or args.model == 'EleStack') and args.smlp_ly == 1:
    weight_flag = True
    wei = list(gun.mlp[0].weight.cpu().data.reshape([-1]).numpy())
if args.model == 'WeiSum' and args.smlp_ly == 1:
    weight_flag = True
    wei = list(gun.w1.weight.cpu().data.reshape([-1]).numpy())
    wei += list(gun.w2.weight.cpu().data.reshape([-1]).numpy())
if args.model == 'WeiShareSum' and args.smlp_ly == 1:
    weight_flag = True
    wei = list(gun.w.weight.cpu().data.reshape([-1]).numpy())
if weight_flag:
    with open('../weight/%s%s.txt' % (args.dataset, args.model), 'a') as f:
        f.write('%s,%s,%s,%d,' % (args.dataset, args.embed_meth, args.user_samp + args.item_samp, args.seed))
        f.write(','.join([str(_) for _ in wei]))
        f.write('\n')

# In[15]:


with open('./result/%s-%s-%d.txt' % (args.dataset, args.model, args.smlp_ly), 'a') as f:
    f.write('%s,%s,%d,%d,%s,' % (result[0], result[1], args.gun_layer, args.samp_size, args.loss))
    f.write('%d,%f,%f,%d,%d\n' % (args.skip, args.drop_rate, args.lr, args.smlp_size, args.smlp_ly))
    f.write('%d,%.2f,%.2f,' % (result[2], result[3], result[4]))
    f.write('%.2f,%.2f,%.2f\n' % (result[5], result[6], result[7]))
    f.write('%.4f,' % (res['auc']))
    f.write(''.join(['%.4f,' % _ for _ in res['hr']]) + ''.join(['%.4f,' % _ for _ in res['ndcg']]))
    f.write('\n')
