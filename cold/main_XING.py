import utils
import numpy as np
import pandas as pd
import tensorflow as tf
import data
import model
from pprint import pprint
import argparse
import scipy.sparse as sp
from utils import *
import pickle

seed = 0
set_seed(seed)

parser = argparse.ArgumentParser(description="main_XING")

parser.add_argument('--data', type=str, default='XING', help='path to eval in the downloaded folder')
parser.add_argument('--datadir', type=str, default='../data/process/')
parser.add_argument('--warm_model', type=str, default='lgn', choices=['grmf', 'bprmf', 'meta2vec', 'lgn'])
parser.add_argument('--model-select', nargs='+', type=int, default=[200],
                    help='specify the fully-connected architecture, starting from input,'
                         ' numbers indicate numbers of hidden units')
parser.add_argument('--rank', type=int, default=200, help='output rank of latent model')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--eval-every', type=int, default=1, help='evaluate every X user-batch')
parser.add_argument('--eval_batch_size', type=int, default=5000)
parser.add_argument('--neg', type=float, default=5, help='negative sampling rate')
parser.add_argument('--lr', type=float, default=0.005, help='starting learning rate')
parser.add_argument('--alpha', type=float, default=0.1, help='diff loss parameter')
parser.add_argument('--reg', type=float, default=0.0001, help='regularization')
parser.add_argument('--dim', type=int, default=5, help='number of experts')
parser.add_argument('--gpu_id', type=int, default=0, help='0 for NAIS_prod, 1 for NAIS_concat')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--type', type=int, default=0, help='type of cold start - 0:user-item, 1:item, 2:user')

args = parser.parse_args()
pprint(vars(args))


def load_data(data_name):
    timer = utils.timer(name='main').tic()
    data_path = args.datadir + data_name
    train_file = data_path + '/warm_emb.csv'
    warm_test_file = data_path + '/warm_test.csv'
    val_file = [data_path + '/cold_both_val.csv',
                data_path + '/cold_item_val.csv',
                data_path + '/cold_user_val.csv']
    test_file = [data_path + '/cold_both_test.csv',
                 data_path + '/cold_item_test.csv',
                 data_path + '/cold_user_test.csv']
    warm_dict_file = data_path + '/warm_dict.pkl'
    cold_dict_file = [data_path + '/cold_both_dict.pkl',
                      data_path + '/cold_item_dict.pkl',
                      data_path + '/cold_user_dict.pkl']
    pref_file = data_path + f'/{args.warm_model}.npy'
    item_content_file = data_path + '/item_content.npz'
    user_content_file = data_path + '/user_content.npz'
    warm_dict = pickle.load(open(warm_dict_file, 'rb'))
    cold_dict = [pickle.load(open(file, 'rb')) for file in cold_dict_file]  # [both, item, user]
    cold_type = ['both', 'item', 'user']

    dat_path = data_path + '/dat.pkl'
    dat = {}
    if not os.path.exists(dat_path):
        # load preference data
        pref = np.load(pref_file)
        dat['user_num'] = warm_dict['user_num']
        dat['item_num'] = warm_dict['item_num']
        dat['u_pref'] = pref[:dat['user_num']]
        dat['v_pref'] = pref[dat['user_num']:]
        dat['warm_items'] = np.array(list(warm_dict['emb_nb_reverse'].keys()), dtype=np.int)
        dat['cold_items'] = np.setdiff1d(np.arange(dat['item_num']), dat['warm_items'])
        timer.toc('Load U:%s, V:%s and standardize.' % (str(dat['u_pref'].shape), str(dat['v_pref'].shape))).tic()

        # load split, cold item, warm user
        timer.tic()
        train = pd.read_csv(train_file, dtype=np.int)
        dat['user_list'] = train['user'].values
        dat['item_list'] = train['item'].values
        timer.toc('read train triplets %s' % str(train.shape)).tic()

        dat['val_eval'] = []
        dat['cold_eval'] = []
        for ct in range(3):
            val_eval_cold = pd.read_csv(val_file[ct], dtype=np.int)
            val_recs = val_eval_cold[['user', 'item']].values
            val_users = np.unique(val_recs[:, 0])
            val_items = np.unique(val_recs[:, 1])
            # 如果是 cold user 情况，要求 item 都是 warm 的，在 eval 时会将 cold items 相关的 scores 给 block 掉，所以不用管这里的 cold items
            dat['val_eval'].append(
                data.EvalData(val_recs, test_items=val_items, test_users=val_users, cold_items=dat['cold_items'],
                              n_items=dat['item_num'], batch_size=args.eval_batch_size))
            timer.toc('read %s val triplets %s' % (cold_type[ct], str(val_eval_cold.shape))).tic()

            cold_eval = pd.read_csv(test_file[ct], dtype=np.int)
            cold_test_recs = cold_eval[['user', 'item']].values
            cold_test_users = np.unique(cold_test_recs[:, 0])
            cold_test_items = np.unique(cold_test_recs[:, 1])
            dat['cold_eval'].append(
                data.EvalData(cold_test_recs, test_items=cold_test_items, test_users=cold_test_users, cold_items=dat['cold_items'],
                              n_items=dat['item_num'], batch_size=args.eval_batch_size))
            timer.toc('read %s cold test triplets %s' % (cold_type[ct], str(cold_eval.shape))).tic()

        warm_eval = pd.read_csv(warm_test_file, dtype=np.int)
        warm_test_recs = warm_eval[['user', 'item']].values
        warm_test_users = np.unique(warm_test_recs[:, 0])
        warm_test_items = np.unique(warm_test_recs[:, 1])
        dat['warm_eval'] = data.EvalData(warm_test_recs, test_items=warm_test_items, test_users=warm_test_users,
                                         cold_items=dat['cold_items'], n_items=dat['item_num'], batch_size=args.eval_batch_size)
        timer.toc('read warm test triplets %s' % str(warm_eval.shape)).tic()

        # load user content data
        dat['user_content'] = sp.load_npz(user_content_file)
        timer.toc('loaded user feature sparse matrix: %s' % (str(dat['user_content'].shape))).tic()
        dat['item_content'] = sp.load_npz(item_content_file)
        timer.toc('loaded item feature sparse matrix: %s' % (str(dat['item_content'].shape))).tic()

        # load metric
        pos_nb = warm_dict['pos_nb']
        warm_test_nb = warm_dict['test_nb']
        dat['metric'] = []
        for ct in range(3):
            item_arr = np.arange(dat['item_num']) if ct != 2 else dat['warm_items']
            cold_test_nb = cold_dict[ct]['test_nb']
            # map user
            cold_test_metric = utils.at_k_sampling(list(cold_test_nb.keys()), cold_test_nb, 100, item_arr,
                                                   pos_nb if ct == 1 else None)
            users = list(map(dat['cold_eval'][ct].test_user_ids_map.get, cold_test_metric[:, 0]))
            cold_test_metric[:, 0] = users
            # map user
            warm_test_metric = utils.at_k_sampling(list(warm_test_nb.keys()), warm_test_nb, 100, item_arr, pos_nb)
            users = list(map(dat['warm_eval'].test_user_ids_map.get, warm_test_metric[:, 0]))
            warm_test_metric[:, 0] = users
            metric = {
                'warm_test': warm_test_metric,
                'cold_test': cold_test_metric,
            }
            dat['metric'].append(metric)  # [both, item, user]
            timer.toc('loaded %s metric: 100 samples' % cold_type[ct]).tic()
        dat['pos_nb'] = pos_nb
        pickle.dump(dat, open(dat_path, 'wb'), protocol=4)
    else:
        dat = pickle.load(open(dat_path, 'rb'))

    return dat


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
data_name = args.data
model_select = args.model_select
rank_out = args.rank
data_batch_size = 1024
dropout = args.dropout
num_epoch = args.epochs
neg = args.neg
_lr = args.lr
_decay_lr_every = 3
_lr_decay = 0.8
cold_type = ['both', 'item', 'user']

dat = load_data(data_name)
u_pref = dat['u_pref']
v_pref = dat['v_pref']
user_content = dat['user_content'].todense()
item_content = dat['item_content'].todense()
test_eval = dat['cold_eval']  # [both, item, user]
val_eval = dat['val_eval']  # [both, item, user]
warm_test_eval = dat['warm_eval']
user_list = dat['user_list']
item_list = dat['item_list']
item_warm = dat['warm_items']

timer = utils.timer(name='main').tic()
# build model
heater = model.Heater(latent_rank_in=u_pref.shape[-1],
                      user_content_rank=user_content.shape[-1],
                      item_content_rank=item_content.shape[-1],
                      model_select=model_select, rank_out=rank_out,
                      reg=args.reg, alpha=args.alpha, dim=args.dim)
heater.build_model()
heater.build_predictor()

saver = tf.train.Saver()
save_path = './model_save/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    timer.toc('initialized tf')

    # original result
    # org_warm_test = utils.batch_eval(sess, heater.eval_preds_warm,
    #                                  eval_feed_dict=heater.get_eval_dict,
    #                                  eval_data=warm_test_eval,
    #                                  metric=dat['metric']['warm_test'],
    #                                  warm=True)

    best_epoch = [0] * 3
    patience = [0] * 3
    best_val_auc = [0.] * 3
    for epoch in range(num_epoch):
        user_array, item_array, target_array = utils.negative_sampling(user_list, item_list, neg, item_warm)
        random_idx = np.random.permutation(user_array.shape[0])
        data_batch = [(n, min(n + data_batch_size, len(random_idx))) for n in
                      range(0, len(random_idx), data_batch_size)]
        loss_epoch = 0.
        rec_loss_epoch = 0.
        reg_loss_epoch = 0.
        diff_loss_epoch = 0.
        for (start, stop) in data_batch:

            batch_idx = random_idx[start:stop]
            batch_users = user_array[batch_idx]
            batch_items = item_array[batch_idx]
            batch_targets = target_array[batch_idx]

            # dropout
            n_to_drop = int(np.floor(dropout * len(batch_idx)))  # number of u-i pairs to be dropped
            if dropout != 0:
                zero_user_index = np.random.choice(np.arange(len(batch_idx)), n_to_drop, replace=False)
                zero_item_index = np.random.choice(np.arange(len(batch_idx)), n_to_drop, replace=False)
            else:
                zero_item_index = np.array([])
                zero_user_index = np.array([])
            dropout_item_indicator = np.zeros_like(batch_targets).reshape((-1, 1))
            dropout_item_indicator[zero_item_index] = 1
            dropout_user_indicator = np.zeros_like(batch_targets).reshape((-1, 1))
            dropout_user_indicator[zero_user_index] = 1

            _, _, loss_out, rec_loss_out, reg_loss_out, diff_loss_out = sess.run(
                [heater.preds, heater.optimizer, heater.loss,
                 heater.rec_loss, heater.reg_loss, heater.diff_loss],
                feed_dict={
                    heater.Uin: u_pref[batch_users, :],
                    heater.Vin: v_pref[batch_items, :],
                    heater.Ucontent: user_content[batch_users, :],
                    heater.Vcontent: item_content[batch_items, :],
                    heater.dropout_user_indicator: dropout_user_indicator,
                    heater.dropout_item_indicator: dropout_item_indicator,
                    heater.target: batch_targets,
                    heater.lr_placeholder: _lr,
                    heater.is_training: True
                })
            loss_epoch += loss_out
            rec_loss_epoch += rec_loss_out
            reg_loss_epoch += reg_loss_out
            diff_loss_epoch += diff_loss_out
            if np.isnan(loss_epoch):
                raise Exception('f is nan')

        timer.toc('%d loss=%.4f reg_loss=%.4f diff_loss=%.4f rec_loss=%.4f' % (
            epoch, loss_epoch / len(data_batch), reg_loss_epoch / len(data_batch),
            diff_loss_epoch / len(data_batch), rec_loss_epoch / len(data_batch)
        )).tic()
        if (epoch + 1) % _decay_lr_every == 0:
            _lr = _lr_decay * _lr
            print('decayed lr:' + str(_lr))

        for ct in range(3):
            # seperately early stop
            if patience[ct] > 10:
                continue

            # eval on val
            val_auc = utils.batch_eval(sess, heater.eval_preds_cold,
                                       eval_feed_dict=heater.get_eval_dict,
                                       eval_data=val_eval[ct],
                                       U_pref=u_pref, V_pref=v_pref,
                                       U_content=user_content,
                                       V_content=item_content,
                                       val=True,
                                       ignore_cold_item=ct == 2)

            # checkpoint
            if val_auc > best_val_auc[ct]:
                saver.save(sess, save_path + args.data + args.warm_model + cold_type[ct])
                patience[ct] = 0
                best_val_auc[ct] = val_auc
                best_epoch[ct] = epoch
            # print results at every epoch
            timer.toc('[%d/10] Current %s val auc:%.4f\tbest:%.4f' %
                      (patience[ct], cold_type[ct], val_auc, best_val_auc[ct])).tic()
            patience[ct] += 1

        # early stop
        if sum(patience) >= 33:
            print(f"Early stop at epoch {epoch}")
            break

    print('=' * 30)
    for ct in range(3):
        saver.restore(sess, save_path + args.data + args.warm_model + cold_type[ct])
        best_warm_test = utils.batch_eval(sess, heater.eval_preds_cold,
                                          eval_feed_dict=heater.get_eval_dict,
                                          eval_data=warm_test_eval,
                                          U_pref=u_pref, V_pref=v_pref,
                                          excluded_dict=dat['pos_nb'],
                                          U_content=user_content,
                                          V_content=item_content,
                                          metric=dat['metric'][ct]['warm_test'],
                                          warm=True,
                                          ignore_cold_item=ct == 2)
        best_cold_test = utils.batch_eval(sess, heater.eval_preds_cold,
                                          eval_feed_dict=heater.get_eval_dict,
                                          eval_data=test_eval[ct],
                                          U_pref=u_pref, V_pref=v_pref,
                                          excluded_dict=dat['pos_nb'] if ct == 1 else None,
                                          U_content=user_content,
                                          V_content=item_content,
                                          metric=dat['metric'][ct]['cold_test'],
                                          ignore_cold_item=ct == 2)

        print('\t\t\t\t\t' + '\t '.join([str(i).ljust(6) for i in ['auc', 'hr', 'ndcg']]))  # padding to fixed len
        # print('origin warm test:\t%s' % (' '.join(['%.6f' % i for i in org_warm_test])))
        print('best[%d] warm test:\t%s' % (best_epoch[ct], ' '.join(['%.6f' % i for i in best_warm_test])))
        print('best[%d] cold test:\t%s' % (best_epoch[ct], ' '.join(['%.6f' % i for i in best_cold_test])))
        print('=' * 30)
