import pickle
import utils
import numpy as np
import pandas as pd
import tensorflow as tf
import data
import model
import scipy.sparse as sp
import argparse
import scipy.sparse
from pprint import pprint
from utils import *

seed = 0
set_seed(seed)

parser = argparse.ArgumentParser(description="main_LastFM")
parser.add_argument('--data', type=str, default='LastFM', help='path to eval in the downloaded folder')
parser.add_argument('--datadir', type=str, default='../data/process/')
parser.add_argument('--warm_model', type=str, default='grmf', choices=['grmf', 'bprmf', 'meta2vec', 'lgn'])
parser.add_argument('--model-select', nargs='+', type=int, default=[200],
                    help='specify the fully-connected architecture, starting from input,'
                         ' numbers indicate numbers of hidden units')
parser.add_argument('--rank', type=int, default=200, help='output rank of latent model')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--eval_batch_size', type=int, default=5000)
parser.add_argument('--eval-every', type=int, default=1, help='evaluate every X user-batch')
parser.add_argument('--neg', type=float, default=5, help='negative sampling rate')
parser.add_argument('--lr', type=float, default=0.005, help='starting learning rate')
parser.add_argument('--alpha', type=float, default=0.1, help='diff loss parameter')
parser.add_argument('--reg', type=float, default=0.001, help='regularization')
parser.add_argument('--dim', type=int, default=5, help='number of experts')
parser.add_argument('--gpu_id', type=int, default=0, help='0 for NAIS_prod, 1 for NAIS_concat')
parser.add_argument('--epochs', type=int, default=100)

args = parser.parse_args()
pprint(vars(args))


def load_data(data_name):
    timer = utils.timer(name='main').tic()
    data_path = args.datadir + data_name
    train_file = data_path + '/warm_emb.csv'
    warm_test_file = data_path + '/warm_test.csv'
    test_file = data_path + '/cold_user_test.csv'
    val_file = data_path + '/cold_user_val.csv'
    pref_file = data_path + f'/{args.warm_model}.npy'
    content_file = data_path + '/user_content.npz'
    warm_dict_file = data_path + '/warm_dict.pkl'
    cold_dict_file = data_path + f'/cold_user_dict.pkl'
    cold_dict = pickle.load(open(cold_dict_file, 'rb'))
    warm_dict = pickle.load(open(warm_dict_file, 'rb'))
    dat = {}

    # load preference data
    pref = np.load(pref_file)
    dat['user_num'] = warm_dict['user_num']
    dat['item_num'] = warm_dict['item_num']
    dat['u_pref'] = pref[:dat['user_num']]
    dat['v_pref'] = pref[dat['user_num']:]
    timer.toc('Load U:%s, V:%s and standardize.' % (str(dat['u_pref'].shape), str(dat['v_pref'].shape))).tic()

    # load split
    timer.tic()
    train = pd.read_csv(train_file, dtype=np.int32)
    dat['user_list'] = train['user'].values
    dat['item_list'] = train['item'].values
    timer.toc('read train triplets %s' % str(train.shape)).tic()

    val_eval = pd.read_csv(val_file, dtype=np.int)
    recs = val_eval[['user', 'item']].values
    test_users = np.unique(recs[:, 0])
    test_items = np.unique(recs[:, 1])
    dat['val_eval'] = data.EvalData(recs, test_items=test_items, test_users=test_users, cold_items=[],
                                    n_items=dat['item_num'], batch_size=args.eval_batch_size)
    timer.toc('read val triplets %s' % str(val_eval.shape)).tic()

    cold_eval = pd.read_csv(test_file, dtype=np.int)
    recs = cold_eval[['user', 'item']].values
    test_users = np.unique(recs[:, 0])
    test_items = np.unique(recs[:, 1])
    dat['cold_eval'] = data.EvalData(recs, test_items=test_items, test_users=test_users, cold_items=[],
                                     n_items=dat['item_num'], batch_size=args.eval_batch_size)
    timer.toc('read cold test triplets %s' % str(cold_eval.shape)).tic()

    warm_eval = pd.read_csv(warm_test_file, dtype=np.int)
    recs = warm_eval[['user', 'item']].values
    test_users = np.unique(recs[:, 0])
    test_items = np.unique(recs[:, 1])
    dat['warm_eval'] = data.EvalData(recs, test_items=test_items, test_users=test_users, cold_items=[],
                                     n_items=dat['item_num'], batch_size=args.eval_batch_size)
    timer.toc('read warm test triplets %s' % str(warm_eval.shape)).tic()

    # load user content data
    dat['user_content'] = sp.load_npz(content_file).tolil()
    timer.toc('loaded user feature sparse matrix: %s' % (str(dat['user_content'].shape))).tic()

    # load metric
    item_array = np.arange(dat['item_num'])
    cold_test_nb = cold_dict['test_nb']
    warm_test_nb = warm_dict['test_nb']
    pos_nb = warm_dict['pos_nb']
    # map user
    cold_test_metric = utils.at_k_sampling(list(cold_test_nb.keys()), cold_test_nb, 100, item_array)
    users = list(map(dat['cold_eval'].test_user_ids_map.get, cold_test_metric[:, 0]))
    cold_test_metric[:, 0] = users
    # map user
    warm_test_metric = utils.at_k_sampling(list(warm_test_nb.keys()), warm_test_nb, 100, item_array, pos_nb)
    users = list(map(dat['warm_eval'].test_user_ids_map.get, warm_test_metric[:, 0]))
    warm_test_metric[:, 0] = users
    metric = {
        'warm_test': warm_test_metric,
        'cold_test': cold_test_metric,
    }
    dat['pos_nb'] = pos_nb
    dat['metric'] = metric
    timer.toc('loaded metric: @100').tic()

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
_decay_lr_every = 2
_lr_decay = 0.9

dat = load_data(data_name)
u_pref = dat['u_pref']  # all user pre embedding
v_pref = dat['v_pref']  # all item pre embedding
user_content = dat['user_content'].todense()  # all item context matrix
test_eval = dat['cold_eval']  # EvalData
val_eval = dat['val_eval']  # EvalData
warm_test_eval = dat['warm_eval']  # EvalData
user_list = dat['user_list']  # users of train interactions
item_list = dat['item_list']  # items of train interactions
item_warm = np.arange(dat['item_num'])  # train item set

timer = utils.timer(name='main').tic()
# build model
heater = model.Heater(latent_rank_in=u_pref.shape[1],
                      user_content_rank=user_content.shape[1],
                      item_content_rank=0,
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
    #                                  U_pref=u_pref, V_pref=v_pref,
    #                                  excluded_dict=dat['pos_nb'],
    #                                  V_content=user_content,
    #                                  metric=dat['metric']['warm_test'],
    #                                  warm=True)

    best_epoch = 0
    patience = 0
    val_auc, best_val_auc = 0., 0.
    best_warm_test = np.zeros(3)
    best_cold_test = np.zeros(3)
    for epoch in range(num_epoch):
        user_array, item_array, target_array = utils.negative_sampling(user_list, item_list, neg, item_warm)
        random_idx = np.random.permutation(user_array.shape[0])
        data_batch = [(n, min(n + data_batch_size, len(random_idx))) for n in
                      range(0, len(random_idx), data_batch_size)]
        loss_epoch = 0.
        reg_loss_epoch = 0.
        diff_loss_epoch = 0.
        rec_loss_epoch = 0.
        for (start, stop) in data_batch:

            batch_idx = random_idx[start:stop]
            batch_users = user_array[batch_idx]
            batch_items = item_array[batch_idx]
            batch_targets = target_array[batch_idx]

            # content
            user_content_batch = user_content[batch_users, :]
            # dropout
            if dropout != 0:
                n_to_drop = int(np.floor(dropout * len(batch_idx)))  # number of u-i pairs to be dropped
                zero_index = np.random.choice(np.arange(len(batch_idx)), n_to_drop, replace=False)
            else:
                zero_index = np.array([])
            dropout_indicator = np.zeros_like(batch_targets).reshape((-1, 1))
            dropout_indicator[zero_index] = 1

            _, _, loss_out, rec_loss_out, reg_loss_out, diff_loss_out = sess.run(
                [heater.preds, heater.optimizer, heater.loss,
                 heater.rec_loss, heater.reg_loss, heater.diff_loss],
                feed_dict={
                    heater.Uin: u_pref[batch_users, :],
                    heater.Vin: v_pref[batch_items, :],
                    heater.Ucontent: user_content_batch,
                    heater.dropout_user_indicator: dropout_indicator,
                    heater.target: batch_targets,
                    heater.lr_placeholder: _lr,
                    heater.is_training: True
                }
            )
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

        # eval on val
        val_auc = utils.batch_eval(sess, heater.eval_preds_cold,
                                   eval_feed_dict=heater.get_eval_dict,
                                   eval_data=val_eval,
                                   U_pref=u_pref, V_pref=v_pref,
                                   U_content=user_content,
                                   val=True)

        # checkpoint
        if val_auc > best_val_auc:
            saver.save(sess, save_path + args.data + args.warm_model)
            patience = 0
            best_val_auc = val_auc
            best_epoch = epoch
        # print val results at every epoch
        timer.toc('[%d/10] Current val auc:%.4f\tbest:%.4f' % (patience, val_auc, best_val_auc)).tic()

        # early stop
        patience += 1
        if patience > 10:
            print(f"Early stop at epoch {epoch}")
            break

    saver.restore(sess, save_path + args.data + args.warm_model)
    best_warm_test = utils.batch_eval(sess, heater.eval_preds_cold,
                                      eval_feed_dict=heater.get_eval_dict,
                                      eval_data=warm_test_eval,
                                      U_pref=u_pref, V_pref=v_pref,
                                      excluded_dict=dat['pos_nb'],
                                      U_content=user_content,
                                      metric=dat['metric']['warm_test'],
                                      warm=True)
    best_cold_test = utils.batch_eval(sess, heater.eval_preds_cold,
                                      eval_feed_dict=heater.get_eval_dict,
                                      eval_data=test_eval,
                                      U_pref=u_pref, V_pref=v_pref,
                                      U_content=user_content,
                                      metric=dat['metric']['cold_test'],)
    print('\t\t\t\t\t' + '\t '.join([str(i).ljust(6) for i in ['auc', 'hr', 'ndcg']]))  # padding to fixed len
    # print('origin warm test:\t%s' % (' '.join(['%.6f' % i for i in org_warm_test])))
    print('best[%d] warm test:\t%s' % (best_epoch, ' '.join(['%.6f' % i for i in best_warm_test])))
    print('best[%d] cold test:\t%s' % (best_epoch, ' '.join(['%.6f' % i for i in best_cold_test])))



