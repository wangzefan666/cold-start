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


def main():
    data_name = args.data
    model_select = args.model_select
    rank_out = args.rank
    data_batch_size = 1024
    dropout = args.dropout
    eval_batch_size = 5000  # the batch size when test
    num_epoch = 100
    neg = args.neg
    _lr = args.lr
    _decay_lr_every = 3
    _lr_decay = 0.8

    dat = load_data(data_name)
    u_pref = dat['u_pref']
    v_pref = dat['v_pref']
    user_content = dat['user_content']
    item_content = dat['item_content']
    test_eval = dat['test_eval']
    val_eval = dat['val_eval']
    warm_test_eval = dat['warm_test']
    user_list = dat['user_list']
    item_list = dat['item_list']
    item_warm = np.unique(item_list)
    timer = utils.timer(name='main').tic()

    # prep eval
    timer.tic()
    cold_user = True if args.type != 1 else False
    cold_item = True if args.type != 2 else False
    test_eval.init_tf(u_pref, v_pref, user_content, item_content, eval_batch_size,
                      cold_user=cold_user, cold_item=cold_item)
    val_eval.init_tf(u_pref, v_pref, user_content, item_content, eval_batch_size,
                     cold_user=cold_user, cold_item=cold_item)
    warm_test_eval.init_tf(u_pref, v_pref, user_content, item_content, eval_batch_size,
                           cold_user=cold_user, cold_item=cold_item)
    timer.toc('initialized eval data').tic()

    heater = model.Heater(latent_rank_in=u_pref.shape[1],
                          user_content_rank=user_content.shape[1] if cold_user else 0,
                          item_content_rank=item_content.shape[1] if cold_item else 0,
                          model_select=model_select,
                          rank_out=rank_out, reg=args.reg, alpha=args.alpha, dim=args.dim)
    heater.build_model()
    heater.build_predictor()

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        timer.toc('initialized tf')

        # original result
        org_warm_test = utils.batch_eval(sess, heater.eval_preds_warm,
                                         eval_feed_dict=heater.get_eval_dict,
                                         eval_data=warm_test_eval,
                                         metric=dat['metric']['warm_test'],
                                         warm=True)

        best_epoch = 0
        patience = 0
        val_auc, best_val_auc = 0., 0.
        best_warm_test = np.zeros(3)
        best_cold_test = np.zeros(3)
        for epoch in range(num_epoch):
            user_array, item_array, target_array = utils.negative_sampling(user_list, item_list, neg, item_warm)
            random_idx = np.random.permutation(user_array.shape[0])
            n_targets = len(random_idx)
            data_batch = [(n, min(n + data_batch_size, n_targets)) for n in
                          range(0, n_targets, data_batch_size)]
            loss_epoch = 0.
            rec_loss_epoch = 0.
            reg_loss_epoch = 0.
            diff_loss_epoch = 0.
            for (start, stop) in data_batch:

                batch_idx = random_idx[start:stop]
                batch_users = user_array[batch_idx]
                batch_items = item_array[batch_idx]
                batch_targets = target_array[batch_idx]

                # pref
                u_pref_batch = u_pref[batch_users, :]
                v_pref_batch = v_pref[batch_items, :]
                # content
                item_content_batch = item_content[batch_items, :].todense() if cold_item else None
                user_content_batch = user_content[batch_users, :].todense() if cold_user else None
                # dropout
                dropout_item_indicator, dropout_user_indicator = None, None
                n_to_drop = int(np.floor(dropout * len(batch_idx)))  # number of u-i pairs to be dropped
                if cold_item:
                    if dropout != 0:
                        zero_item_index = np.random.choice(np.arange(len(batch_idx)), n_to_drop, replace=False)
                    else:
                        zero_item_index = np.array([])
                    dropout_item_indicator = np.zeros_like(batch_targets).reshape((-1, 1))
                    dropout_item_indicator[zero_item_index] = 1
                if cold_user:
                    if dropout != 0:
                        zero_user_index = np.random.choice(np.arange(len(batch_idx)), n_to_drop, replace=False)
                    else:
                        zero_user_index = np.array([])
                    dropout_user_indicator = np.zeros_like(batch_targets).reshape((-1, 1))
                    dropout_user_indicator[zero_user_index] = 1

                _, _, loss_out, rec_loss_out, reg_loss_out, diff_loss_out = sess.run(
                    [heater.preds, heater.optimizer, heater.loss,
                     heater.rec_loss, heater.reg_loss, heater.diff_loss],
                    feed_dict=[{
                        heater.Uin: u_pref_batch,
                        heater.Vin: v_pref_batch,
                        heater.Ucontent: user_content_batch,
                        heater.Vcontent: item_content_batch,
                        heater.dropout_user_indicator: dropout_user_indicator,
                        heater.dropout_item_indicator: dropout_item_indicator,
                        heater.target: batch_targets,
                        heater.lr_placeholder: _lr,
                        heater.is_training: True
                    }, {
                        heater.Uin: u_pref_batch,
                        heater.Vin: v_pref_batch,
                        heater.Vcontent: item_content_batch,
                        heater.dropout_item_indicator: dropout_item_indicator,
                        heater.target: batch_targets,
                        heater.lr_placeholder: _lr,
                        heater.is_training: True
                    }, {
                        heater.Uin: u_pref_batch,
                        heater.Vin: v_pref_batch,
                        heater.Ucontent: user_content_batch,
                        heater.dropout_user_indicator: dropout_user_indicator,
                        heater.target: batch_targets,
                        heater.lr_placeholder: _lr,
                        heater.is_training: True
                    }][args.type]
                )
                loss_epoch += loss_out
                rec_loss_epoch += rec_loss_out
                reg_loss_epoch += reg_loss_out
                diff_loss_epoch += diff_loss_out
                if np.isnan(loss_epoch):
                    raise Exception('f is nan')

            if (epoch + 1) % _decay_lr_every == 0:
                _lr = _lr_decay * _lr
                print('decayed lr:' + str(_lr))

            val_auc = utils.batch_eval_auc(sess, heater.eval_preds_cold,
                                           eval_feed_dict=heater.get_eval_dict,
                                           eval_data=val_eval)

            # checkpoint
            if val_auc > best_val_auc:
                patience = 0
                best_val_auc = val_auc
                best_warm_test = utils.batch_eval(sess, heater.eval_preds_cold,
                                                  eval_feed_dict=heater.get_eval_dict,
                                                  eval_data=warm_test_eval,
                                                  metric=dat['metric']['warm_test'],
                                                  warm=True)
                best_cold_test = utils.batch_eval(sess, heater.eval_preds_cold,
                                                  eval_feed_dict=heater.get_eval_dict,
                                                  eval_data=test_eval,
                                                  metric=dat['metric']['cold_test'])
                best_epoch = epoch

            # print results at every epoch
            timer.toc('%d loss=%.4f reg_loss=%.4f diff_loss=%.4f rec_loss=%.4f' % (
                epoch, loss_epoch / len(data_batch), reg_loss_epoch / len(data_batch),
                diff_loss_epoch / len(data_batch), rec_loss_epoch / len(data_batch)
            )).tic()
            print('Current val auc:%.4f\tbest:%.4f' % (val_auc, best_val_auc))
            print('\t\t\t\t\t' + '\t '.join([str(i).ljust(6) for i in ['auc', 'hr', 'ndcg']]))  # padding to fixed len
            print('origin warm test:\t%s' % (' '.join(['%.6f' % i for i in org_warm_test])))
            print('best[%d] warm test:\t%s' % (best_epoch, ' '.join(['%.6f' % i for i in best_warm_test])))
            print('best[%d] cold test:\t%s' % (best_epoch, ' '.join(['%.6f' % i for i in best_cold_test])))

            # early stop
            patience += 1
            if patience > 10:
                print(f"Early stop at epoch {epoch}")
                break


def load_data(data_name):
    timer = utils.timer(name='main').tic()
    data_path = args.datadir + data_name
    train_file = data_path + '/warm_emb.csv'
    warm_test_file = data_path + '/warm_test.csv'
    val_file = [data_path + '/cold_both_val.csv', data_path + '/cold_item_val.csv', data_path + '/cold_user_val.csv']
    test_file = [data_path + '/cold_both_test.csv', data_path + '/cold_item_test.csv',
                 data_path + '/cold_user_test.csv']
    pref_file = data_path + f'/{args.warm_model}.npy'
    item_content_file = data_path + '/item_content.npz'
    user_content_file = data_path + '/user_content.npz'
    warm_dict_file = data_path + '/warm_dict.pkl'
    cold_dict_file = [data_path + '/cold_both_dict.pkl', data_path + '/cold_item_dict.pkl', data_path + '/cold_user_dict.pkl']
    dat = {}

    # load split
    timer.tic()
    train = pd.read_csv(train_file, dtype=np.int32)
    dat['user_list'] = train['user'].values
    dat['item_list'] = train['item'].values
    dat['warm_user'] = np.unique(train['user'])
    dat['warm_item'] = np.unique(train['item'])
    dat['test_eval'] = data.load_eval_data(test_file[args.type])
    dat['val_eval'] = data.load_eval_data(val_file[args.type])
    dat['warm_test'] = data.load_eval_data(warm_test_file)
    timer.toc('read train triplets %s' % str(train.shape)).tic()

    # load preference data
    pref = np.load(pref_file)
    n_warm_user = len(np.unique(dat['user_list']))
    max_user_id = np.max([np.max(dat['user_list']),
                          np.max(dat['test_eval'].test_user_ids),
                          np.max(dat['val_eval'].test_user_ids)])
    max_item_id = np.max([np.max(dat['item_list']),
                          np.max(dat['test_eval'].test_item_ids),
                          np.max(dat['val_eval'].test_item_ids)])
    # mapped object
    mapped_user = pref[:n_warm_user]
    mapped_item = pref[n_warm_user:]
    # reversely mapped user
    user_map = pd.read_csv(data_path + '/warm_user_mapped.csv', dtype=np.int)
    new2old = user_map['org_id'].values
    dat['u_pref'] = np.zeros((max_user_id + 1, pref.shape[1]))
    dat['u_pref'][new2old] = mapped_user[:]
    # reversely mapped item
    item_map = pd.read_csv(data_path + '/warm_item_mapped.csv', dtype=np.int)
    new2old = item_map['org_id'].values
    dat['v_pref'] = np.zeros((max_item_id + 1, pref.shape[1]))
    dat['v_pref'][new2old] = mapped_item[:]
    # standardize
    _, dat['u_pref'] = utils.standardize(dat['u_pref'])
    _, dat['v_pref'] = utils.standardize_3(dat['v_pref'])
    timer.toc('Load U:%s, V:%s and standardize.' % (str(dat['u_pref'].shape), str(dat['v_pref'].shape))).tic()

    # load content data
    dat['user_content'] = sp.load_npz(user_content_file).tolil()
    dat['item_content'] = sp.load_npz(item_content_file).tolil()
    timer.toc('loaded item feature sparse matrix: %s' % (str(dat['item_content'].shape)))
    timer.toc('loaded user feature sparse matrix: %s' % (str(dat['user_content'].shape))).tic()

    # load metric
    cold_dict = pickle.load(open(cold_dict_file[args.type], 'rb'))
    warm_dict = pickle.load(open(warm_dict_file, 'rb'))
    metric = {
        'val': cold_dict['val@100'],
        'warm_test': warm_dict['test@100'],
        'cold_test': cold_dict['test@100'],
    }
    dat['metric'] = metric

    return dat


if __name__ == "__main__":
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
    parser.add_argument('--neg', type=float, default=5, help='negative sampling rate')
    parser.add_argument('--lr', type=float, default=0.005, help='starting learning rate')
    parser.add_argument('--alpha', type=float, default=0.1, help='diff loss parameter')
    parser.add_argument('--reg', type=float, default=0.0001, help='regularization')
    parser.add_argument('--dim', type=int, default=5, help='number of experts')
    parser.add_argument('--type', type=int, default=0, help='type of cold start - 0:user-item, 1:item, 2:user')

    args = parser.parse_args()
    pprint(vars(args))

    main()
