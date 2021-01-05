import os
import pickle
import utils
import numpy as np
import pandas as pd
import tensorflow as tf
import data
import model
import scipy.sparse as sp
import argparse
from pprint import pprint

parser = argparse.ArgumentParser(description="Edge classifier")
parser.add_argument('--data', type=str, default='CiteULike', help='path to eval in the downloaded folder')
parser.add_argument('--datadir', type=str, default='../data/process/')
parser.add_argument('--warm_model', type=str, default='grmf', choices=['grmf', 'bprmf', 'meta2vec', 'lgn'])
parser.add_argument('--neg_rate', type=int, default=5, help='negative sampling rate')
parser.add_argument('--lr', type=float, default=0.001, help='starting learning rate')
parser.add_argument('--n_hop', type=int, default=1)
parser.add_argument('--train_batch', type=int, default=1024)
parser.add_argument('--eval_batch_size', type=int, default=50000)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--cold_object', type=str, default='item')
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--hid_dim', type=int, default=256)
parser.add_argument('--type', type=int, default=0, help="0:only embedding  1:only feature  2:both")
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--gpu_id', type=int, default=3)

args = parser.parse_args()
pprint(vars(args))

timer = utils.timer(name='main').tic()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
data_path = args.datadir + args.data
# *******************************
# 使用 warm feature 后结果明显下降
# has_warm_feature = False
# *******************************
data = data.load_data(args)
warm_embeddings = data['warm_embeddings']
cold_features = data['cold_features']
warm_features = data['warm_features']
timer.toc('loaded data')

# build model
classifier = model.Edge_classifier(cold_feature_dim=cold_features.shape[-1],
                                   warm_feature_dim=warm_features.shape[-1],
                                   embed_dim=warm_embeddings.shape[-1],
                                   type=args.type, lr=args.lr,
                                   n_layers=args.n_layers, hid_dim=args.hid_dim,
                                   dropout=args.dropout)
classifier.build_model()

saver = tf.train.Saver()
save_path = './model_save/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    timer.toc('initialized tf').tic()

    best_epoch = 0
    patience = 0
    val_auc, best_val_auc = 0., 0.
    train_array = data['train']
    for epoch in range(args.epochs):
        random_idx = np.random.permutation(train_array.shape[0])  # 生成一个打乱的 range 序列作为下标
        data_batch = [(n, min(n + args.train_batch, len(random_idx))) for n in
                      range(0, len(random_idx), args.train_batch)]
        loss_epoch = 0.
        reg_loss_epoch = 0.
        pred_loss_epoch = 0.

        for (start, stop) in data_batch:
            batch_idx = random_idx[start:stop]
            batch_root_index = train_array[batch_idx, 0]
            batch_nei_index = train_array[batch_idx, 1]
            batch_targets = train_array[batch_idx, 2]

            # run
            batch_root_features = cold_features[batch_root_index, :]
            batch_warm_embeddings = warm_embeddings[batch_nei_index, :]
            batch_warm_features = warm_features[batch_nei_index, :]

            _, _, loss, pred_loss, reg_loss = sess.run(
                [classifier.preds, classifier.optimizer, classifier.loss, classifier.pred_loss, classifier.reg_loss],
                feed_dict=classifier.get_train_dict(
                    root_feature=batch_root_features,
                    warm_embedding=batch_warm_embeddings,
                    target=batch_targets,
                    warm_feature=batch_warm_features
                )
            )

            loss_epoch += loss
            pred_loss_epoch += pred_loss
            reg_loss_epoch += reg_loss
            if np.isnan(loss_epoch):
                raise Exception('f is nan')
        timer.toc('%d loss=%.4f reg_loss=%.4f pred_loss=%.4f' % (
            epoch, loss_epoch / len(data_batch), reg_loss_epoch / len(data_batch), pred_loss_epoch / len(data_batch)
        )).tic()

        # eval on val
        val_auc, _ = utils.batch_eval(sess, classifier.preds,
                                      eval_feed_dict=classifier.get_eval_dict,
                                      metric=data['val'],
                                      cold_features=cold_features,
                                      warm_embeddings=warm_embeddings,
                                      warm_features=warm_features,
                                      batch_size=args.eval_batch_size,
                                      )
        # if get a better eval result on val, update test result
        # best_recall and best_test_recall are global variables while others are local ones
        if val_auc > best_val_auc:
            saver.save(sess, save_path + args.data + f'_{args.warm_model}_{args.cold_object}_{str(args.n_hop)}_{str(args.type)}')
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

    # Test
    saver.restore(sess, save_path + args.data + f'_{args.warm_model}_{args.cold_object}_{str(args.n_hop)}_{str(args.type)}')
    warm_test = utils.batch_eval(sess, classifier.preds,
                                 eval_feed_dict=classifier.get_eval_dict,
                                 metric=data['test'],
                                 cold_features=cold_features,
                                 warm_embeddings=warm_embeddings,
                                 warm_features=warm_features,
                                 batch_size=args.eval_batch_size,
                                 )
    timer.toc('[Test]').tic()
    print('\t\t\t\t\t' + '\t '.join([str(i).ljust(6) for i in ['auc', 'acc']]))  # padding to fixed len
    print('best[%d] test:\t%s' % (best_epoch, ' '.join(['%.6f' % i for i in warm_test])))
