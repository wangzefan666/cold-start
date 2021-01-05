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
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Edge classifier")
parser.add_argument('--data', type=str, default='CiteULike', help='path to eval in the downloaded folder')
parser.add_argument('--datadir', type=str, default='../data/process/')
parser.add_argument('--warm_model', type=str, default='grmf', choices=['grmf', 'bprmf', 'meta2vec', 'lgn'])
parser.add_argument('--neg_rate', type=int, default=5, help='negative sampling rate')
parser.add_argument('--lr', type=float, default=0.001, help='starting learning rate')
parser.add_argument('--n_hop', type=int, default=2)
parser.add_argument('--train_batch', type=int, default=1024)
parser.add_argument('--eval_batch_size', type=int, default=5000)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--cold_object', type=str, default='item')
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--hid_dim', type=int, default=256)
parser.add_argument('--rank_n', type=int, default=25, help='rank_n neighbor of gcr embedding')
parser.add_argument('--type', type=int, default=0)
parser.add_argument('--reuse', type=int, default=0)
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
cold_features = data['cold_features']
warm_embeddings = data['warm_embeddings']
warm_features = data['warm_features']
warm_object = data['warm_object']
# cold item or all item
item_array = np.arange(data['item_num'])
user_array = np.arange(data['user_num'])
# 注意文件名
if not args.reuse:
    gcr_embeddings = np.load(data_path + f'/{args.warm_model}_25_012_u(sepdot)_i(sepdot)_agg.npy')
else:
    gcr_embeddings = np.load(data_path + f'/{args.warm_model}_25_012_u(sepdot)_i(sepdot)_agg_fake.npy')

timer.toc('loaded data').tic()

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

    saver.restore(sess, save_path + args.data + f'_{args.warm_model}_{args.cold_object}_{str(args.n_hop)}_{str(args.type)}')
    warm_test = utils.batch_eval(sess, classifier.preds,
                                 eval_feed_dict=classifier.get_eval_dict,
                                 metric=data['test'],
                                 cold_features=cold_features,
                                 warm_embeddings=warm_embeddings,
                                 warm_features=warm_features,
                                 )
    timer.toc('[Test]').tic()
    print('\t\t\t\t' + '\t '.join([str(i).ljust(6) for i in ['auc', 'acc']]))  # padding to fixed len
    print('Test:\t%s' % (' '.join(['%.6f' % i for i in warm_test])))

    # Generate n_hop E
    # get rank k warm objects' index
    eval_warm_embeddings = warm_embeddings[warm_object, :]
    eval_warm_features = warm_features[warm_object, :]
    arg_rank_list = []

    # item
    if args.cold_object == 'item':
        for item in tqdm(item_array):
            eval_root_features = np.tile(cold_features[item, :], [len(warm_object), 1])
            # return shape: len(warm_object)
            eval_preds = sess.run(classifier.preds, feed_dict=classifier.get_eval_dict(root_feature=eval_root_features,
                                                                                       warm_embedding=eval_warm_embeddings,
                                                                                       warm_feature=eval_warm_features))
            arg_rank = np.argsort(-eval_preds)
            arg_rank_list.append(arg_rank)
        arg_rank_list = np.vstack(arg_rank_list).astype(np.int)
        arg_rank_list = arg_rank_list[:, :args.rank_n]

        # get generated gcr embeddings
        cold_gcr_embeddings_list = []
        for arg_rank in tqdm(arg_rank_list):
            cold_gcr_embedding = np.mean(warm_embeddings[arg_rank], axis=0)
            cold_gcr_embeddings_list.append(cold_gcr_embedding)
        cold_gcr_embeddings_list = np.vstack(cold_gcr_embeddings_list).astype(np.float64)

        # store them into original gcr embeddings replacing all-zero or randomly generated embeddings
        # remember that in gcr embeddings users' and items' are concat in row, so add items' index should added by user_num
        gcr_embeddings[item_array + data['user_num'], args.n_hop, :] = cold_gcr_embeddings_list
        np.save(data_path + f'/{args.warm_model}_25_012_u(sepdot)_i(sepdot)_agg_fake.npy', gcr_embeddings)

    # user
    elif args.cold_object == 'user':
        for user in tqdm(user_array):
            eval_root_features = np.tile(cold_features[user, :], [len(warm_object), 1])
            # return shape: len(warm_object)
            eval_preds = sess.run(classifier.preds, feed_dict=classifier.get_eval_dict(root_feature=eval_root_features,
                                                                                       warm_embedding=eval_warm_embeddings,
                                                                                       warm_feature=eval_warm_features))
            arg_rank = np.argsort(-eval_preds)
            arg_rank_list.append(arg_rank)
        arg_rank_list = np.vstack(arg_rank_list).astype(np.int)
        arg_rank_list = arg_rank_list[:, :args.rank_n]

        # get generated gcr embeddings
        cold_gcr_embeddings_list = []
        for arg_rank in tqdm(arg_rank_list):
            cold_gcr_embedding = np.mean(warm_embeddings[arg_rank], axis=0)
            cold_gcr_embeddings_list.append(cold_gcr_embedding)
        cold_gcr_embeddings_list = np.vstack(cold_gcr_embeddings_list).astype(np.float64)

        # store them into original gcr embeddings replacing all-zero or randomly generated embeddings
        gcr_embeddings[user_array, args.n_hop, :] = cold_gcr_embeddings_list
        np.save(data_path + f'/{args.warm_model}_25_012_u(sepdot)_i(sepdot)_agg_fake.npy',
                gcr_embeddings)


