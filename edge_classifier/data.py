import utils
import copy
import numpy as np
import scipy.sparse as sp
import pandas as pd
import pickle
import os


def load_data(args):
    data_path = args.datadir + args.data
    warm_dict_file = data_path + '/warm_dict.pkl'
    cold_feature_file = data_path + f'/{args.cold_object}_content.npz'
    warm_embedding_file = data_path + f'/{args.warm_model}.npy'
    warm_dict = pickle.load(open(warm_dict_file, 'rb'))
    user_num = warm_dict['user_num']
    item_num = warm_dict['item_num']
    item_array = np.arange(item_num)
    user_array = np.arange(user_num)
    warm_user = np.array(list(warm_dict['emb_nb'].keys())).astype(np.int)
    warm_item = np.array(list(warm_dict['emb_nb_reverse'].keys())).astype(np.int)

    data = {}
    data['user_num'] = user_num
    data['item_num'] = item_num
    data['cold_features'] = sp.load_npz(cold_feature_file).todense().A

    embeddings = np.load(warm_embedding_file)

    if args.cold_object == 'item':
        if args.n_hop == 1:
            uni_items = list(warm_dict['emb_nb_reverse'].keys())
            train_array = utils.label_neg_samp(uni_items, warm_dict['emb_nb_reverse'], user_array,
                                               args.neg_rate, pos_dict=warm_dict['pos_nb_reverse'])
            uni_items = list(warm_dict['val_nb_reverse'].keys())
            val_array = utils.label_neg_samp(uni_items, warm_dict['val_nb_reverse'], user_array,
                                             5, pos_dict=warm_dict['pos_nb_reverse'])
            uni_items = list(warm_dict['test_nb_reverse'].keys())
            test_array = utils.label_neg_samp(uni_items, warm_dict['test_nb_reverse'], user_array,
                                              5, pos_dict=warm_dict['pos_nb_reverse'])
            data['warm_embeddings'] = embeddings[:user_num]
            data['warm_object'] = np.array(list(warm_dict['emb_nb'].keys()), dtype=np.int)
            warm_object = 'user'
        elif args.n_hop == 2:
            uni_item_1 = list(warm_dict['ii_train_nb'].keys())
            train_array = utils.label_neg_samp(uni_item_1, warm_dict['ii_train_nb'], warm_item,
                                               args.neg_rate, pos_dict=warm_dict['ii_pos_nb'])
            uni_item_1 = list(warm_dict['ii_val_nb'].keys())
            val_array = utils.label_neg_samp(uni_item_1, warm_dict['ii_val_nb'], warm_item,
                                             5, pos_dict=warm_dict['ii_pos_nb'])
            uni_item_1 = list(warm_dict['ii_test_nb'].keys())
            test_array = utils.label_neg_samp(uni_item_1, warm_dict['ii_test_nb'], warm_item,
                                              5, pos_dict=warm_dict['ii_pos_nb'])
            data['warm_embeddings'] = embeddings[user_num:]
            data['warm_object'] = np.array(list(warm_dict['emb_nb_reverse'].keys()), dtype=np.int)
            warm_object = 'item'
        elif args.n_hop > 2:
            raise NotImplementedError('Not implemented!')

    elif args.cold_object == 'user':
        if args.n_hop == 1:
            emb_users = list(warm_dict['emb_nb'].keys())
            train_array = utils.label_neg_samp(emb_users, warm_dict['emb_nb'], item_array,
                                               args.neg_rate, pos_dict=warm_dict['pos_nb'])
            val_users = list(warm_dict['val_nb'].keys())
            val_array = utils.label_neg_samp(val_users, warm_dict['val_nb'], item_array,
                                             5, pos_dict=warm_dict['pos_nb'])
            test_users = list(warm_dict['test_nb'].keys())
            test_array = utils.label_neg_samp(test_users, warm_dict['test_nb'], item_array,
                                              5, pos_dict=warm_dict['pos_nb'])
            data['warm_embeddings'] = embeddings[user_num:]
            data['warm_object'] = np.array(list(warm_dict['emb_nb_reverse'].keys()), dtype=np.int)
            warm_object = 'item'
        elif args.n_hop == 2:
            train_users = list(warm_dict['uu_train_nb'].keys())
            train_array = utils.label_neg_samp(train_users, warm_dict['uu_train_nb'], warm_user,
                                               args.neg_rate, pos_dict=warm_dict['uu_pos_nb'])
            val_users = list(warm_dict['uu_val_nb'].keys())
            val_array = utils.label_neg_samp(val_users, warm_dict['uu_val_nb'], warm_user,
                                             5, pos_dict=warm_dict['uu_pos_nb'])
            test_users = list(warm_dict['uu_test_nb'].keys())
            test_array = utils.label_neg_samp(test_users, warm_dict['uu_test_nb'], warm_user,
                                              5, pos_dict=warm_dict['uu_pos_nb'])
            data['warm_embeddings'] = embeddings[:user_num]
            data['warm_object'] = np.array(list(warm_dict['emb_nb'].keys()), dtype=np.int)
            warm_object = 'user'
        elif args.n_hop > 2:
            raise NotImplementedError('Not implemented!')

    warm_feature_file = data_path + f'/{warm_object}_content.npz'
    if os.path.exists(warm_feature_file):
        data['warm_features'] = sp.load_npz(warm_feature_file).todense().A
    else:
        data['warm_features'] = np.tile([None], [len(data['warm_embeddings']), 1])

    data['train'] = train_array
    data['val'] = val_array
    data['test'] = test_array

    print("n_user:%d  n_item:%d" % (data['user_num'], data['item_num']))
    print("cold feature: %s" % str(data['cold_features'].shape))
    print("warm embedding: %s" % str(data['warm_embeddings'].shape))
    print("warm feature: %s" % str(data['warm_features'].shape))
    print("Train: %s" % str(data['train'].shape))
    print("Val: %s" % str(data['val'].shape))
    print("Test: %s" % str(data['test'].shape))

    return data












