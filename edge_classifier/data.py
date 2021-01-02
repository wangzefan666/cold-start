import utils
import copy
import numpy as np
import scipy.sparse as sp
import pandas as pd
import pickle


def load_data(args):
    data_path = args.datadir + args.data
    warm_dict_file = data_path + '/warm_dict.pkl'
    cold_dict_file = data_path + f'/cold_{args.cold_object}_dict.pkl'
    feature_file = data_path + f'/{args.cold_object}_content.npz'
    warm_embedding_file = data_path + f'/{args.warm_model}.npy'

    warm_dict = pickle.load(open(warm_dict_file, 'rb'))
    cold_dict = pickle.load(open(cold_dict_file, 'rb'))
    user_num = warm_dict['user_num']
    item_num = warm_dict['item_num']
    item_array = np.arange(item_num)
    user_array = np.arange(user_num)
    warm_user = np.array(list(warm_dict['emb_nb'].keys())).astype(np.int)
    warm_item = np.array(list(warm_dict['emb_nb_reverse'].keys())).astype(np.int)

    data = {}
    data['user_num'] = user_num
    data['item_num'] = item_num
    data['cold_features'] = sp.load_npz(feature_file).tolil()
    embeddings = np.load(warm_embedding_file)

    if args.cold_object == 'item':
        if args.n_hop == 1:
            uni_items = list(warm_dict['emb_nb_reverse'].keys())
            train_array = utils.label_neg_samp(uni_items, warm_dict['emb_nb_reverse'], user_array,
                                               args.neg_rate, pos_dict=warm_dict['pos_nb_reverse'])
            uni_items = list(warm_dict['val_nb_reverse'].keys())
            val_array = utils.label_neg_samp(uni_items, warm_dict['val_nb_reverse'], user_array,
                                             args.neg_rate, pos_dict=warm_dict['pos_nb_reverse'])
            uni_items = list(warm_dict['test_nb_reverse'].keys())
            test_array = utils.label_neg_samp(uni_items, warm_dict['test_nb_reverse'], user_array,
                                              args.neg_rate, pos_dict=warm_dict['pos_nb_reverse'])
            data['warm_embeddings'] = embeddings[:user_num]
            data['warm_object'] = np.array(list(warm_dict['emb_nb'].keys()), dtype=np.int)
        if args.n_hop == 2:
            uni_item_1 = list(warm_dict['ii_train_nb'].keys())
            train_array = utils.label_neg_samp(uni_item_1, warm_dict['ii_train_nb'], warm_item,
                                               args.neg_rate, pos_dict=warm_dict['ii_pos_nb'])
            uni_item_1 = list(warm_dict['ii_val_nb'].keys())
            val_array = utils.label_neg_samp(uni_item_1, warm_dict['ii_val_nb'], warm_item,
                                             args.neg_rate, pos_dict=warm_dict['ii_pos_nb'])
            uni_item_1 = list(warm_dict['ii_test_nb'].keys())
            test_array = utils.label_neg_samp(uni_item_1, warm_dict['ii_test_nb'], warm_item,
                                              args.neg_rate, pos_dict=warm_dict['ii_pos_nb'])
            data['warm_embeddings'] = embeddings[user_num:]
            data['warm_object'] = np.array(list(warm_dict['emb_nb_reverse'].keys()), dtype=np.int)
        elif args.n_hop > 2:
            raise NotImplementedError('Not implemented!')

        data['cold_item'] = np.array(list(cold_dict['val_nb_reverse']) + list(cold_dict['test_nb_reverse']), dtype=np.int)

    data['train'] = train_array
    data['val'] = val_array
    data['test'] = test_array

    return data












