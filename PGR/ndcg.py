import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
import utils
import pandas as pd
from multiprocessing import Pool


def init(args):
    global POS_ITEMS, USER_NUM, ITEM_NUM, ITEM_ARRAY, Ks, VA_LBS, TS_LBS, TS_NDCG, BATCH_SIZE
    para_dict = pickle.load(open(args.datadir + args.dataset + '/warm_dict.pkl', 'rb'))
    print('ndcg init for %s, %d user %d items' % (args.dataset, para_dict['user_num'], para_dict['item_num']))
    POS_ITEMS = para_dict['pos_nb']
    USER_NUM = para_dict['user_num']
    ITEM_NUM = para_dict['item_num']
    ITEM_ARRAY = np.array(list(range(ITEM_NUM)))

    val_items = para_dict['val_nb']
    VA_LBS = utils.label_neg_samp(list(val_items.keys()),
                                  support_dict=val_items,
                                  item_array=ITEM_ARRAY,
                                  neg_rate=5,
                                  exclude_dict=POS_ITEMS)

    test_items = para_dict['test_nb']
    TS_LBS = utils.label_auc(list(test_items.keys()),
                             support_dict=test_items,
                             item_array=ITEM_ARRAY,
                             exclude_dict=POS_ITEMS)
    TS_NDCG = list(
        map(lambda x: utils.ndcg_sampling(x, test_items, 100, ITEM_ARRAY, POS_ITEMS), list(test_items.keys())))

    BATCH_SIZE = args.batch_size
    if isinstance(args.Ks, str):
        Ks = eval(args.Ks)
    else:
        Ks = args.Ks


def _dcg(x):
    # compute dcg_vle
    return x[0] + np.sum(x[1:] / np.log2(np.arange(2, len(x) + 1)))


def _auc(model, pred_func, lbs=None):
    if lbs is None:
        lbs = VA_LBS
    pred = pred_func(model, lbs)
    y_true = lbs[:, -1]
    fpr, tpr, thresholds = roc_curve(y_true, pred, pos_label=1)
    return auc(fpr, tpr)


def _fast_user_ndcg(pairs):
    def _simp_ndcg(lbs, pred):
        # Compute ncdg when the feeding data is one positive vs all negative
        labels = lbs[:, -1]
        pred = pred.reshape(-1)
        rerank_indices = np.argsort(pred)[::-1]
        rerank_labels = labels[rerank_indices]
        # DCG scores
        dcgs = np.array([_dcg(rerank_labels[:k]) for k in Ks])
        hrs = np.array([np.sum(rerank_labels[:k]) for k in Ks])
        return np.stack([hrs, dcgs])

    # compute the ndcg value of a given user
    return np.mean(np.stack(
        [_simp_ndcg(pairs[0][_], pairs[1][_]) for _ in range(pairs[0].shape[0])]), axis=0)


def batch_predict(model, pred_func, lbs):
    outs = []
    for begin in range(0, lbs.shape[0], BATCH_SIZE):
        end = min(begin + BATCH_SIZE, lbs.shape[0])
        batch_lbs = lbs[begin:end, :]
        outs.append(pred_func(model, batch_lbs))
    out = np.hstack(outs)
    return out


def _fast_ndcg(model, pred_func, blocks=None, partial=False):
    if blocks is None:
        blocks = TS_NDCG
    if partial:
        blocks = TS_NDCG[:partial]
    user_id = []
    for b in blocks:
        user_id.append((b[0][0][0], len(b)))
    b_size = blocks[0].shape[1]
    lbs = np.vstack(blocks).reshape([-1, 3])
    pred_lbs = batch_predict(model, pred_func, lbs).reshape([-1, b_size, 1])
    s = 0
    user_lbs = []
    for _ in user_id:
        user_lbs.append(pred_lbs[s:s + _[1]])
        s += _[1]
    assert s == pred_lbs.shape[0]
    ndcg_list = list(zip(blocks, user_lbs))
    with Pool(5) as pool:
        user_scores = np.stack(list(pool.map(_fast_user_ndcg, ndcg_list)))
    #     user_scores = np.stack(list(map(_fast_user_ndcg, ndcg_list)))
    scores = np.mean(np.stack(user_scores), axis=0)
    d = {}
    d['hr'] = scores[0]
    d['ndcg'] = scores[1]
    d['auc'] = _auc(model, pred_func, TS_LBS)
    return d
