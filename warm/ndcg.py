import pickle
import numpy as np
from sklearn.metrics import roc_curve, auc

global TR_ITEMS, USER_NUM, ITEM_NUM, ITEM_ARRAY, Ks, TS_LBS, TS_NDCG, BATCH_SIZE


def init(args):
    global TR_ITEMS, USER_NUM, ITEM_NUM, ITEM_ARRAY, Ks, TS_LBS, TS_NDCG, BATCH_SIZE
    para_dict = pickle.load(open(args.datadir + args.dataset + '/warm_dict.pkl', 'rb'))
    print('ndcg init for %s, %d user %d items' % (args.dataset, para_dict['user_num'], para_dict['item_num']))
    TR_ITEMS = para_dict['emb_nb']
    USER_NUM = para_dict['user_num']
    ITEM_NUM = para_dict['item_num']
    ITEM_ARRAY = np.array(list(range(ITEM_NUM)))
    TS_LBS = para_dict['test_sampling']
    TS_NDCG = para_dict['ndcg@50']
    BATCH_SIZE = args.batch_size
    if isinstance(args.Ks, str):
        Ks = eval(args.Ks)
    else:
        Ks = args.Ks


def AUC(model, pred_func, lbs=None):
    if lbs is None:
        lbs = TS_LBS
    pred = pred_func(model, lbs)
    y_true = lbs[:, -1]
    fpr, tpr, thresholds = roc_curve(y_true, pred, pos_label=1)
    return auc(fpr, tpr)


def l1out_test(model, pred_func, blocks=None, partial=False):
    if blocks is None:
        blocks = TS_NDCG
    if partial:
        blocks = TS_NDCG[:partial]
    # Leave one out test
    user_scores = np.stack(list(map(lambda x: _simp_user_ndcg(model, pred_func, x), blocks)))
    scores = np.mean(np.stack(user_scores), axis=0)
    d = {}
    d['hr'] = scores[0]
    d['ndcg'] = scores[1]
    d['auc'] = AUC(model, pred_func, TS_LBS)
    return d


def _simp_user_ndcg(model, pred_func, user_block):
    def _simp_ndcg(model, lbs):
        # Compute ncdg when the feeding data is one positive vs all negative
        pred = pred_func(model, lbs)
        labels = lbs[:, -1]
        rerank_indices = np.argsort(pred)[::-1]
        rerank_labels = labels[rerank_indices]
        # DCG scores
        dcgs = np.array([_dcg(rerank_labels[:k]) for k in Ks])
        hrs = np.array([np.sum(rerank_labels[:k]) for k in Ks])
        return np.stack([hrs, dcgs])

    # compute the ndcg value of a given user
    return np.mean(np.stack(
        [_simp_ndcg(model, user_block[_]) for _ in range(user_block.shape[0])]), axis=0)


def _dcg(x):
    # compute dcg_vle
    return x[0] + np.sum(x[1:] / np.log2(np.arange(2, len(x) + 1)))
