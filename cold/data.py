import numpy as np
import scipy.sparse
import utils
import pandas as pd

"""
This module contains class and methods related to data used in Heater  
"""


def load_eval_data(test_file):
    timer = utils.timer()
    test_data = pd.read_csv(test_file, dtype=np.int)[['user', 'item']]

    test_user_ids = np.array(list(set(test_data['user'].values)), np.int)
    test_item_ids = np.array(list(set(test_data['item'].values)), np.int)

    timer.toc('read %s triplets' % test_data.shape[0]).tic()
    eval_data = EvalData(test_data, test_item_ids, test_user_ids)
    print(eval_data.get_stats_string())
    return eval_data


class EvalData:
    """
    EvalData:
        EvalData packages test triplet (user, item, score) into appropriate formats for evaluation
        就是说在基于测试集构建稀疏矩阵和嵌入矩阵，进行测试，可以加速测试过程，因为忽略了不相关的user和item

        Args:
            test_df: user-item-interaction_value triplet to build the test data.
            test_item_ids: test-item-ids from train set or test set.

    """

    def __init__(self, test_df, test_item_ids, test_user_ids):

        # test_item_id_set and id2index
        self.test_item_ids = test_item_ids
        self.test_item_ids_map = {iid: i for i, iid in enumerate(self.test_item_ids)}
        # test_user_id_set and id2index
        self.test_user_ids = test_user_ids
        self.test_user_ids_map = {uid: i for i, uid in enumerate(self.test_user_ids)}

        # mapped
        _test_ij_for_inf = [(t[0], t[1]) for t in test_df.values]
        _test_i_for_inf = [self.test_user_ids_map[_t[0]] for _t in _test_ij_for_inf]
        _test_j_for_inf = [self.test_item_ids_map[_t[1]] for _t in _test_ij_for_inf]

        # transform interactions to sparse matrix
        self.R_test_inf = scipy.sparse.coo_matrix(
            (np.ones(len(_test_i_for_inf)),
             (_test_i_for_inf, _test_j_for_inf)),
            shape=[len(self.test_user_ids), len(self.test_item_ids)]
        ).tolil(copy=False)

        # allocate fields
        self.U_pref_test = None
        self.V_pref_test = None
        self.V_content_test = None
        self.U_content_test = None
        self.eval_batch = None

    def init_tf(self, user_factors, item_factors, user_content, item_content, eval_run_batchsize,
                cold_user=False, cold_item=False):
        """
        Get dense preference matrix for all test users and items;
        Get dense content matrix for cold users(items);
        Get user batch marks .

        Args:
            user_factors: user embedding matrix
            item_factors: item embedding matrix
            user_content: user content auxiliary vector matrix
            item_content: item content auxiliary vector matrix
            eval_run_batchsize: the batch size when test
            cold_user: whether to evaluate cold user
            cold_item: whether to evaluate cold item

        """
        # get dense pref matrix for all test user(item)
        # cold object is all 0.
        self.U_pref_test = user_factors[self.test_user_ids, :]
        self.V_pref_test = item_factors[self.test_item_ids, :]

        # get dense content matrix for test user(item) to eval cold start
        if cold_user:
            self.U_content_test = user_content[self.test_user_ids, :]
            if scipy.sparse.issparse(self.U_content_test):
                self.U_content_test = self.U_content_test.todense()

        if cold_item:
            self.V_content_test = item_content[self.test_item_ids, :]
            if scipy.sparse.issparse(self.V_content_test):
                self.V_content_test = self.V_content_test.todense()

        # get every batch's start index and end index for **users**
        eval_l = self.R_test_inf.shape[0]  # num of test users
        self.eval_batch = [(x, min(x + eval_run_batchsize, eval_l)) for x in range(0, eval_l, eval_run_batchsize)]

    def get_stats_string(self):
        return ('\tn_test_users:[%d]\n\tn_test_items:[%d]' % (len(self.test_user_ids), len(self.test_item_ids))
                + '\n\tR_train_inf: %s' % 'no R_train_inf for cold'
                + '\n\tR_test_inf: shape=%s nnz=[%d]' % (str(self.R_test_inf.shape), len(self.R_test_inf.nonzero()[0])))
