import numpy as np
import scipy.sparse
import utils
import pandas as pd

"""
This module contains class and methods related to data used in Heater  
"""

class EvalData:
    """
    EvalData:
        EvalData packages test triplet (user, item, score) into appropriate formats for evaluation
        就是说在基于测试集构建稀疏矩阵和嵌入矩阵，进行测试，可以加速测试过程，因为忽略了不相关的user和item

        Args:
            test_df: user-item-interaction_value triplet to build the test data.
            test_item_ids: test-item-ids from whole set
    """

    def __init__(self, test_rec, test_items, test_users, cold_items, n_items, batch_size):
        # 需要所有 item 一起 rank，无论 cold 还是 warm，推荐系统不知道你是 cold 还是 warm
        self.test_items = test_items
        self.cold_items = cold_items
        # test_user_id_set and id2index
        self.test_users = test_users
        self.test_user_ids_map = {uid: i for i, uid in enumerate(self.test_users)}

        # transform interactions to sparse matrix
        _test_i_for_inf = [self.test_user_ids_map[_t[0]] for _t in test_rec]  # mapped
        _test_j_for_inf = [_t[1] for _t in test_rec]
        self.R_test_inf = scipy.sparse.coo_matrix(
            (np.ones(len(_test_i_for_inf)),
             (_test_i_for_inf, _test_j_for_inf)),
            shape=[len(self.test_users), n_items]
        ).tolil()

        # allocate fields
        eval_l = self.R_test_inf.shape[0]  # num of test users
        self.eval_batch = [(x, min(x + batch_size, eval_l)) for x in range(0, eval_l, batch_size)]

        print('\tn_test_users:[%d]\n\tn_test_items:[%d]' % (len(self.test_users), len(self.test_items))
              + '\n\tR_test_inf: shape=%s nnz=[%d]' % (str(self.R_test_inf.shape), self.R_test_inf.nnz))
