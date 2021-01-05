import tensorflow as tf
import numpy as np


def l2_norm(para):
    return (1 / 2) * tf.reduce_sum(tf.square(para))


def dense_batch_fc_tanh(x, units, is_training, scope, do_norm=False):
    with tf.variable_scope(scope):
        init = tf.truncated_normal_initializer(stddev=0.01)
        h1_w = tf.get_variable(scope + '_w', shape=[x.get_shape().as_list()[1], units], initializer=init)
        h1_b = tf.get_variable(scope + '_b', shape=[1, units], initializer=tf.zeros_initializer())
        h1 = tf.matmul(x, h1_w) + h1_b
        if do_norm:
            h2 = tf.contrib.layers.batch_norm(h1, decay=0.9, center=True, scale=True, is_training=is_training, scope=scope + '_bn')
            return tf.nn.tanh(h2), l2_norm(h1_w) + l2_norm(h1_b)
        else:
            return tf.nn.tanh(h1), l2_norm(h1_w) + l2_norm(h1_b)


def dense_fc(x, units, scope):
    with tf.variable_scope(scope):
        init = tf.truncated_normal_initializer(stddev=0.01)
        h1_w = tf.get_variable(scope + '_w', shape=[x.get_shape().as_list()[1], units], initializer=init)
        h1_b = tf.get_variable(scope + '_b', shape=[1, units], initializer=tf.zeros_initializer())
        h1 = tf.matmul(x, h1_w) + h1_b
        return h1, l2_norm(h1_w) + l2_norm(h1_b)


class Heater:
    def __init__(self, latent_rank_in, user_content_rank, item_content_rank,
                 model_select, rank_out, reg, alpha, dim):

        self.rank_in = latent_rank_in  # input embedding dimension
        self.phi_u_dim = user_content_rank  # user content dimension
        self.phi_v_dim = item_content_rank  # item content dimension
        self.model_select = model_select  # model architecture
        self.rank_out = rank_out  # output dimension
        self.reg = reg  # coefficient of regularization
        self.alpha = alpha  # coefficient of diff loss
        self.dim = dim  # num of experts

        # inputs
        self.Uin = None  # input user embedding
        self.Vin = None  # input item embedding
        self.Ucontent = None  # input user content
        self.Vcontent = None  # input item content
        self.is_training = None
        self.target = None  # input training target

        # self.eval_trainR = None  # input training rating matrix for evaluation
        self.U_pref_tf = None
        self.V_pref_tf = None
        self.rand_target_ui = None

        # outputs in the model
        self.preds = None  # output of the model, the predicted scores
        self.optimizer = None  # the optimizer
        self.rec_loss = None
        self.diff_loss = None
        self.reg_loss = 0.
        self.loss = None

        self.U_embedding = None  # new user embedding
        self.V_embedding = None  # new item embedding

        self.lr_placeholder = None  # learning rate
        self.dropout_user_indicator = None  # dropout users' pretrain CF rep filter. 1:dropout 0:retain.
        self.dropout_item_indicator = None

        # predictor
        self.preds_random = None
        self.tf_latent_topk_cold = None
        self.eval_preds_cold = None  # the top-k predicted indices for cold evaluation
        self.eval_preds_warm = None

    def build_model(self):
        self.lr_placeholder = tf.placeholder(tf.float32, shape=[], name='learn_rate')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.target = tf.placeholder(tf.float32, shape=[None], name='target')

        self.Uin = tf.placeholder(tf.float32, shape=[None, self.rank_in], name='U_in_raw')
        self.Vin = tf.placeholder(tf.float32, shape=[None, self.rank_in], name='V_in_raw')

        # calculate diff loss
        if self.phi_v_dim > 0:
            self.Vcontent = tf.placeholder(tf.float32, shape=[None, self.phi_v_dim], name='V_content')
            self.dropout_item_indicator = tf.placeholder(tf.float32, shape=[None, 1], name='dropout_item_indicator')

            vcontent_gate, vcontent_gate_reg = dense_fc(self.Vcontent, self.dim, 'vcontent_gate_layer')
            vcontent_gate = tf.nn.tanh(vcontent_gate)
            self.reg_loss += vcontent_gate_reg

            vcontent_expert_list = []
            for i in range(self.dim):
                tmp_expert = self.Vcontent
                for ihid, hid in enumerate(self.model_select):
                    tmp_expert, tmp_reg = dense_fc(tmp_expert, hid, 'Vexpert_' + str(ihid) + '_' + str(i))
                    tmp_expert = tf.nn.tanh(tmp_expert)
                    self.reg_loss += tmp_reg
                vcontent_expert_list.append(tf.reshape(tmp_expert, [-1, 1, self.rank_out]))
            vcontent_expert_concat = tf.concat(vcontent_expert_list, axis=1)

            Vcontent_last = tf.matmul(tf.reshape(vcontent_gate, [-1, 1, self.dim]), vcontent_expert_concat)
            Vcontent_last = tf.reshape(tf.nn.tanh(Vcontent_last), [-1, self.rank_out])
            # 为啥diff loss没有用mean
            diff_item_loss = self.alpha * (tf.reduce_mean(tf.reduce_sum(tf.square(Vcontent_last - self.Vin), axis=1)))

            v_last = (self.Vin * (1 - self.dropout_item_indicator) + Vcontent_last * self.dropout_item_indicator)
        else:
            v_last = self.Vin
            diff_item_loss = 0

        if self.phi_u_dim > 0:
            self.Ucontent = tf.placeholder(tf.float32, shape=[None, self.phi_u_dim], name='U_content')
            self.dropout_user_indicator = tf.placeholder(tf.float32, shape=[None, 1], name='dropout_user_indicator')

            ucontent_gate, ucontent_gate_reg = dense_fc(self.Ucontent, self.dim, 'ucontent_gate_layer')
            ucontent_gate = tf.nn.tanh(ucontent_gate)
            self.reg_loss += ucontent_gate_reg

            ucontent_expert_list = []
            for i in range(self.dim):
                tmp_expert = self.Ucontent
                for ihid, hid in enumerate(self.model_select):
                    tmp_expert, tmp_reg = dense_fc(tmp_expert, hid, 'Uexpert_' + str(ihid) + '_' + str(i))
                    tmp_expert = tf.nn.tanh(tmp_expert)
                    self.reg_loss += tmp_reg
                ucontent_expert_list.append(tf.reshape(tmp_expert, [-1, 1, self.rank_out]))
            ucontent_expert_concat = tf.concat(ucontent_expert_list, 1)

            Ucontent_last = tf.matmul(tf.reshape(ucontent_gate, [-1, 1, self.dim]), ucontent_expert_concat)
            Ucontent_last = tf.reshape(tf.nn.tanh(Ucontent_last), [-1, self.rank_out])
            # 为啥diff loss没有用mean
            diff_user_loss = self.alpha * (tf.reduce_mean(tf.reduce_sum(tf.square(Ucontent_last - self.Uin), axis=1)))

            u_last = (self.Uin * (1 - self.dropout_user_indicator) + Ucontent_last * self.dropout_user_indicator)
        else:
            u_last = self.Uin
            diff_user_loss = 0

        # calc recommendation loss
        for ihid, hid in enumerate([self.rank_out]):
            u_last, u_reg = dense_batch_fc_tanh(u_last, hid, self.is_training, 'user_layer_%d' % ihid, do_norm=True)
            v_last, v_reg = dense_batch_fc_tanh(v_last, hid, self.is_training, 'item_layer_%d' % ihid, do_norm=True)
            self.reg_loss += u_reg
            self.reg_loss += v_reg

        with tf.variable_scope("U_embedding"):
            u_emb_w = tf.Variable(tf.truncated_normal([u_last.get_shape().as_list()[1], self.rank_out], stddev=0.01), name='u_emb_w')
            u_emb_b = tf.Variable(tf.zeros([1, self.rank_out]), name='u_emb_b')
            self.U_embedding = tf.matmul(u_last, u_emb_w) + u_emb_b

        with tf.variable_scope("V_embedding"):
            v_emb_w = tf.Variable(tf.truncated_normal([v_last.get_shape().as_list()[1], self.rank_out], stddev=0.01), name='v_emb_w')
            v_emb_b = tf.Variable(tf.zeros([1, self.rank_out]), name='v_emb_b')
            self.V_embedding = tf.matmul(v_last, v_emb_w) + v_emb_b

        self.reg_loss += (l2_norm(v_emb_w) + l2_norm(v_emb_b) + l2_norm(u_emb_w) + l2_norm(u_emb_b))
        self.reg_loss *= self.reg

        with tf.variable_scope("loss"):
            self.diff_loss = diff_item_loss + diff_user_loss
            self.preds = tf.multiply(self.U_embedding, self.V_embedding)
            self.preds = tf.reduce_sum(self.preds, axis=1)  # output of the model, the predicted scores
            self.rec_loss = tf.reduce_mean(tf.squared_difference(self.preds, self.target))
            self.loss = self.rec_loss + self.reg_loss + self.diff_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            self.optimizer = tf.train.MomentumOptimizer(self.lr_placeholder, 0.9).minimize(self.loss)

    def build_predictor(self):
        with tf.variable_scope("eval"):
            self.eval_preds_cold = tf.matmul(self.U_embedding, self.V_embedding, transpose_b=True, name='pred_cold')
            self.eval_preds_warm = tf.matmul(self.Uin, self.Vin, transpose_b=True, name='pred_warm')

    def get_eval_dict(self, U_pref, V_pref, eval_data, U_content=None, V_content=None, warm=False):

        _eval_dict = {
            self.Uin: U_pref,  # test batch users embedding
            self.Vin: V_pref,  # all test item embedding
            self.is_training: False
        }

        if self.phi_v_dim > 0:
            dropout_item_indicator = np.zeros((len(V_pref), 1))
            dropout_item_indicator[eval_data.cold_items] = 1
            _eval_dict[self.dropout_item_indicator] = dropout_item_indicator
            _eval_dict[self.Vcontent] = V_content

        if self.phi_u_dim > 0:
            dropout_user_indicator = np.ones((len(U_pref), 1)) if not warm else np.zeros((len(U_pref), 1))
            _eval_dict[self.dropout_user_indicator] = dropout_user_indicator
            _eval_dict[self.Ucontent] = U_content

        return _eval_dict

