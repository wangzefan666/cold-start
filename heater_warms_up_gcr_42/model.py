import tensorflow as tf
import numpy as np


def l2_norm(para):
    return (1 / 2) * tf.reduce_sum(tf.square(para))


def dense_batch_fc_tanh(x, units, is_training, scope, dropout=0.5, act='tanh', init_meth='normal'):
    with tf.variable_scope(scope):
        if init_meth == 'normal':
            init = tf.truncated_normal_initializer(stddev=0.01)
        elif init_meth == 'xavier':
            init = tf.contrib.layers.xavier_initializer(uniform=True)
        h1_w = tf.get_variable(scope + '_w', shape=[x.get_shape().as_list()[1], units], initializer=init)
        h1_b = tf.get_variable(scope + '_b', shape=[1, units], initializer=tf.zeros_initializer())
        h1 = tf.matmul(x, h1_w) + h1_b
        h2 = tf.layers.batch_normalization(h1, momentum=0.9, training=is_training)
        if act == 'tanh':
            h3 = tf.nn.tanh(h2)
        elif act == 'relu':
            h3 = tf.nn.relu(h2)
        elif act == 'leaky_relu':
            h3 = tf.nn.leaky_relu(h2, alpha=0.01)
        return tf.layers.dropout(h3, rate=dropout, training=is_training), l2_norm(h1_w) + l2_norm(h1_b)


def dense_fc(x, units, scope, init_meth='normal'):
    with tf.variable_scope(scope):
        if init_meth == 'normal':
            init = tf.truncated_normal_initializer(stddev=0.01)
        elif init_meth == 'xavier':
            init = tf.contrib.layers.xavier_initializer(uniform=True)
        h1_w = tf.get_variable(scope + '_w', shape=[x.get_shape().as_list()[1], units], initializer=init)
        h1_b = tf.get_variable(scope + '_b', shape=[1, units], initializer=tf.zeros_initializer())
        h1 = tf.matmul(x, h1_w) + h1_b
        return h1, l2_norm(h1_w) + l2_norm(h1_b)


class Heater:
    def __init__(self, latent_rank_in, user_content_rank, item_content_rank, args):

        self.rank_in = latent_rank_in  # input embedding dimension
        self.phi_u_dim = user_content_rank  # user content dimension
        self.phi_v_dim = item_content_rank  # item content dimension

        self.model_select = args.model_select  # model architecture
        self.rank_out = args.rank_out  # output dimension
        self.reg = args.reg  # coefficient of regularization
        self.alpha = args.alpha  # coefficient of diff loss
        self.dim = args.dim  # num of experts
        self.dropout = args.model_dropout
        self.loss_func = args.loss
        self.act = args.act
        self.optim = args.optim
        self.init = args.init
        self.beta = args.beta
        self.omega = args.omega
        self.reg_loss = 0.

    def build_model(self):
        self.lr_placeholder = tf.placeholder(tf.float32, shape=[], name='learn_rate')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.target = tf.placeholder(tf.float32, shape=[None], name='target')

        self.Uin = tf.placeholder(tf.float32, shape=[None, 3, self.rank_in], name='U_in_raw')
        Uin = self.Uin[:, 0, :]
        U_nei_1 = self.Uin[:, 1, :]
        U_nei_2 = self.Uin[:, 2, :]

        self.Vin = tf.placeholder(tf.float32, shape=[None, 3, self.rank_in], name='V_in_raw')
        Vin = self.Vin[:, 0, :]
        V_nei_1 = self.Vin[:, 1, :]
        V_nei_2 = self.Vin[:, 2, :]

        # calculate diff loss
        if self.phi_v_dim > 0:
            self.Vcontent = tf.placeholder(tf.float32, shape=[None, self.phi_v_dim], name='V_content')
            self.fake_v_nei_1 = tf.placeholder(tf.float32, shape=[None, self.rank_in], name='V_fake_nei_1')
            self.fake_v_nei_2 = tf.placeholder(tf.float32, shape=[None, self.rank_in], name='V_fake_nei_2')
            self.dropout_item_indicator = tf.placeholder(tf.float32, shape=[None, 1], name='dropout_item_indicator')

            # v_content
            vcontent_gate, vcontent_gate_reg = dense_fc(self.Vcontent, self.dim, 'vcontent_gate_layer', init_meth=self.init)
            vcontent_gate = tf.nn.tanh(vcontent_gate)
            self.reg_loss += vcontent_gate_reg
            vcontent_expert_list = []
            for i in range(self.dim):
                tmp_expert = self.Vcontent
                for ihid, hid in enumerate(self.model_select):
                    tmp_expert, tmp_reg = dense_fc(tmp_expert, hid, 'Vexpert_' + str(ihid) + '_' + str(i), init_meth=self.init)
                    tmp_expert = tf.nn.tanh(tmp_expert)
                    self.reg_loss += tmp_reg
                vcontent_expert_list.append(tf.reshape(tmp_expert, [-1, 1, self.model_select[-1]]))
            vcontent_expert_concat = tf.concat(vcontent_expert_list, axis=1)
            Vcontent_last = tf.matmul(tf.reshape(vcontent_gate, [-1, 1, self.dim]), vcontent_expert_concat)
            Vcontent_last = tf.reshape(tf.nn.tanh(Vcontent_last), [-1, self.model_select[-1]])
            # diff_loss 跟 indicator 是无关的
            diff_item_loss = tf.reduce_mean(tf.reduce_sum(tf.square(Vcontent_last - Vin), axis=1))
            v_last = (Vin * (1 - self.dropout_item_indicator) + Vcontent_last * self.dropout_item_indicator)

            # v_nei_1
            v_nei_gate_1, v_nei_gate_1_reg = dense_fc(self.fake_v_nei_1, self.dim, 'v_nei_gate_1_layer', init_meth=self.init)
            v_nei_gate_1 = tf.nn.tanh(v_nei_gate_1)
            self.reg_loss += v_nei_gate_1_reg
            v_nei_expert_list_1 = []
            for i in range(self.dim):
                tmp_expert = self.fake_v_nei_1
                for ihid, hid in enumerate(self.model_select):
                    tmp_expert, tmp_reg = dense_fc(tmp_expert, hid, 'v_nei_expert_1_' + str(ihid) + '-' + str(i), init_meth=self.init)
                    tmp_expert = tf.nn.tanh(tmp_expert)
                    self.reg_loss += tmp_reg
                v_nei_expert_list_1.append(tf.reshape(tmp_expert, [-1, 1, self.model_select[-1]]))
            v_nei_expert_concat_1 = tf.concat(v_nei_expert_list_1, axis=1)
            v_nei_last_1 = tf.matmul(tf.reshape(v_nei_gate_1, [-1, 1, self.dim]), v_nei_expert_concat_1)
            v_nei_last_1 = tf.reshape(tf.nn.tanh(v_nei_last_1), [-1, self.model_select[-1]])
            diff_vnei1_loss = tf.reduce_mean(tf.reduce_sum(tf.square(v_nei_last_1 - V_nei_1), axis=1))
            v_nei_last_1 = (V_nei_1 * (1 - self.dropout_item_indicator) + v_nei_last_1 * self.dropout_item_indicator)

            # v_nei_2
            v_nei_gate_2, v_nei_gate_2_reg = dense_fc(self.fake_v_nei_2, self.dim, 'v_nei_gate_2_layer', init_meth=self.init)
            v_nei_gate_2 = tf.nn.tanh(v_nei_gate_2)
            self.reg_loss += v_nei_gate_2_reg
            v_nei_expert_list_2 = []
            for i in range(self.dim):
                tmp_expert = self.fake_v_nei_2
                for ihid, hid in enumerate(self.model_select):
                    tmp_expert, tmp_reg = dense_fc(tmp_expert, hid, 'v_nei_expert_2_' + str(ihid) + '-' + str(i), init_meth=self.init)
                    tmp_expert = tf.nn.tanh(tmp_expert)
                    self.reg_loss += tmp_reg
                v_nei_expert_list_2.append(tf.reshape(tmp_expert, [-1, 1, self.model_select[-1]]))
            v_nei_expert_concat_2 = tf.concat(v_nei_expert_list_2, axis=1)
            v_nei_last_2 = tf.matmul(tf.reshape(v_nei_gate_2, [-1, 1, self.dim]), v_nei_expert_concat_2)
            v_nei_last_2 = tf.reshape(tf.nn.tanh(v_nei_last_2), [-1, self.model_select[-1]])
            diff_vnei2_loss = tf.reduce_mean(tf.reduce_sum(tf.square(v_nei_last_2 - V_nei_2), axis=1))
            v_nei_last_2 = (V_nei_2 * (1 - self.dropout_item_indicator) + v_nei_last_2 * self.dropout_item_indicator)

        else:
            v_last = Vin
            v_nei_last_1 = V_nei_1
            v_nei_last_2 = V_nei_2
            diff_item_loss = 0
            diff_vnei1_loss = 0
            diff_vnei2_loss = 0

        if self.phi_u_dim > 0:
            self.Ucontent = tf.placeholder(tf.float32, shape=[None, self.phi_u_dim], name='U_content')
            self.dropout_user_indicator = tf.placeholder(tf.float32, shape=[None, 1], name='dropout_user_indicator')

            ucontent_gate, ucontent_gate_reg = dense_fc(self.Ucontent, self.dim, 'ucontent_gate_layer', init_meth=self.init)
            ucontent_gate = tf.nn.tanh(ucontent_gate)
            self.reg_loss += ucontent_gate_reg

            ucontent_expert_list = []
            for i in range(self.dim):
                tmp_expert = self.Ucontent
                for ihid, hid in enumerate(self.model_select):
                    tmp_expert, tmp_reg = dense_fc(tmp_expert, hid, 'Uexpert_' + str(ihid) + '_' + str(i), init_meth=self.init)
                    tmp_expert = tf.nn.tanh(tmp_expert)
                    self.reg_loss += tmp_reg
                ucontent_expert_list.append(tf.reshape(tmp_expert, [-1, 1, self.model_select[-1]]))
            ucontent_expert_concat = tf.concat(ucontent_expert_list, 1)
            Ucontent_last = tf.matmul(tf.reshape(ucontent_gate, [-1, 1, self.dim]), ucontent_expert_concat)
            Ucontent_last = tf.reshape(tf.nn.tanh(Ucontent_last), [-1, self.model_select[-1]])
            # diff_loss 跟 indicator 是无关的
            diff_user_loss = tf.reduce_mean(tf.reduce_sum(tf.square(Ucontent_last - Uin), axis=1))
            u_last = (Uin * (1 - self.dropout_user_indicator) + Ucontent_last * self.dropout_user_indicator)
        else:
            u_last = Uin
            diff_user_loss = 0

        # calc recommendation loss
        for ihid, hid in enumerate(self.rank_out):
            u_last, reg_1 = dense_batch_fc_tanh(u_last, hid, self.is_training, 'user_layer_%d' % ihid,
                                                dropout=self.dropout, act=self.act, init_meth=self.init)
            v_last, reg_2 = dense_batch_fc_tanh(v_last, hid, self.is_training, 'item_layer_%d' % ihid,
                                                dropout=self.dropout, act=self.act, init_meth=self.init)
            U_nei_1, reg_3 = dense_batch_fc_tanh(U_nei_1, hid, self.is_training, 'U_nei_1_layer_%d' % ihid,
                                                 dropout=self.dropout, act=self.act, init_meth=self.init)
            v_nei_last_1, reg_4 = dense_batch_fc_tanh(v_nei_last_1, hid, self.is_training, 'V_nei_1_layer_%d' % ihid,
                                                      dropout=self.dropout, act=self.act, init_meth=self.init)
            U_nei_2, reg_5 = dense_batch_fc_tanh(U_nei_2, hid, self.is_training, 'U_nei_2_layer_%d' % ihid,
                                                 dropout=self.dropout, act=self.act, init_meth=self.init)
            v_nei_last_2, reg_6 = dense_batch_fc_tanh(v_nei_last_2, hid, self.is_training, 'V_nei_2_layer_%d' % ihid,
                                                      dropout=self.dropout, act=self.act, init_meth=self.init)
            self.reg_loss += (reg_1 + reg_2 + reg_3 + reg_4 + reg_5 + reg_6)

        self.U_embedding, reg_1 = dense_fc(u_last, self.rank_in, "U_embedding", init_meth=self.init)
        self.V_embedding, reg_2 = dense_fc(v_last, self.rank_in, "V_embedding", init_meth=self.init)
        self.U_nei_1, reg_3 = dense_fc(U_nei_1, self.rank_in, "U_nei_1", init_meth=self.init)
        self.V_nei_1, reg_4 = dense_fc(v_nei_last_1, self.rank_in, "V_nei_1", init_meth=self.init)
        self.U_nei_2, reg_5 = dense_fc(U_nei_2, self.rank_in, "U_nei_2", init_meth=self.init)
        self.V_nei_2, reg_6 = dense_fc(v_nei_last_2, self.rank_in, "V_nei_2", init_meth=self.init)
        self.reg_loss += (reg_1 + reg_2 + reg_3 + reg_4 + reg_5 + reg_6)

        # =======================================
        # =========== Weighted Sum ==============
        # =======================================
        self.ws_u_embedding = tf.stack([self.U_embedding, self.U_nei_1, self.U_nei_2], axis=1)
        self.ws_u_embedding = tf.reduce_mean(self.ws_u_embedding, axis=1)
        self.ws_v_embedding = tf.stack([self.V_embedding, self.V_nei_1, self.V_nei_2], axis=1)
        self.ws_v_embedding = tf.reduce_mean(self.ws_v_embedding, axis=1)

        # loss
        self.reg_loss *= self.reg
        with tf.variable_scope("loss"):
            self.diff_loss = self.alpha * (diff_item_loss + diff_user_loss) + self.beta * (diff_vnei1_loss) + self.omega * (diff_vnei2_loss)
            # output of the model, the predicted scores
            self.preds = tf.reduce_sum(tf.multiply(self.ws_u_embedding, self.ws_v_embedding), axis=-1)
            if self.loss_func == 'sse':
                self.rec_loss = tf.reduce_mean(tf.squared_difference(self.preds, self.target))
            elif self.loss_func == 'sce':
                self.rec_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.target, logits=self.preds)
            self.loss = self.rec_loss + self.reg_loss + self.diff_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            if self.optim == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(self.lr_placeholder, 0.9).minimize(self.loss)
            elif self.optim == 'adam':
                self.optimizer = tf.train.AdamOptimizer(self.lr_placeholder).minimize(self.loss)

    def build_predictor(self):
        with tf.variable_scope("eval"):
            self.eval_preds_cold = tf.matmul(self.ws_u_embedding, self.ws_v_embedding, transpose_b=True, name='pred_cold')

    def get_eval_dict(self, U_pref, V_pref, eval_data, U_content, V_content,
                      fake_u_nei, fake_v_nei, warm=False):
        _eval_dict = {
            self.Uin: U_pref,  # test batch users embedding
            self.Vin: V_pref,  # all test item embedding
            self.is_training: False
        }

        dropout_item_indicator = np.zeros((len(V_pref), 1))
        if self.phi_v_dim > 0:
            if not warm:
                dropout_item_indicator[eval_data.test_items] = 1
        _eval_dict[self.dropout_item_indicator] = dropout_item_indicator
        _eval_dict[self.Vcontent] = V_content
        _eval_dict[self.fake_v_nei_1] = fake_v_nei[:, 1, :]
        _eval_dict[self.fake_v_nei_2] = fake_v_nei[:, 2, :]

        if self.phi_u_dim > 0:
            dropout_user_indicator = np.ones((len(U_pref), 1)) if not warm else np.zeros((len(U_pref), 1))
            _eval_dict[self.dropout_user_indicator] = dropout_user_indicator
            _eval_dict[self.Ucontent] = U_content

        return _eval_dict

