import tensorflow as tf
import numpy as np


def l2_norm(para):
    return (1 / 2) * tf.reduce_sum(tf.square(para))


def MLP_layer(x, units, is_training, scope, dropout=0.5, is_first=False, is_output=False):
    with tf.variable_scope(scope):
        init = tf.contrib.layers.xavier_initializer(uniform=True)
        if is_first:
            mlp_in = tf.layers.batch_normalization(x, momentum=0.9, training=is_training)
        else:
            mlp_in = x
        h1_w = tf.get_variable(scope + '_w', shape=[mlp_in.get_shape().as_list()[1], units], initializer=init)
        h1_b = tf.get_variable(scope + '_b', shape=[1, units], initializer=tf.zeros_initializer())
        h1 = tf.matmul(mlp_in, h1_w) + h1_b
        if is_output:
            return h1, l2_norm(h1_w) + l2_norm(h1_b)
        else:
            h2 = tf.layers.batch_normalization(h1, momentum=0.9, training=is_training)
            h3 = tf.nn.leaky_relu(h2, alpha=0.01)
            h4 = tf.layers.dropout(h3, rate=dropout, training=is_training)
            return h4, l2_norm(h1_w) + l2_norm(h1_b)


class Edge_classifier:
    def __init__(self, cold_feature_dim, warm_feature_dim, embed_dim, type, lr, n_layers, hid_dim, dropout):

        self.cold_feature_dim = cold_feature_dim
        self.warm_feature_dim = warm_feature_dim  # input embedding dimension
        self.embed_dim = embed_dim  # user content dimension
        self.type = type
        self.n_layers = n_layers
        self.hid_dim = hid_dim
        self.dropout = dropout

        self.reg_loss = 0.
        self.reg = 1e-4
        self.lr = lr

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.target = tf.placeholder(tf.float32, shape=[None], name='target')
        self.root_feature = tf.placeholder(tf.float32, shape=[None, self.cold_feature_dim], name='root_feature')

        if self.type == 0:
            self.warm_embedding = tf.placeholder(tf.float32, shape=[None, self.embed_dim], name='warm_embedding')
            self.MLP_output = tf.concat([self.root_feature, self.warm_embedding], axis=-1)
        elif self.type == 1:
            self.warm_feature = tf.placeholder(tf.float32, shape=[None, self.warm_feature_dim], name='warm_feature')
            self.MLP_output = tf.concat([self.root_feature, self.warm_feature], axis=-1)
        elif self.type == 2:
            self.warm_embedding = tf.placeholder(tf.float32, shape=[None, self.embed_dim], name='warm_embedding')
            self.warm_feature = tf.placeholder(tf.float32, shape=[None, self.warm_feature_dim], name='warm_feature')
            self.MLP_output = tf.concat([self.root_feature, self.warm_embedding, self.warm_feature], axis=-1)

        # =========== MLP =================
        if self.n_layers <= 1:
            self.MLP_output, h_reg = MLP_layer(
                self.MLP_output, 1, self.is_training, 'MLP_%d' % self.n_layers, is_first=True, is_output=True)
            self.reg_loss += h_reg
        else:
            for i in range(self.n_layers - 1):
                self.MLP_output, h_reg = MLP_layer(
                    self.MLP_output, self.hid_dim, self.is_training, 'MLP_%d' % (i + 1), self.dropout, is_first=not bool(i))
                self.reg_loss += h_reg
            self.MLP_output, h_reg = MLP_layer(
                self.MLP_output, 1, self.is_training, 'MLP_%d' % self.n_layers, is_output=True)
            self.reg_loss += h_reg

        # loss
        with tf.variable_scope("loss"):
            # output of the model, the predicted scores
            self.reg_loss *= self.reg
            self.preds = tf.reshape(self.MLP_output, [-1])
            # self.pred_loss = tf.reduce_mean(tf.squared_difference(self.preds, self.target))
            self.pred_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.target, logits=self.preds)
            self.loss = self.pred_loss + self.reg_loss

        # update
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def get_eval_dict(self, root_feature, warm_embedding, warm_feature=None):
        _eval_dict = {
            self.root_feature: root_feature,  # test batch users embedding
            self.is_training: False
        }

        if self.type == 0:
            _eval_dict[self.warm_embedding] = warm_embedding
        elif self.type == 1:
            _eval_dict[self.warm_feature] = warm_feature
        elif self.type == 2:
            _eval_dict[self.warm_embedding] = warm_embedding
            _eval_dict[self.warm_feature] = warm_feature
        return _eval_dict

    def get_train_dict(self, root_feature, warm_embedding, target, warm_feature=None):
        _train_dict = {
            self.root_feature: root_feature,
            self.target: target,
            self.is_training: True,
        }

        if self.type == 0:
            _train_dict[self.warm_embedding] = warm_embedding
        elif self.type == 1:
            _train_dict[self.warm_feature] = warm_feature
        elif self.type == 2:
            _train_dict[self.warm_embedding] = warm_embedding
            _train_dict[self.warm_feature] = warm_feature

        return _train_dict

