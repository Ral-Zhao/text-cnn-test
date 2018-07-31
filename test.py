import tensorflow as tf
import os
from tensorflow.python.framework import graph_util


class MyTextCNN:
    def __init__(self,
                 seq_len,
                 num_classes,
                 vocab_size,
                 l2_reg_lambda=0.0,
                 learning_rate=0.001):
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.l2_reg_lambda = l2_reg_lambda
        self.learning_rate = learning_rate

    def build_model(self):
        self._init_placeholder()
        self._build_model()

    def _init_placeholder(self):
        self.input_x = tf.placeholder(tf.int32, [None, self.seq_len], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.global_step = tf.Variable(0, name="global_step", dtype=tf.int32, trainable=False)

    def _build_model(self):
        embedding_size = 128
        filter_sizes = [3, 4, 5]
        num_filters = 128

        with tf.variable_scope('embedding'):
            embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, embedding_size], -1.0, 1.0)
            )
            embeddings_char = tf.nn.embedding_lookup(embeddings, self.input_x)
            embeddings_char_expanded = tf.expand_dims(embeddings_char, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(
                    embeddings_char_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv'
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.seq_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool'
                )
                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        self.h_dropout = tf.nn.dropout(self.h_pool_flat, self.keep_prob)

        l2_loss = tf.constant(0.0)
        with tf.name_scope('output'):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_dropout, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores,
                labels=self.input_y
            )
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver(tf.global_variables())
        print('Built model!')

    def train(self, sess, X, Y, global_step, keep_prob=0.8):
        _, loss_val, accuracy = sess.run([self.train_op, self.loss, self.accuracy],
                                         feed_dict={
                                             self.input_x: X,
                                             self.input_y: Y,
                                             self.global_step: global_step,
                                             self.keep_prob: keep_prob
                                         })
        return loss_val, accuracy

    def predict(self, sess, X):
        rst = sess.run(self.predictions,
                       feed_dict={
                           self.input_x: X,
                           self.keep_prob: 1.0
                       })
        return rst

    def restore(self, sess, path):
        ckpt = tf.train.get_checkpoint_state(path)
        self.saver.restore(sess, ckpt.model_checkpoint_path)

    def save(self, sess, path, epoch):
        self.saver.save(sess, os.path.join(path, "basic"), global_step=epoch)

    def print_nodes(self):
        for name in [n.name for n in tf.get_default_graph().as_graph_def().node]:
            print(name)
        print('---------------')

    def save_pb(self, sess, path):
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output/predictions"])
        with tf.gfile.FastGFile(path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())
            print("Saved!")
