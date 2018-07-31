#! -*- coding:utf-8 -*-

import sys

# reload(sys)
# sys.setdefaultencoding('utf-8')
# print(sys.getdefaultencoding())

import tensorflow as tf
from data import DataSet
from model import MyTextCNN


def _predict(data_set, sess, sentence, input_x, keep_prob, predict_op,vec):
    X = data_set.build_predict_data(sentence)
    predict_rst = sess.run(predict_op,
                           feed_dict={
                               input_x: X,
                               keep_prob: 1.0
                           })
    correct_prediction = tf.equal(predict_rst, tf.argmax(vec,1))
    print(sess.run(correct_prediction))

def predict():
    seq_len = 128
    data_set = DataSet(10, 'data_assistant/vocabs', seq_len)

    with tf.Graph().as_default():
        with open('data_assistant/output/cnn.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)

        with tf.Session() as sess:
            input_x = sess.graph.get_tensor_by_name("import/input_x:0")
            keep_prob = sess.graph.get_tensor_by_name("import/keep_prob:0")
            predict_op = sess.graph.get_tensor_by_name("import/output/predictions:0")

            _predict(data_set, sess, u'p2是华为', input_x, keep_prob, predict_op,[[0, 1]])
            _predict(data_set, sess, u'儿歌播放', input_x, keep_prob, predict_op,[[1, 0]])

if __name__ == '__main__':
    predict()
