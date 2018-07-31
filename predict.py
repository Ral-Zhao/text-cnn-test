#! -*- coding:utf-8 -*-

# reload(sys)
# sys.setdefaultencoding('utf-8')
# print(sys.getdefaultencoding())

import tensorflow as tf

from data import DataSet
from model import MyTextCNN


def _predict(data_set, model, sess, sentence):

    X = data_set.build_predict_data(sentence)
    print(X)
    predict_rst = model.predict(sess, X)
    print(predict_rst)


def predict():
    seq_len = 128
    data_set = DataSet(10, 'data_assistant/vocabs', seq_len)

    graph = tf.Graph()
    with graph.as_default():
        model = MyTextCNN(seq_len, 2, data_set.get_vocab_size())
        model.build_model()
        model.print_nodes()

    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)

    with tf.Session(graph=graph, config=session_conf) as sess:
        model.restore(sess, 'data_assistant/model')
        model.save_pb(sess, 'data_assistant/output/cnn.pb')

        _predict(data_set, model, sess, u'有没有手机卖？')
        _predict(data_set, model, sess, u'干什么呢')

if __name__ == '__main__':
    predict()
