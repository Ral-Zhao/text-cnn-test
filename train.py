#! -*- coding:utf-8 -*-

import math

import tensorflow as tf

from data import DataSet
from model import MyTextCNN


def train():
    batch_size = 100
    seq_len = 128

    data_set = DataSet(batch_size, 'data_assistant/vocabs', seq_len)
    data_set.load_data('data_assistant/train/')

    graph = tf.Graph()
    with graph.as_default():
        model = MyTextCNN(seq_len, 2, data_set.get_vocab_size())
        model.build_model()

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()

        print('start!')

        epoch_batch_cnt = data_set.get_data_size() // batch_size
        print('batch_per_epoch={b}'.format(b=epoch_batch_cnt))

        total_step = 0
        for epoch in range(1000):
            print('epoch {e}'.format(e=epoch))

            for ii in range(epoch_batch_cnt + 1):
                X, Y = data_set.get_batch()

                loss_val, accuracy = model.train(sess, X, Y, total_step)

                if total_step % 2 == 0:
                    print('step {c}, loss={l}, accuracy={a}'.format(c=total_step, l=loss_val, a=accuracy))

                if math.isnan(loss_val):
                    print('Nan loss!!')
                    return

                if total_step % 100 == 0:
                    model.save(sess, "data_assistant/model/", total_step)
                    print('Saved!')

                total_step += 1

if __name__ == '__main__':
    train()
    # predict()
    # predict_pb()
