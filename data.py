#! -*- coding:utf-8 -*-

import numpy as np
import os


def process_data(input_dir, output_path):
    file_list = []
    for data_file in sorted(os.listdir(input_dir)):
        full_path_name = os.path.join(input_dir, data_file)
        if os.path.isfile(full_path_name) and data_file.lower().endswith('.txt'):
            file_list.append(full_path_name)

    print('file_cnt=' + str(len(file_list)))
    cnt = 0
    with open(output_path, 'w') as output_f:
        for file in file_list:
            with open(file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    if line.startswith('补充点评'):
                        try:
                            idx = line.index('：')
                            line = line[idx + 1:].strip()
                        except:
                            continue

                    output_f.write(line + "\n")
                    cnt += 1
    print('line: ' + str(cnt))


def split(data_path, train_path, test_path, test_ratio):
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(line)

    total_cnt = len(data)
    test_cnt = int(total_cnt * test_ratio)
    train_cnt = total_cnt - test_cnt
    print(str(total_cnt) + ' ' + str(train_cnt) + " " + str(test_cnt))
    with open(train_path, 'w') as f:
        for i in range(train_cnt):
            f.write(data[i] + '\n')
    with open(test_path, 'w') as f:
        for i in range(train_cnt, total_cnt):
            f.write(data[i] + '\n')
    print('Done')


def get_vocabs(file_list, output_path):
    wc_dict = {}
    for file in file_list:
        with open(file) as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                for ch in line:
                    if ch == ' ' or ch == '':
                        continue
                    if ch not in wc_dict:
                        wc_dict[ch] = 1
                    else:
                        wc_dict[ch] += 1

    word_cnt_list = sorted(wc_dict.items(), key=lambda x: -x[1])

    output_f = open(output_path, 'w')
    cnt = 0
    for w, c in word_cnt_list:
        if c < 2:
            continue
        output_f.write(w + '\n')
        cnt += 1
    print(cnt)


class DataSet():
    PAD_ID = 0
    PAD = '<PAD>'
    UNK_ID = 1
    UNK = '<UNK>'

    def __init__(self, batch_size, vocab_path, max_len=128):
        self.batch_size = batch_size
        self.build_dict(vocab_path)
        self.data_index = 0
        self.data_len = 0
        self.max_len = max_len

    def get_data_size(self):
        return self.data_len

    def get_vocab_size(self):
        return len(self.dictionary)

    def get_char(self, idx):
        if idx not in self.reverse_dictionary:
            return self.UNK
        else:
            return self.reverse_dictionary[idx]

    def get_idx(self, ch):
        if ch not in self.dictionary:
            return self.UNK_ID
        else:
            return self.dictionary[ch]

    def build_predict_data(self, sentence):

        x = [self.dictionary[ch] if ch in self.dictionary else self.UNK_ID for ch in sentence]
        size = len(x)
        if size <= self.max_len:
            x = x + [0] * (self.max_len - size)
        else:
            x = x[:self.max_len]

        return [x]

    def convert_to_string(self, X):
        return ''.join([self.reverse_dictionary[x] if x in self.reverse_dictionary else self.UNK for x in X])

    def _load_data(self, path):
        X = []
        with open(path, 'r',encoding='UTF-8') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                x = [self.get_idx(ch) for ch in line]
                size = len(x)
                if size <= self.max_len:
                    x = x + [0] * (self.max_len - size)
                else:
                    x = x[:self.max_len]
                X.append(x)
        return X

    def load_data(self, data_path):

        pos_file_path = data_path + '/assistant/data.txt'
        neg_file_path = data_path + '/non_assistant/data.txt'
        pos_data = self._load_data(pos_file_path)
        neg_data = self._load_data(neg_file_path)
        X = pos_data + neg_data

        pos_label = [[0, 1] for _ in range(len(pos_data))]
        neg_label = [[1, 0] for _ in range(len(neg_data))]
        Y = pos_label + neg_label

        shuffle_indices = np.random.permutation(range(len(X)))
        self.X = []
        self.Y = []
        for idx in shuffle_indices:
            self.X.append(X[idx])
            self.Y.append(Y[idx])

        self.data_len = len(X)
        print('Loaded data!')

    def get_batch(self):

        XX = []
        YY = []
        for i in range(self.batch_size):
            XX.append(self.X[self.data_index][:])
            YY.append(self.Y[self.data_index][:])
            self.data_index = (self.data_index + 1) % self.data_len

        return XX, YY

    def build_dict(self, vocab_path):
        self.dictionary = {}
        self.reverse_dictionary = {}
        with open(vocab_path, 'r',encoding='UTF-8') as f:
            cnt = 0
            for line in f:
                # line = unicode(line)
                word = line.strip('\n')


                self.dictionary[word] = cnt
                self.reverse_dictionary[cnt] = word
                cnt += 1

        # print(self.dictionary[u'的'])  # 3
        print(self.reverse_dictionary[3])


if __name__ == '__main__':
    # process_data('data_hotel/data/positive', 'data_hotel/train/positive/data.txt')
    # process_data('data_hotel/data/negative', 'data_hotel/train/negative/data.txt')

    # split('data_hotel/data/positive/data.txt', 'data_hotel/train/positive/data.txt',
    #       'data_hotel/test/positive/data.txt', 0.05)
    # split('data_hotel/data/negative/data.txt', 'data_hotel/train/negative/data.txt',
    #       'data_hotel/test/negative/data.txt', 0.05)

    # file_list = [
    #     'data_hotel/data/positive/data.txt',
    #     'data_hotel/data/negative/data.txt'
    # ]
    # get_vocabs(file_list, 'data_hotel/vocabs')


    data_set = DataSet(2, 'data_assistant/vocabs')
    data_set.load_data('data_assistant/train')
    # data_set.build_dict()
    # data_set.process()
    X, Y = data_set.get_batch()
    for x, y in zip(X, Y):
        print(data_set.convert_to_string(x))
        print(y)

    X, Y = data_set.get_batch()
    for x, y in zip(X, Y):
        print(data_set.convert_to_string(x))
        print(y)
