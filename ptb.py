import urllib.request
import os
import tarfile
import numpy as np
from collections import Counter

EOS = '<eos>'
UNK = '<unk>'
PAD = '<pad>'

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class PTB(object):
    def __init__(self, train_batch_size, test_batch_size, train_seq_len, test_seq_len):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_seq_len = train_seq_len
        self.test_seq_len = test_seq_len
        self.dictionary = Dictionary()
        self.dictionary.add_word(UNK)
         
        data_dir = '../data'
        self.ptb_dir = os.path.join(data_dir, 'ptb')
        self.train_inputs, self.train_targets = self.load_train_data()
        self.train_batch_count = self.train_inputs.shape[1] // self.train_seq_len
        self.test_inputs, self.test_targets = self.load_test_data('valid.txt')
        self.test_batch_count = self.test_inputs.shape[1] // self.test_seq_len
        self.vocabulary_size = len(self.dictionary) 

    def load_train_data(self):
        train_file = 'train.txt'
        inputs = []

        full_path = os.path.join(self.ptb_dir, train_file)
        with open(full_path, 'r') as f:
            for line in f:
                words = line.split() + [EOS]
                for word in words:
                    self.dictionary.add_word(word)
                    inputs.append(self.dictionary.word2idx[word])
            count = len(inputs) // self.train_batch_size
            inputs = inputs[:count * self.train_batch_size]
            targets = inputs[1:] + [self.dictionary.word2idx[EOS]]
        return np.array(inputs).astype(np.int64).reshape(self.train_batch_size, -1), np.array(targets).astype(np.int64).reshape(self.train_batch_size, -1)

    def load_test_data(self, test_file):
        inputs = []

        full_path = os.path.join(self.ptb_dir, test_file)
        with open(full_path, 'r') as f:
            for line in f:
                words = line.split() + [EOS]
                for word in words:
                    if word in self.dictionary.word2idx:
                        inputs.append(self.dictionary.word2idx[word])
                    else:
                        inputs.append(self.dictionary.word2idx[UNK])
            count = len(inputs) // self.test_batch_size
            inputs = inputs[:count * self.test_batch_size]
            targets = inputs[1:] + [self.dictionary.word2idx[EOS]]
        return np.array(inputs).astype(np.int64).reshape(self.test_batch_size, -1), np.array(targets).astype(np.int64).reshape(self.test_batch_size, -1)

    def next_train_batch(self, idx):
        batch_inputs = self.train_inputs[:, idx * self.train_seq_len : (idx + 1) * self.train_seq_len]
        batch_targets = self.train_targets[:, idx * self.train_seq_len : (idx + 1) * self.train_seq_len]
        return batch_inputs, batch_targets

    def next_test_batch(self, idx):
        batch_inputs = self.test_inputs[:, idx * self.test_seq_len : (idx + 1) * self.test_seq_len]
        batch_targets = self.test_targets[:, idx * self.test_seq_len : (idx + 1) * self.test_seq_len]
        return batch_inputs, batch_targets




#ptb = PTB(10, 10, 5, 5)
#print(ptb.next_train_batch(0))
