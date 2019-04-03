# -*- coding: utf-8 -*-

import re
import os
import sys
import numpy as np
import pickle

from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, x, y):
        self.data = list(zip(x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        assert idx < len(self)
        return self.data[idx]


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def extract_vocab(positive_data_file, negative_data_file):
    '''
    extract vocab from txt
    '''
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = list(map(lambda x: x.split(), x_text))

    vocab = []
    for line in x_text:
        vocab.extend(line)

    vocab = list(set(vocab))
    print("vocab size: {}.".format(len(vocab)))
    open("./data/vocab.txt", "w").write("\n".join(vocab))


def get_glove(w2v_path, vocab_path):

    vocab = {j.strip(): i for i, j in enumerate(open(vocab_path), 0)}
    id2word = {vocab[i]: i for i in vocab}

    dim = 0
    w2v = {}
    for line in open(w2v_path):
        line = line.strip().split()
        word = line[0]
        vec = list(map(float, line[1:]))
        dim = len(vec)
        w2v[word] = vec

    vecs = []
    vecs.append(np.random.uniform(low=-1.0, high=1.0, size=dim))

    hit = 0
    for i in range(1, len(vocab) - 1):
        if id2word[i] in w2v:
            hit += 1
            vecs.append(w2v[id2word[i]])
        else:
            vecs.append(vecs[0])
    vecs.append(np.zeros(dim))
    assert(len(vecs) == len(vocab))

    print("vocab size: {}, dim: {}; hit in glove:{}".format(len(vocab), dim, hit))
    np.save("./data/glove/glove_{}d.npy".format(dim), np.array(vecs, dtype=np.float32))
    np.save("./data/glove/word2id.npy", vocab)
    np.save("./data/glove/id2word.npy", id2word)


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = list(map(lambda x: x.split(), x_text))
    # Generate labels
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    y = np.array(positive_labels + negative_labels)
    return [x_text, y]


if __name__ == "__main__":
    import fire
    fire.Fire()
