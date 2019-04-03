# -*- coding: utf-8 -*-

from allennlp.modules.elmo import Elmo, batch_to_ids
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class TextCNN(nn.Module):
    def __init__(self, opt):

        super(TextCNN, self).__init__()
        self.opt = opt

        if self.opt.emb_method == 'elmo':
            self.init_elmo()
        elif self.opt.emb_method == 'glove':
            self.init_glove()
        elif self.opt.emb_method == 'elmo_glove':
            self.init_elmo()
            self.init_glove()
            self.word_dim = self.opt.elmo_dim + self.opt.glove_dim

        self.cnns = nn.ModuleList([nn.Conv2d(1, self.opt.num_filters, (i, self.word_dim)) for i in self.opt.k])
        self.linear = nn.Linear(self.opt.num_filters * len(self.opt.k), self.opt.num_labels)
        self.dropout = nn.Dropout(self.opt.dropout)
        self.use_gpu = self.opt.use_gpu

    def init_elmo(self):
        self.elmo = Elmo(self.opt.elmo_options_file, self.opt.elmo_weight_file, 1)
        self.word_dim = self.opt.elmo_dim

    def init_glove(self):
        self.word2id = np.load(self.opt.word2id_file).tolist()
        self.glove = nn.Embedding(self.opt.vocab_size, self.opt.glove_dim)
        emb = torch.from_numpy(np.load(self.opt.glove_file))
        emb = emb.to(self.opt.device)
        self.glove.weight.data.copy_(emb)
        self.word_dim = self.opt.glove_dim

    def get_elmo(self, sentence_lists):
        character_ids = batch_to_ids(sentence_lists)
        if self.opt.use_gpu:
            character_ids = character_ids.to(self.opt.device)
        embeddings = self.elmo(character_ids)
        return embeddings['elmo_representations'][0]

    def get_glove(self, sentence_lists):
        # __import__('ipdb').set_trace()
        max_len = max(map(lambda x: len(x), sentence_lists))
        # max_len = 120
        sentence_lists = list(map(lambda x: list(map(lambda w: self.word2id.get(w, 0), x)), sentence_lists))
        sentence_lists = list(map(lambda x: x + [self.opt.vocab_size-1] * (max_len - len(x)), sentence_lists))
        sentence_lists = torch.LongTensor(sentence_lists)
        if self.use_gpu:
            sentence_lists = sentence_lists.to(self.opt.device)
        embeddings = self.glove(sentence_lists)

        return embeddings

    def forward(self, x):
        if self.opt.emb_method == 'elmo':
            word_embs = self.get_elmo(x)
        elif self.opt.emb_method == 'glove':
            word_embs = self.get_glove(x)
        elif self.opt.emb_method == 'elmo_glove':
            glove = self.get_glove(x)
            elmo = self.get_elmo()
            word_embs = torch.cat([elmo, glove], -1)

        x = word_embs.unsqueeze(1)
        x = [F.relu(cnn(x)).squeeze(3) for cnn in self.cnns]   # batch_size * num_filter * (max_length-3+1)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.linear(x)    # batch_size * num_label
        return x
