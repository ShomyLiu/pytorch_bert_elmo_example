# -*- coding: utf-8 -*-

from allennlp.modules.elmo import Elmo, batch_to_ids
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np

from encoder import Encoder


class Model(nn.Module):
    def __init__(self, opt):

        super(Model, self).__init__()
        self.opt = opt
        self.use_gpu = self.opt.use_gpu

        if opt.emb_method == 'elmo':
            self.init_elmo()
        elif self.opt.emb_method == 'glove':
            self.init_glove()
        elif self.opt.emb_method == 'bert':
            self.init_bert()

        self.encoder = Encoder(opt.enc_method, self.word_dim, opt.hidden_size, opt.out_size)
        self.cls = nn.Linear(opt.out_size, opt.num_labels)
        nn.init.uniform_(self.cls.weight, -0.1, 0.1)
        nn.init.uniform_(self.cls.bias, -0.1, 0.1)
        self.dropout = nn.Dropout(self.opt.dropout)

    def forward(self, x):
        if self.opt.emb_method == 'elmo':
            word_embs = self.get_elmo(x)
        elif self.opt.emb_method == 'glove':
            word_embs = self.get_glove(x)
        elif self.opt.emb_method == 'bert':
            word_embs = self.get_bert(x)

        x = self.encoder(word_embs)
        x = self.dropout(x)
        x = self.cls(x)    # batch_size * num_label
        return x

    def init_bert(self):
        '''
        initilize the Bert model
        '''
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.opt.bert_path)
        self.bert = AutoModel.from_pretrained(self.opt.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.word_dim = self.opt.bert_dim

    def init_elmo(self):
        '''
        initilize the ELMo model
        '''
        self.elmo = Elmo(self.opt.elmo_options_file, self.opt.elmo_weight_file, 1)
        for param in self.elmo.parameters():
            param.requires_grad = False
        self.word_dim = self.opt.elmo_dim

    def init_glove(self):
        '''
        load the GloVe model
        '''
        self.word2id = np.load(self.opt.word2id_file, allow_pickle=True).tolist()
        self.glove = nn.Embedding(self.opt.vocab_size, self.opt.glove_dim)
        emb = torch.from_numpy(np.load(self.opt.glove_file, allow_pickle=True))
        if self.use_gpu:
            emb = emb.to(self.opt.device)
        self.glove.weight.data.copy_(emb)
        self.word_dim = self.opt.glove_dim

    def get_bert(self, sentence_lists):
        '''
        get the ELMo word embedding vectors for a sentences
        '''
        sentence_lists = [' '.join(x) for x in sentence_lists]
        ids = self.bert_tokenizer(sentence_lists, padding=True, return_tensors="pt")
        inputs = ids['input_ids']
        if self.opt.use_gpu:
            inputs = inputs.to(self.opt.device)

        embeddings = self.bert(inputs)
        return embeddings[0]

    def get_elmo(self, sentence_lists):
        '''
        get the ELMo word embedding vectors for a sentences
        '''
        character_ids = batch_to_ids(sentence_lists)
        if self.opt.use_gpu:
            character_ids = character_ids.to(self.opt.device)
        embeddings = self.elmo(character_ids)
        return embeddings['elmo_representations'][0]

    def get_glove(self, sentence_lists):
        '''
        get the glove word embedding vectors for a sentences
        '''
        max_len = max(map(lambda x: len(x), sentence_lists))
        sentence_lists = list(map(lambda x: list(map(lambda w: self.word2id.get(w, 0), x)), sentence_lists))
        sentence_lists = list(map(lambda x: x + [self.opt.vocab_size-1] * (max_len - len(x)), sentence_lists))
        sentence_lists = torch.LongTensor(sentence_lists)
        if self.use_gpu:
            sentence_lists = sentence_lists.to(self.opt.device)
        embeddings = self.glove(sentence_lists)

        return embeddings

