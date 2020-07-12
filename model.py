# -*- coding: utf-8 -*-

from allennlp.modules.elmo import Elmo, batch_to_ids
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy


from encoder import Encoder


class Model(pl.LightningModule):
    '''
    '''
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.net = Net(opt)

    def forward(self, x):
        return self.net(x, self.device)

    def training_step(self, batch, batch_id):
        self.opt.device = self.device
        x, labels = batch
        labels = torch.LongTensor(labels).to(self.device)
        output = self(x)
        loss = F.cross_entropy(output, labels)
        return {"loss": loss}

    def validation_step(self, batch, batch_id):
        x, labels = batch
        labels = torch.LongTensor(labels).to(self.device)
        output = self(x)
        loss = F.cross_entropy(output, labels)
        val_acc = accuracy(output, labels)
        return {"loss": loss, 'val_acc': val_acc}

    def validation_epoch_end(self, outputs):
        val_loss = 0.
        val_acc = 0.
        for o in outputs:
            val_loss += o['loss']
            val_acc += o['val_acc']

        val_loss /= len(outputs)
        val_acc /= len(outputs)
        tqdm_dict = {"val_loss": val_loss, "val_acc": val_acc}
        return {"progress_bar": tqdm_dict}

    def test_step(self, batch, batch_id):
        x, labels = batch
        labels = torch.LongTensor(labels).to(self.device)
        output = self(x)
        loss = F.cross_entropy(output, labels)
        val_acc = accuracy(output, labels)
        return {"loss": loss, 'test_acc': val_acc}

    def test_epoch_end(self, outputs):
        test_loss = 0.
        test_acc = 0.
        for o in outputs:
            test_loss += o['loss']
            test_acc += o['test_acc']

        test_loss /= len(outputs)
        test_acc /= len(outputs)
        tqdm_dict = {"test_loss": test_loss, "test_acc": test_acc}
        return {"progress_bar": tqdm_dict}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.opt.lr)


class Net(nn.Module):
    def __init__(self, opt):

        super().__init__()
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

    def forward(self, x, device):
        self.device = device
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
        self.glove.weight.data.copy_(emb)
        self.word_dim = self.opt.glove_dim

    def get_bert(self, sentence_lists):
        '''
        get the ELMo word embedding vectors for a sentences
        '''
        sentence_lists = [' '.join(x) for x in sentence_lists]
        ids = self.bert_tokenizer(sentence_lists, padding=True, return_tensors="pt")
        inputs = ids['input_ids'].to(self.device)
        embeddings = self.bert(inputs)
        return embeddings[0]

    def get_elmo(self, sentence_lists):
        '''
        get the ELMo word embedding vectors for a sentences
        '''
        character_ids = batch_to_ids(sentence_lists).to(self.device)
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
        sentence_lists = sentence_lists.to(self.device)
        embeddings = self.glove(sentence_lists)

        return embeddings
