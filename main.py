# -*- coding: utf-8 -*-

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data_pro import load_data_and_labels, Data
from model import Model
from config import opt


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def collate_fn(batch):
    data, label = zip(*batch)
    return data, label


def train(**kwargs):

    opt.parse(kwargs)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    x_text, y = load_data_and_labels("./data/rt-polarity.pos", "./data/rt-polarity.neg")
    x_train, x_test, y_train, y_test = train_test_split(x_text, y, test_size=opt.test_size)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=opt.test_size)

    train_data = Data(x_train, y_train)
    test_data = Data(x_test, y_test)
    val_data = Data(x_test, y_test)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    print(f"{now()} train data: {len(train_data)}, test data: {len(test_data)}")

    model = Model(opt)
    print(f"{now()} {opt.emb_method} init model finished")

    trainer = pl.Trainer(gpus=opt.gpu_id, distributed_backend='ddp',
                         max_epochs=opt.epochs, default_root_dir='./checkpoints/')
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    import fire
    fire.Fire()
