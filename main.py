# -*- coding: utf-8 -*-

import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

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
    device = torch.device("cuda:{}".format(opt.gpu_id) if torch.cuda.is_available() else "cpu")
    opt.device = device

    x_text, y = load_data_and_labels("./data/rt-polarity.pos", "./data/rt-polarity.neg")
    x_train, x_test, y_train, y_test = train_test_split(x_text, y, test_size=opt.test_size)

    train_data = Data(x_train, y_train)
    test_data = Data(x_test, y_test)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"{now()} train data: {len(train_data)}, test data: {len(test_data)}")

    model = Model(opt)
    print(f"{now()} {opt.emb_method} init model finished")

    if opt.use_gpu:
        model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    lr_sheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    best_acc = -0.1
    best_epoch = -1
    for epoch in range(opt.epochs):
        total_loss = 0.0
        model.train()
        for step, batch_data in enumerate(train_loader):
            x, labels = batch_data
            labels = torch.LongTensor(labels)
            if opt.use_gpu:
                labels = labels.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        acc = test(model, test_loader)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
        print(f"{now()} Epoch{epoch}: loss: {total_loss}, test_acc: {acc}")
        lr_sheduler.step()

    print("*"*20)
    print(f"{now()} finished; epoch {best_epoch} best_acc: {best_acc}")


def test(model, test_loader):
    correct = 0
    num = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            x, labels = data
            num += len(labels)
            output = model(x)
            labels = torch.LongTensor(labels)
            if opt.use_gpu:
                output = output.cpu()
            predict = torch.max(output.data, 1)[1]
            correct += (predict == labels).sum().item()
        return correct * 1.0 / num


if __name__ == "__main__":
    import fire
    fire.Fire()
