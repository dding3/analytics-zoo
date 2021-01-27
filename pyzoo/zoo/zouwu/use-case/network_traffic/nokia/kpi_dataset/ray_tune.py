import sys
sys.path.append(".")
import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import pandas as pd

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler, FIFOScheduler

from dataldr import AltranDataset

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, past_seq_len, feature_num, future_seq_len, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = feature_num if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                     stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(past_seq_len, future_seq_len)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = x.permute(0,2,1)
        y1 = self.tcn(x)
        y = self.linear(y1)
        y = y.permute(0,2,1)
        return y

output_time_step = 16

def train(model, optimizer, train_loader, device=torch.device("cpu")):
    print("Start training this model")
    print(model)
    loss_fn = torch.nn.MSELoss()
    model.train()
    total_step = len(train_loader)
    for epoch in range(0, EPOCH_NUM):
        for i, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device),  y_batch.to(device)
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            if (i+1) % 20 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                    .format(epoch, EPOCH_NUM, i+1, total_step, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def test(model, data_loader, device=torch.device("cpu")):
    model.eval()
    loss_fn = torch.nn.MSELoss()
    loss_sum = 0
    with torch.no_grad():
        for i, (x_valid_batch, y_valid_batch) in enumerate(data_loader):
            x_valid_batch, y_valid_batch = x_valid_batch.to(device),  y_valid_batch.to(device)
            y_pred = model(x_valid_batch)
            loss = loss_fn(y_pred, y_valid_batch)
            loss_sum += loss.item()
    return (loss_sum/len(data_loader))**0.5

def get_data_loaders(tx):
    df = pd.read_csv("/root/dingding/nokia/kpi_data/kpi_data_obfuscated.csv")
    dataset_train = AltranDataset(df, mode="train", tx=tx)
    dataset_valid = AltranDataset(df, mode="valid", tx=tx)
    train_loader = DataLoader(dataset_train, batch_size=512, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset_valid, batch_size=1, num_workers=4)
    return train_loader, valid_loader

def train_altran(config):
    train_loader, test_loader = get_data_loaders(config["lookback"])
    # model = Others(dropout=config["dropout"],
    #              features=2,
    #              hidden_channels=config["hidden_size"],
    #              input_time_step=config["lookback"],
    #              output_time_step=40,
    #              kernal_size=config["kernal_size"],
    #              dilation=config["dilation"])

    model = TemporalConvNet(past_seq_len=config["lookback"],
                        feature_num=1,
                        future_seq_len=output_time_step,
                        num_channels=[config["hidden_size"]]*(8-1)+[1],
                        dropout=config["dropout"],
                        kernel_size=config["kernal_size"],
                        )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"])

    while True:
        train(model, optimizer, train_loader)
        rmse = test(model, test_loader)
        tune.report(RMSE=rmse)

if __name__ == "__main__":
    ray.init(num_cpus=80, local_mode=True)
    EPOCH_NUM = 1

    # for early stopping
    # sched = AsyncHyperBandScheduler()
    sched = FIFOScheduler()
    smoke_test = False

    analysis = tune.run(
        train_altran,
        name="TCN_search_lookback_time_{}".format(time.time()),
        mode="min",
        metric="RMSE",
        scheduler=sched,
        stop={
            "training_iteration": 4 if smoke_test else 21
        },
        resources_per_trial={
            "cpu": 8
        },
        num_samples= 1 if smoke_test else 1,
        config={
            "lr": tune.choice([0.001, 0.003]),
            "dropout": tune.choice([0]),
            "hidden_size": tune.choice([8]),
            "kernal_size": tune.choice([3]),
            "lookback": tune.grid_search([80]),
        } if smoke_test else {
            "lr": tune.grid_search([0.001, 0.003]),
            "dropout": tune.grid_search([0, 0.1]),
            "hidden_size": tune.grid_search([48, 96, 128, 192, 256]),
            "kernal_size": tune.grid_search([3]),
            "lookback": tune.grid_search([48, 80, 96, 160, 192])
        })
    print("Best config is:", analysis.best_config)
