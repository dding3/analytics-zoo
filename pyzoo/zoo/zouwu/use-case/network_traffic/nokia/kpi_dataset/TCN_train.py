import sys
sys.path.append("../model")
sys.path.append(".")
from dataldr import AltranDataset
from torch.utils.data import DataLoader
# from modtcn_model import Others
from tcn_model import TemporalConvNet
import torch
import numpy as np
import time
import os
import pandas as pd
import logging

import argparse
parser = argparse.ArgumentParser()

# General configs

# model configs
parser.add_argument("--dropout",
                    default=0.1,
                    type=float,
                    help="Dropout rate (1 - keep probability)")
parser.add_argument("--hidden_channels",
                    default=48,
                    type=int,
                    help="Hidden size for TCNN encoder")
parser.add_argument("--input_time_step",
                    default=96,
                    type=int,
                    help="Input time step length(tx)")
parser.add_argument("--output_time_step",
                    default=16,
                    type=int,
                    help="Output time step length(ty)")
parser.add_argument("--kernal_size",
                    default=3,
                    type=int,
                    help="Kernal size for TCNN layers")
parser.add_argument("--dilation",
                    default=[2,4,8,16,32],
                    nargs='+',
                    type=int,
                    help="(Only for Modified TCN) Diliation list for TCNN layers, the length of this argument will be the number of layer")
parser.add_argument("--level",
                    default=8,
                    type=int,
                    help="(Only for TCN) Choose level for TCN model")
parser.add_argument("--model",
                    default="TCN",
                    type=str,
                    help="Choose between TCN and ModTCN")

# training configs
parser.add_argument("--train_batch_size",
                    default=512,
                    type=int,
                    help="Train batch size")
parser.add_argument("--lr",
                    default=0.001,
                    type=float,
                    help="Learning rate")
parser.add_argument("--epoch",
                    default=30,
                    type=int,
                    help="Epoch number you want to train")

# data
parser.add_argument("--data_path",
                    default="/home/ding/data/nokia/kpi_data_obfuscated.csv",
                    type=str,
                    help="Data csv you want to load")

args = parser.parse_args()

# echo argument
print("--------------------")
print("Model config")
print("--------------------")
print("dropout:", args.dropout)
print("hidden_channels:", args.hidden_channels)
print("input_time_step:", args.input_time_step)
print("output_time_step:", args.output_time_step)
print("kernal_size:", args.kernal_size)
print("dilation:", args.dilation)
print("level:", args.level)
print("model type:", args.model)
print("--------------------")

print("--------------------")
print("Training config")
print("--------------------")
print("train_batch_size:", args.train_batch_size)
print("learning rate:", args.lr)
print("epoch number:", args.epoch)
print("--------------------")


# model
loss_fn = torch.nn.MSELoss()
if args.model.lower() == "modtcn":
    model = Others(dropout=args.dropout,
                    features=2,
                    hidden_channels=args.hidden_channels,
                    input_time_step=args.input_time_step,
                    output_time_step=args.output_time_step,
                    kernal_size=args.kernal_size,
                    dilation=args.dilation)
else:
    model = TemporalConvNet(past_seq_len=args.input_time_step,
                        feature_num=1,
                        future_seq_len=args.output_time_step, 
                        num_channels=[args.hidden_channels]*(args.level-1)+[1],
                        dropout=args.dropout,
                        kernel_size=args.kernal_size
                        )
print(model)

# data
df = pd.read_csv(args.data_path)
dataset_train = AltranDataset(df, mode="train", tx=args.input_time_step, ty=args.output_time_step)
dataset_valid = AltranDataset(df, mode="valid", tx=args.input_time_step, ty=args.output_time_step)
train_loader = DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=4) # num_workers can be set higher or lower, experimentally it is set to 4
valid_loader = DataLoader(dataset_valid, batch_size=1, num_workers=4)
total_step = len(train_loader)
print(total_step)

# hyper parameter
learning_rate = args.lr
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epoch_num = args.epoch
dir_checkpoint = 'checkpoints/'

# train and validation
for epoch in range(0, epoch_num):

    # train
    model.train()
    for i, (x_batch,y_batch) in enumerate(train_loader):
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)

        if (i+1) % 20 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                .format(epoch, epoch_num, i+1, total_step, loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 1 == 0:
        try:
            os.mkdir(dir_checkpoint)
            logging.info('Created checkpoint directory')
        except OSError:
            pass
        torch.save(model.state_dict(),dir_checkpoint + f'Others_lookback_{args.input_time_step}_hidden_{args.hidden_channels}_model_{args.model}_time_{time.time()}_epoch_{epoch}.pth')
        logging.info(f'Checkpoint {epoch} saved !')

    # evaluation on valid
    model.eval()
    loss_sum = 0
    for i, (x_valid_batch, y_valid_batch) in enumerate(valid_loader):
        y_pred = model(x_valid_batch)
        loss = loss_fn(y_pred, y_valid_batch)
        loss_sum += loss.item()
    print ("Evaluate Epoch [{}/{}], Loss: {:.4f}"
                .format(epoch, epoch_num, loss_sum/len(valid_loader)))
    print ("Evaluate Epoch [{}/{}], RMSE Loss: {:.4f}"
                .format(epoch, epoch_num, (loss_sum/len(valid_loader))**0.5))