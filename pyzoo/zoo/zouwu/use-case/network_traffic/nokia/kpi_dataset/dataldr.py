import sys
import torch
from torch.utils.data import Dataset, DataLoader
import logging
import time
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

cell_num = 148

def preprocess_data(tx, ty, df, offset):
    trainx = df.iloc[offset:offset+tx, 2:3].values
    trainy = df.iloc[offset+tx:offset+tx+ty, 2:3].values
    return trainx, trainy


class AltranDataset(Dataset):
    """AltranDataset"""

    def __init__(self, df, mode="train", tx=96, ty=16):
        
        assert mode in ["train", "valid", "test"]

        self.mode = mode
        self.tx = tx
        self.ty = ty

        df = df[["Time", "Cell", "DL_PRB_Util_Percent"]]
        # df0 = df[df["Cell"]==0]
        # df1 = df[df["Cell"]==1]
        # df2 = df[df["Cell"]==2]
        # df3 = df[df["Cell"]==3]
        # df4 = df[df["Cell"]==4]
        # df5 = df[df["Cell"]==5]
        # df6 = df[df["Cell"]==6]
        # df7 = df[df["Cell"]==7]
        #
        # self.df = [df0, df1, df2, df3, df4, df5, df6, df7]

        self.df = []
        cells = df.Cell.unique()
        for cell in cells:
            tmp = df[df["Cell"] == cell]
            self.df.append(tmp)

        self.train_num = int(0.8*(len(self.df[0])-tx-ty+1))
        self.valid_num = int(0.1*(len(self.df[0])-tx-ty+1))
        self.test_num = (len(self.df[0]) - tx - ty + 1 - self.train_num - self.valid_num)

        self.train_num *= cell_num
        self.valid_num *= cell_num
        self.test_num *= cell_num

        print("---------------------------")
        print("Altran Dataset is created!")
        print("Dataset Overiew:")
        print("---------------------------")
        print("tx:", self.tx)
        print("ty:", self.ty)
        print("mode:", self.mode)
        if self.mode == "train":
            print("trian_num:", self.train_num) 
        if self.mode == "valid":
            print("valid_num:", self.valid_num)
        if self.mode == "test":
            print("test_num:", self.test_num)
        print("---------------------------")

    def __len__(self):
        if self.mode == "train":
            return self.train_num
        if self.mode == "valid":
            return self.valid_num
        if self.mode == "test":
            return self.test_num
        return None

    def __getitem__(self, idx):

        offset = 0
        if self.mode == "valid":
            offset = self.train_num//cell_num
        if self.mode == "test":
            offset = self.train_num//cell_num + self.valid_num//cell_num
        
        df_num = idx%cell_num
        idx = idx//cell_num
        
        x, y = preprocess_data(tx=self.tx, ty=self.ty, df=self.df[df_num], offset=idx+offset)
        x = torch.from_numpy(x).type(torch.float)
        y = torch.from_numpy(y).type(torch.float)
        return x, y

if __name__ == "__main__":
    # df = pd.read_csv('/home/ding/data/altran/stats_ue.csv')
    df = pd.read_csv("/home/ding/data/nokia/kpi_data_obfuscated.csv")
    dataset_train = AltranDataset(df, mode="train", tx=240)
    dataset_valid = AltranDataset(df, mode="valid", tx=240)
    train_loader = DataLoader(dataset_train, batch_size=512, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=1)
    for i, (x_batch,y_batch) in enumerate(train_loader):
        print("train x batch shape:", x_batch.shape)
        print("train y batch shape:", y_batch.shape)
        break
    for i, (x_valid_batch,y_valid_batch) in enumerate(valid_loader):
        print("valid x batch shape:", x_valid_batch.shape)
        print("valid y batch shape:", y_valid_batch.shape)
        break   