import io
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch.onnx
from torch.utils.data import DataLoader

import sys
sys.path.append("../model")
from tcn_model import TemporalConvNet
sys.path.append("../data")
from dataldr import AltranDataset

import time

import argparse
parser = argparse.ArgumentParser()

# General configs

# model configs
parser.add_argument("--dropout",
                    default=0.25,
                    type=float,
                    help="Dropout rate (1 - keep probability)")
parser.add_argument("--hidden_channels",
                    # default=60,
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
parser.add_argument("--model_path",
                    type=str,
                    # default="/home/ding/proj/analytics-zoo/pyzoo/zoo/zouwu/use-case/network_traffic/nokia/kpi_dataset/checkpoints/Others_lookback_96_hidden_48_model_TCN_time_1610681364.5952737_epoch_23.pth",
                    default="/home/ding/proj/analytics-zoo/pyzoo/zoo/zouwu/use-case/network_traffic/nokia/kpi_dataset/checkpoints/Others_lookback_96_hidden_48_model_TCN_time_1610668915.0310676_epoch_4.pth",
                    help="Model you want to load")

# data
parser.add_argument("--data_path",
                    type=str,
                    default="/home/ding/data/nokia/kpi_data_obfuscated.csv",
                    help="Data csv you want to load")

args = parser.parse_args()

# test RMSE by pytorch
loss_fn = torch.nn.MSELoss()
# if args.model.lower() == "modtcn":
#     model = Others(dropout=args.dropout,
#                     features=1,
#                     hidden_channels=args.hidden_channels,
#                     input_time_step=args.input_time_step,
#                     output_time_step=args.output_time_step,
#                     kernal_size=args.kernal_size,
#                     dilation=args.dilation)
# else:
#     model = TemporalConvNet(past_seq_len=args.input_time_step,
#                         feature_num=1,
#                         future_seq_len=args.output_time_step,
#                         num_channels=[args.hidden_channels]*(args.level-1)+[1],
#                         dropout=args.dropout,
#                         kernel_size=args.kernal_size
#                         )
# print(model)

# data
df = pd.read_csv(args.data_path)
dataset_test = AltranDataset(df, mode="test", tx=args.input_time_step, ty=args.output_time_step)
print(len(dataset_test))
test_loader = DataLoader(dataset_test, batch_size=1, num_workers=4)

from os import listdir
from os.path import isfile, join
path = "/home/ding/proj/analytics-zoo/pyzoo/zoo/zouwu/use-case/network_traffic/nokia/kpi_dataset/checkpoints/"
model_paths = [path+f for f in listdir(path) if isfile(join(path, f))]

for model_path in model_paths:
    model = TemporalConvNet(past_seq_len=args.input_time_step,
                        feature_num=1,
                        future_seq_len=args.output_time_step,
                        num_channels=[args.hidden_channels]*(args.level-1)+[1],
                        dropout=args.dropout,
                        kernel_size=args.kernal_size
                        )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Export the model
    x = torch.randn(148, args.input_time_step, 1)
    torch.onnx.export(model,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      "cache.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                    'output' : {0 : 'batch_size'}})

    from zoo.automl.common.metrics import *
    mae_sum = 0
    mape_sum = 0
    for i, (x_test_batch, y_test_batch) in enumerate(test_loader):
        y_pred = model(x_test_batch)
        # for i in range(y_pred.shape[0]):
            # mse = mean_squared_error(y_test_batch.detach().numpy()[i], y_pred.detach().numpy()[i])
            # loss_sum += mse

        for i in range(y_pred.shape[0]):
            mae = MAE(y_test_batch.detach().numpy()[i], y_pred.detach().numpy()[i])
            mape = MAPE(y_test_batch.detach().numpy()[i], y_pred.detach().numpy()[i], multioutput='uniform_average')
            mae_sum += mae
            mape_sum += mape

    mae_sum = mae_sum / y_pred.shape[0]
    print ("path:", model_path)
    print ("Evaluate Loss MAE: {:.4f}"
           .format(mae_sum / len(test_loader)))
    mape_sum = mape_sum / y_pred.shape[0]
    print ("Evaluate Loss MAPE: {:.4f}"
           .format(mape_sum / len(test_loader)))


# test through Onnx
# import onnx
# onnx_model = onnx.load("cache.onnx")
# onnx.checker.check_model(onnx_model)
#
# import onnxruntime
# ort_session = onnxruntime.InferenceSession("cache.onnx")
#
# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
#
# # compute ONNX Runtime output prediction
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
# for i in range(100):
#     start_time = time.time()
#     ort_outs = ort_session.run(None, ort_inputs)
#     print("Onnx time test {}/100 (ms):".format(i), (time.time()-start_time)*1000)
