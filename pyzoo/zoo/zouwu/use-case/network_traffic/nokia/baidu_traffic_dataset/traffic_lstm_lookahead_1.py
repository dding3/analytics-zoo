import pandas as pd
import numpy as np

raw_df = pd.read_csv("/home/ding/data/nokia/data/traffic_speed_sub-dataset",  names=['road_segment', 'time_step', 'traffic_speed'])

raw_df['time'] = raw_df['time_step']*15*60

df = pd.DataFrame(pd.to_datetime(raw_df.time, unit="s", origin=pd.Timestamp('2020-01-06')))
df["traffic_speed"] = raw_df["traffic_speed"]
df["Cell"] = raw_df["road_segment"]
df.set_index("time", inplace=True)
df["month"] = df.index.month
df["week"] = df.index.week
df["dayofweek"] = df.index.dayofweek
df["hour"] = df.index.hour
df["minute"] = df.index.minute

cells = df.Cell.unique()
num_sample = 2928 * 2
test_num = 2928
look_back = 96
test_split_index = test_num + look_back
look_head = 1
# num_sample = 10
# test_num = 1
# look_back = 3
# test_split_index = test_num + look_back
# look_head = 1

concat_data_list = []

def gen_dataset_matrix(dataset, look_back):
    """
    Generate input samples from rolling
    """
    X = []
    Y = []

    data = dataset
    num_sample = data.shape[1]
    for j in range(cell_num):
        for i in range(num_sample - look_back):
            X.append(data[j, i: (i + look_back),:])
            Y.append(data[j, i + look_back, 0])
    return np.array(X), np.array(Y)

def unscale(scaler, y, target_col_indexes):
    """
    data needs to be normalized (scaled) before feeding into models. 
    This is to inverse the effect of normlization to get reasonable forecast results.
    """
    dummy_feature_shape = scaler.scale_.shape[0]
    y_dummy = np.zeros((y.shape[0], dummy_feature_shape))
    y_dummy[:, target_col_indexes] = y
    y_unscale = scaler.inverse_transform(y_dummy)[:,target_col_indexes]
    return y_unscale

for cell in cells:
    tmp = df[df["Cell"] == cell]
    tmp = tmp.drop(columns='Cell')
    data = tmp.values
    concat_data_list.append(data)

concat_data = np.array(concat_data_list)
cell_num, num_sample, feature_num = concat_data.shape
concat_data = concat_data.reshape((cell_num*num_sample, feature_num))
concat_data = concat_data.astype('float32')
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
scaled_concat_data = standard_scaler.fit_transform(concat_data)
scaled_concat_data = scaled_concat_data.reshape((cell_num, num_sample, feature_num))

train_x, train_y = gen_dataset_matrix(scaled_concat_data[:,:-test_num,:], look_back)
test_x, test_y = gen_dataset_matrix(scaled_concat_data[:,-test_split_index:,:], look_back)


EPSILON = 1e-10
def sMAPE(y_true, y_pred, multioutput="uniform_average"):
    """
    Symmetric Mean Average Percentage Error
    """
    if multioutput not in ["raw_values", "uniform_average"]:
        raise ValueError("multioutput must be 'raw_values' or 'uniform_average', got {}".format(multioutput))
    output_errors = np.mean(100 * np.abs(y_true - y_pred)/(np.abs(y_true) + np.abs(y_pred) + EPSILON), axis=0,)
    if multioutput == "raw_values":
        return output_errors
    return np.mean(output_errors)


from zoo.zouwu.model.forecast.lstm_forecaster import LSTMForecaster
lstm_config = {"lstm_units": [64]*2, "lr":0.001, "loss":"mae"}
forecaster = LSTMForecaster(target_dim=look_head, feature_dim=train_x.shape[-1], **lstm_config)
forecaster.fit(x=train_x, y=train_y, batch_size=1024, epochs=10, distributed=False)
pred_y = forecaster.predict(test_x)

pred_y=pred_y.reshape(pred_y.shape[0],)
test_y=test_y.reshape(test_y.shape[0],)
pred_y_unscale = unscale(standard_scaler, pred_y, 0)
test_y_unscale = unscale(standard_scaler, test_y, 0)
from zoo.automl.common.metrics import *
print("MAPE is", MAPE(test_y_unscale, pred_y_unscale))
from sklearn.metrics import mean_absolute_error
print("mean_squared error is", mean_absolute_error(test_y_unscale, pred_y_unscale))