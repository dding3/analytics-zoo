import os
import numpy as np
import pandas as pd

raw_df = pd.read_csv("/home/ding/data/nokia/data/train.csv", names=['road_segment', 'time', 'traffic_speed'])
raw_df.set_index("time", inplace=True)

df.index.name = "datetime"
df = df.reset_index()
df.shape

cells = df.Cell.unique()

tmp = df[df["Cell"]==cells[0]]
concat_df = pd.DataFrame(tmp["datetime"])

for cell in cells:
    tmp = df[df['Cell'] == cell]
    new_column = cell+"_DL_PRB_Util_Percent"
    concat_df[new_column] = tmp['DL_PRB_Util_Percent'].values

from zoo import init_spark_on_local
from zoo.ray import RayContext
sc = init_spark_on_local(cores=4, spark_log_level="INFO")
ray_ctx = RayContext(sc=sc, object_store_memory="1g")
ray_ctx.init()

from zoo.zouwu.autots.forecast import AutoTSTrainer

target_cols = [cell + "_DL_PRB_Util_Percent" for cell in cells]
trainer = AutoTSTrainer(dt_col="datetime",
                        target_col=target_cols,
                        horizon=1,
                        extra_features_col=None)

look_back = (15, 45)

from zoo.automl.common.util import train_val_test_split
train_df, val_df, test_df = train_val_test_split(concat_df,
                                                 val_ratio=0.1,
                                                 test_ratio=0.1,
                                                 look_back=look_back[0])

from zoo.automl.config.recipe import *
import time
start = time.time()
ts_pipeline = trainer.fit(train_df, val_df,
                          recipe=LSTMGridRandomRecipe(
                              num_rand_samples=1,
                              epochs=1,
                              look_back=look_back,
                              batch_size=[64]),
                          metric="mae")
end = time.time()
print("training time: ", end-start)
pred_df = ts_pipeline.predict(test_df)

mae, smape = ts_pipeline.evaluate(test_df, metrics=["mae", "mape"])
print("Evaluate: mae is", np.mean(mae))
print("Evaluate: mape is", np.mean(smape))

