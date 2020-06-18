#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import cv2

from zoo.common.nncontext import init_nncontext
from zoo.models.image.objectdetection import *

from pyspark.sql import SparkSession

file_path = "/home/ding/proj/AsiaInfo/huying/"
base_data_path = file_path + "new_beijing_telecom_gongcan.csv"
data_path = file_path + "test.csv"

spark = SparkSession \
    .builder \
    .getOrCreate()
base_data = spark.read.csv(base_data_path, sep=",", inferSchema=True, header=True)
base_data = base_data.withColumnRenamed("src_eci", "CellID")
new_data = base_data.na.drop()
data = spark.read.csv(data_path, sep=",", inferSchema=True, header=True)
filter_data = data.filter((data.Longitude > 115.5) & (data.Longitude < 117.5)
                          & (data.Latitude > 39.5) & (data.Latitude < 41))
join_data = filter_data.join(new_data, on=["CellID"], how="inner").drop("CellID")

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


assembler = VectorAssembler(
    inputCols=["base_lon","base_lat","base_angle","NC1PCI","NC1Freq","NC2PCI","NC2Freq","SCRSRP","NC1RSRP","NC2RSRP"],
    outputCol="features")

df = assembler.transform(join_data).select("features", "Longitude", "Latitude")

from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors

scaler = MinMaxScaler(inputCol="features", outputCol="scaled")

# Compute summary statistics and generate MinMaxScalerModel
scalerModel = scaler.fit(df)
scaledData = scalerModel.transform(df)

train, test = scaledData.randomSplit([0.8, 0.2])
params = {"n_estimators":500,
      "max_depth" : 50,
      "n_jobs" : -1,
      "tree_method" : "hist",
      "random_state" : 2,
      "learning_rate" : 0.1,
      "min_child_weight" : 1,
      "seed" : 0,
      "subsample" : 0.8,
      "colsample_bytree" : 0.8,
      "gamma" : 0,
      "reg_alpha" : 0,
      "reg_lambda" : 1,
      "verbosity" : 0,
      "num_workers" : spark.sparkContext.defaultParallelism}

from zoo.pipeline.nnframes import *
xgbrf0 = XGBRegressor(params).setFeaturesCol("scaled").setLabelCol("Longitude")
xgbrf1 = XGBRegressor(params).setFeaturesCol("scaled").setLabelCol("Latitude")

xgbRegressorModel0 = xgbrf0.fit(train)
xgbRegressorModel1 = xgbrf1.fit(train)
t=0



sc = init_nncontext("Object Detection Example")

parser = argparse.ArgumentParser()
parser.add_argument('model_path', help="Path where the model is stored")
parser.add_argument('img_path', help="Path where the images are stored")
parser.add_argument('output_path', help="Path to store the detection results")
parser.add_argument("--partition_num", type=int, default=1, help="The number of partitions")


def predict(model_path, img_path, output_path, partition_num):
    model = ObjectDetector.load_model(model_path)
    image_set = ImageSet.read(img_path, sc, image_codec=1, min_partitions=partition_num)
    output = model.predict_image_set(image_set)

    config = model.get_config()
    visualizer = Visualizer(config.label_map(), encoding="jpg")
    visualized = visualizer(output).get_image(to_chw=False).collect()
    for img_id in range(len(visualized)):
        cv2.imwrite(output_path + '/' + str(img_id) + '.jpg', visualized[img_id])


if __name__ == "__main__":
    args = parser.parse_args()
    predict(args.model_path, args.img_path, args.output_path, args.partition_num)

print("finished...")
sc.stop()
