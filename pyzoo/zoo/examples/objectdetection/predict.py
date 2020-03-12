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
# import cv2

from zoo.common.nncontext import init_nncontext
from zoo.models.image.objectdetection import *


sc = init_nncontext("Object Detection Example")

from zoo.feature.common import FeatureSet
feature_set = FeatureSet.csv_dataset()

from zoo.tfpark import TFDataset, TFOptimizer
import tensorflow as tf
from AR_mem.config import Config
from AR_mem.model import Model
dataset = TFDataset.from_feature_set(feature_set,
                                      features=[(tf.float32, [10, 8]), (tf.float32, [77, 8])],
                                      labels=(tf.float32, [8]),
                                      batch_size=4)

from bigdl.optim.optimizer import Adam, MaxEpoch
config = Config()
model = Model(config, dataset.tensors[0][0], dataset.tensors[0][1], dataset.tensors[1])
optimizer = TFOptimizer.from_loss(model.loss, Adam(config.lr),
                                  metrics={"rse": model.rse, "smape": model.smape, "mae": model.mae},
                                  session_config=tf.ConfigProto(inter_op_parallelism_threads=1,
                                                                intra_op_parallelism_threads=10)
                                  )

optimizer.optimize(end_trigger=MaxEpoch(1))



t = 0




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
