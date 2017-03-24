/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.pipeline.ssd

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.pipeline.ssd.Ssd._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat

object SsdVgg {

  private def vgg16Part1(): Sequential[Float] = {
    val vggNetPart1 = Sequential()
    addConvRelu(vggNetPart1, (3, 64, 3, 1, 1), "1_1")
    addConvRelu(vggNetPart1, (64, 64, 3, 1, 1), "1_2")
    vggNetPart1.add(SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool1"))

    addConvRelu(vggNetPart1, (64, 128, 3, 1, 1), "2_1")
    addConvRelu(vggNetPart1, (128, 128, 3, 1, 1), "2_2")
    vggNetPart1.add(SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool2"))

    addConvRelu(vggNetPart1, (128, 256, 3, 1, 1), "3_1")
    addConvRelu(vggNetPart1, (256, 256, 3, 1, 1), "3_2")
    addConvRelu(vggNetPart1, (256, 256, 3, 1, 1), "3_3")
    vggNetPart1.add(SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool3"))

    addConvRelu(vggNetPart1, (256, 512, 3, 1, 1), "4_1")
    addConvRelu(vggNetPart1, (512, 512, 3, 1, 1), "4_2")
    addConvRelu(vggNetPart1, (512, 512, 3, 1, 1), "4_3")
    vggNetPart1
  }

  private def vgg16Part2(): Sequential[Float] = {
    val vggNetPart2 = Sequential()
    vggNetPart2.add(SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool4"))
    addConvRelu(vggNetPart2, (512, 512, 3, 1, 1), "5_1")
    addConvRelu(vggNetPart2, (512, 512, 3, 1, 1), "5_2")
    addConvRelu(vggNetPart2, (512, 512, 3, 1, 1), "5_3")
    vggNetPart2.add(SpatialMaxPooling(3, 3, 1, 1, 1, 1).ceil().setName("pool5"))
    vggNetPart2
  }

  def apply(numClasses: Int, resolution: Int = 300): Module[Float] = {
    require(resolution == 300 || resolution == 512, "only support 300*300 or 512*512 as input")
    val isClip = false
    val isFlip = true
    val variances = Array(0.1f, 0.1f, 0.2f, 0.2f)
    var params = Map[String, ComponetParam]()
    if (resolution == 300) {
      params += "conv4_3_norm" -> ComponetParam(512, 4, minSizes = Array(30),
        maxSizes = Array(60), aspectRatios = Array(2), isFlip, isClip, variances, 8)
      params += "fc7" -> ComponetParam(512, 6, minSizes = Array(60), maxSizes = Array(111),
        aspectRatios = Array(2, 3), isFlip, isClip, variances, 16)
      params += "conv6_2" -> ComponetParam(512, 6, minSizes = Array(111), maxSizes = Array(162),
        aspectRatios = Array(2, 3), isFlip, isClip, variances, 32)
      params += "conv7_2" -> ComponetParam(256, 6, minSizes = Array(162), maxSizes = Array(213),
        aspectRatios = Array(2, 3), isFlip, isClip, variances, 64)
      params += "conv8_2" -> ComponetParam(256, 4, minSizes = Array(213), maxSizes = Array(264),
        aspectRatios = Array(2), isFlip, isClip, variances, 100)
      params += "conv9_2" -> ComponetParam(256, 4, minSizes = Array(264), maxSizes = Array(315),
        aspectRatios = Array(2), isFlip, isClip, variances, 300)
      Ssd(numClasses, resolution, vgg16Part1(), vgg16Part2(), params, normScale = 13.25217724f,
        isLastPool = false)
    } else {
      params += "conv4_3_norm" -> ComponetParam(512, 4, minSizes = Array(35.84f),
        maxSizes = Array(76.8f), aspectRatios = Array(2), isFlip, isClip, variances, 8)
      params += "fc7" -> ComponetParam(512, 6, minSizes = Array(76.8f), maxSizes = Array(153.6f),
        aspectRatios = Array(2, 3), isFlip, isClip, variances, 16)
      params += "conv6_2" -> ComponetParam(512, 6, minSizes = Array(153.6f),
        maxSizes = Array(230.4f), aspectRatios = Array(2, 3), isFlip, isClip, variances, 32)
      params += "conv7_2" -> ComponetParam(256, 6, minSizes = Array(230.4f),
        maxSizes = Array(307.2f), aspectRatios = Array(2, 3), isFlip, isClip, variances, 64)
      params += "conv8_2" -> ComponetParam(256, 6, minSizes = Array(307.2f),
        maxSizes = Array(384.0f), aspectRatios = Array(2, 3), isFlip, isClip, variances, 128)
      params += "conv9_2" -> ComponetParam(256, 4, minSizes = Array(384.0f),
        maxSizes = Array(460.8f), aspectRatios = Array(2), isFlip, isClip, variances, 256)
      params += "conv10_2" -> ComponetParam(256, 4, minSizes = Array(460.8f),
        maxSizes = Array(537.6f), aspectRatios = Array(2), isFlip, isClip, variances, 512)
      Ssd(numClasses, resolution, vgg16Part1(), vgg16Part2(), params, normScale = 12.0166f,
        isLastPool = false)
    }
  }
}
