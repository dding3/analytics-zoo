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

package com.intel.analytics.bigdl.pipeline.common.dataset.roiimage

import java.nio.ByteBuffer

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.hadoop.io.Text

import scala.collection.Iterator

class RecordToByteImageWithRoi extends Transformer[(Text, Text), RoiByteImage] {
  override def apply(prev: Iterator[(Text, Text)]): Iterator[RoiByteImage] = {
    prev.map(record => {
      transform(record._1, record._2)
    })
  }

  @transient var data: Array[Byte] = _
  @transient var gtClasses: Tensor[Float] = _
  @transient var gtBoxes: Tensor[Float] = _


  def transform(key: Text, value: Text): RoiByteImage = {
    val byteArray = value.copyBytes()
    val byteBuffer = ByteBuffer.wrap(byteArray)
    val dataLen = byteBuffer.getInt
    val classLen = byteBuffer.getInt
    if (data == null || data.length < dataLen) {
      data = new Array[Byte](dataLen)
    }
    System.arraycopy(byteArray, 8, data, 0, dataLen)
    val target = if (byteArray.length > 8 + dataLen) {
      if (gtBoxes == null) {
        gtClasses = Tensor[Float]
        gtBoxes = Tensor[Float]
      }
      gtClasses.resize(2, classLen / 4)
      gtBoxes.resize(classLen / 4, 4)
      // label + difficult
      bytesToFloatTensor(byteArray, 8 + dataLen, classLen * 2, gtClasses.storage().array())
      bytesToFloatTensor(byteArray, 8 + dataLen + classLen * 2, classLen * 4,
        gtBoxes.storage().array())
      Target(gtClasses, gtBoxes)
    } else null
    RoiByteImage(data, dataLen, key.toString, target)
  }

  def bytesToFloatTensor(src: Array[Byte], offset: Int, length: Int, target: Array[Float] = null)
  : Unit = {
    val buffer = ByteBuffer.wrap(src, offset, length)
    var i = 0
    while (i < length / 4) {
      target(i) = buffer.getFloat()
      i += 1
    }
  }
}

object RecordToByteImageWithRoi {
  def apply(): RecordToByteImageWithRoi = new RecordToByteImageWithRoi()
}
