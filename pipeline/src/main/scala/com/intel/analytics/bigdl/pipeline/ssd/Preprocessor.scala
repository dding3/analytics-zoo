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

import java.io.File

import com.intel.analytics.bigdl.pipeline.common.dataset.LocalByteRoiimageReader
import com.intel.analytics.bigdl.pipeline.common.dataset.roiimage._
import org.apache.hadoop.io.Text
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD


object Preprocessor {
  private val PIXEL_MEANS_RGB = (123f, 117f, 104f)

  def processSeqFile(resolution: Int, batchSize: Int, nPartition: Int, seqFloder: String,
    sc: SparkContext, resizeRois: Boolean = false, hasLabel: Boolean = false)
  : (RDD[MiniBatch[Float]], RDD[String]) = {
    val transformers = RoiImageResizer(Array(resolution), resizeRois = resizeRois,
      isEqualResize = true) ->
      RoiImageNormalizer(PIXEL_MEANS_RGB) -> RoiimageToBatch(batchSize, toRGB = false, hasLabel)
    val recordToByteImageWithRoi = RecordToByteImageWithRoi()
    val roiByteImages = sc.sequenceFile(seqFloder, classOf[Text], classOf[Text], nPartition)
      .map(x => recordToByteImageWithRoi.transform(x._1, x._2))
    val transformed = roiByteImages.mapPartitions(transformers(_))
    (transformed, roiByteImages.map(_.path))
  }

  def processFolder(resolution: Int, batchSize: Int, nPartition: Int,
    folder: String, sc: SparkContext, resizeRois: Boolean = false, hasLabel: Boolean = false)
  : (RDD[MiniBatch[Float]], RDD[String]) = {
    val transformers = RoiImageResizer(Array(resolution), resizeRois = resizeRois,
      isEqualResize = true) ->
      RoiImageNormalizer(PIXEL_MEANS_RGB) -> RoiimageToBatch(batchSize, toRGB = false, hasLabel)
    val fileNames = new File(folder).listFiles().map(_.getAbsolutePath)
    val roiDataset = fileNames.map(RoiImagePath(_))
    val imgReader = LocalByteRoiimageReader()
    val imageWithRois = sc.parallelize(roiDataset.map(roidb => imgReader.transform(roidb)),
      nPartition)
    val rdd = imageWithRois.mapPartitions(transformers(_))
    (rdd, sc.parallelize(fileNames, nPartition))
  }
}
