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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.pipeline.common.dataset.roiimage.{MiniBatch, Target}
import com.intel.analytics.bigdl.pipeline.ssd.Predictor._
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD


class Predictor(model: Module[Float], param: PostProcessParam) {
  val postProcessor = Postprocessor(param)

  def test(rdd: RDD[MiniBatch[Float]]): Array[Array[Target]] = {
    model.evaluate()
    val broadcastModel = ModelBroadcast().broadcast(rdd.sparkContext, model)
    val broadpostprocessor = rdd.sparkContext.broadcast(postProcessor)
    val recordsNum = rdd.sparkContext.accumulator(0, "record number")
    val start = System.nanoTime()
    val output = rdd.mapPartitions(dataIter => {
      val localModel = broadcastModel.value()
      val localPostProcessor = broadpostprocessor.value.clone()
      dataIter.map(batch => {
        val result = localModel.forward(batch.data).toTable
        recordsNum += batch.data.size(1)
        localPostProcessor.process(result, batch.imInfo)
      })
    }).collect().flatten
    val totalTime = (System.nanoTime() - start) / 1e9
    logger.info(s"[Prediction] ${ recordsNum.value } in $totalTime seconds. Throughput is ${
      recordsNum.value / totalTime
    } record / sec")
    output
  }
}

object Predictor {
  val logger = Logger.getLogger(this.getClass)

  def apply(model: Module[Float], param: PostProcessParam)
  : Predictor = new Predictor(model, param)
}

