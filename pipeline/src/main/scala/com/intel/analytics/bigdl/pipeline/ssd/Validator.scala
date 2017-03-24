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
import com.intel.analytics.bigdl.pipeline.common.{DetectionEvaluator}
import com.intel.analytics.bigdl.pipeline.common.dataset.roiimage.MiniBatch
import com.intel.analytics.bigdl.pipeline.ssd.Validator._
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD


class Validator(model: Module[Float], param: PostProcessParam,
  evaluator: DetectionEvaluator) {

  val postProcessor = Postprocessor(param)

  def test(rdd: RDD[MiniBatch[Float]]): Array[(String, Double)] = {
    model.evaluate()
    val broadcastModel = ModelBroadcast().broadcast(rdd.sparkContext, model)
    val broadpostprocessor = rdd.sparkContext.broadcast(postProcessor)
    val broadcastEvaluator = rdd.sparkContext.broadcast(evaluator)
    val recordsNum = rdd.sparkContext.accumulator(0, "record number")
    val start = System.nanoTime()
    val output = rdd.mapPartitions(dataIter => {
      val localModel = broadcastModel.value()
      val localPostProcessor = broadpostprocessor.value.clone()
      val localEvaluator = broadcastEvaluator.value
      dataIter.map(batch => {
        val result = localModel.forward(batch.data).toTable
        val out = localPostProcessor.process(result, batch.imInfo)
        recordsNum += batch.data.size(1)
        localEvaluator.evaluateBatch(out, batch.labels)
      })
    }).reduce((left, right) => {
      left.zip(right).map { case (l, r) =>
        (l._1 + r._1, l._2 ++ r._2)
      }
    })

    val totalTime = (System.nanoTime() - start) / 1e9
    logger.info(s"[Prediction] ${ recordsNum.value } in $totalTime seconds. Throughput is ${
      recordsNum.value / totalTime
    } record / sec")
    evaluator.map(output)
  }
}

object Validator {
  val logger = Logger.getLogger(this.getClass)

  def apply(model: Module[Float], param: PostProcessParam, evaluator: DetectionEvaluator)
  : Validator = new Validator(model, param, evaluator)
}

