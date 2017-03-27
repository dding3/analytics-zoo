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

package com.intel.analytics.bigdl.pipeline.ssd.example

import com.intel.analytics.bigdl.pipeline.common.{CaffeLoader, PascalVocEvaluator}
import com.intel.analytics.bigdl.pipeline.ssd._
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import scopt.OptionParser

object Test {

  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.models.fasterrcnn").setLevel(Level.INFO)

  case class PascolVocTestParam(folder: String = "",
    modelType: String = "vgg16",
    imageSet: String = "voc_2007_test",
    resultFolder: String = "/tmp/results",
    caffeDefPath: String = "",
    caffeModelPath: String = "",
    batch: Int = 8,
    nClass: Int = 0,
    resolution: Int = 300)

  val parser = new OptionParser[PascolVocTestParam]("Spark-DL SSD Test") {
    head("Spark-DL Faster-RCNN Test")
    opt[String]('f', "folder")
      .text("where you put the PascolVoc data")
      .action((x, c) => c.copy(folder = x))
      .required()
    opt[String]('o', "result")
      .text("where you put the results data")
      .action((x, c) => c.copy(resultFolder = x))
    opt[String]('t', "modelType")
      .text("net type : VGG16 | PVANET")
      .action((x, c) => c.copy(modelType = x))
      .required()
    opt[String]('i', "imageset")
      .text("imageset: voc_2007_test")
      .action((x, c) => c.copy(imageSet = x))
      .required()
    opt[String]("caffeDefPath")
      .text("caffe prototxt")
      .action((x, c) => c.copy(caffeDefPath = x))
      .required()
    opt[String]("caffeModelPath")
      .text("caffe model path")
      .action((x, c) => c.copy(caffeModelPath = x))
      .required()
    opt[Int]('b', "batch")
      .text("batch number")
      .action((x, c) => c.copy(batch = x))
    opt[Int]("nclass")
      .text("class number")
      .action((x, c) => c.copy(nClass = x))
      .required()
    opt[Int]('r', "resolution")
      .text("input resolution 300 or 512")
      .action((x, c) => c.copy(resolution = x))
      .required()
  }

  def main(args: Array[String]) {
    parser.parse(args, PascolVocTestParam()).foreach { params =>
      val conf = Engine.createSparkConf().setAppName("Spark-DL SSD Test")
        .set("spark.akka.frameSize", 64.toString)
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      Engine.init

      val evaluator = new PascalVocEvaluator(params.imageSet)

      val rdd = Preprocessor.processSeqFile(params.resolution, params.batch,
        Engine.nodeNumber * Engine.coreNumber, params.folder, sc, hasLabel = true)

      val model = params.modelType match {
        case "vgg16" => CaffeLoader.load[Float](SsdVgg(params.nClass, params.resolution),
          params.caffeDefPath, params.caffeModelPath)
        case "alexnet" => CaffeLoader.load[Float](SsdAlexnet(params.nClass, params.resolution),
          params.caffeDefPath, params.caffeModelPath)
      }

      val validator = Validator(model, PostProcessParam(params.nClass), evaluator)
      validator.test(rdd._1)
    }
  }
}
