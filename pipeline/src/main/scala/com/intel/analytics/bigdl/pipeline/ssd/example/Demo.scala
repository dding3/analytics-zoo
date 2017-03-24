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

import com.intel.analytics.bigdl.dataset.image.Visualizer
import com.intel.analytics.bigdl.pipeline.ssd._
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.utils.{Engine, File}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import scopt.OptionParser

import scala.io.Source

object Demo {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.models.fasterrcnn").setLevel(Level.INFO)

  val logger = Logger.getLogger(getClass)

  case class PascolVocDemoParam(imageFolder: String = "",
    outputFolder: String = "data/demo",
    folderType: String = "local",
    modelType: String = "vgg16",
    caffeDefPath: String = "",
    caffeModelPath: String = "",
    batch: Int = 8,
    vis: Boolean = true,
    classname: String = "",
    resolution: Int = 300)

  val parser = new OptionParser[PascolVocDemoParam]("Spark-DL SSD Demo") {
    head("Spark-DL Faster-RCNN Demo")
    opt[String]('f', "folder")
      .text("where you put the demo image data")
      .action((x, c) => c.copy(imageFolder = x))
      .required()
    opt[String]("folderType")
      .text("local image folder or hdfs sequence folder")
      .action((x, c) => c.copy(folderType = x))
      .required()
      .validate(x => {
        if (Set("local", "seq").contains(x.toLowerCase)) {
          success
        } else {
          failure("folderType only support local|seq")
        }
      })
    opt[String]('o', "output")
      .text("where you put the output data")
      .action((x, c) => c.copy(outputFolder = x))
      .required()
    opt[String]('t', "modelType")
      .text("net type : vgg16 | alexnet")
      .action((x, c) => c.copy(modelType = x))
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
    opt[Boolean]('v', "vis")
      .text("whether to visualize the detections")
      .action((x, c) => c.copy(vis = x))
    opt[String]("classname")
      .text("file store class name")
      .action((x, c) => c.copy(classname = x))
      .required()
    opt[Int]('r', "resolution")
      .text("input resolution 300 or 512")
      .action((x, c) => c.copy(resolution = x))
      .required()
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, PascolVocDemoParam()).foreach { params =>
      val conf = Engine.createSparkConf().setAppName("Spark-DL SSD Demo")
        .set("spark.akka.frameSize", 64.toString)
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      Engine.init
      val classNames = Source.fromFile(params.classname).getLines().toArray
      val model = params.modelType match {
        case "vgg16" => Module.loadCaffe(SsdVgg(classNames.length, params.resolution),
          params.caffeDefPath, params.caffeModelPath)
        case "alexnet" => Module.loadCaffe(SsdAlexnet(classNames.length, params.resolution),
          params.caffeDefPath, params.caffeModelPath)
      }

      File.save(model, params.caffeDefPath.
        substring(0, params.caffeDefPath.lastIndexOf(".")) + ".bigdl", true)
      val nPartition = Engine.nodeNumber * Engine.coreNumber
      val (minibatch, imagePaths) = params.folderType match {
        case "local" => Preprocessor.processFolder(params.resolution, params.batch, nPartition,
          params.imageFolder, sc)
        case "seq" => Preprocessor.processSeqFile(params.resolution, params.batch, nPartition,
          params.imageFolder, sc, hasLabel = false)
      }

      val predictor = Predictor(model, PostProcessParam(classNames.length))
      val output = predictor.test(minibatch)

      if (params.vis) {
        imagePaths.zipWithIndex.foreach(pair => {
          var clsInd = 1
          val imgId = pair._2.toInt
          while (clsInd < classNames.length) {
            Visualizer.visDetection(pair._1, classNames(clsInd),
              output(imgId)(clsInd).classes,
              output(imgId)(clsInd).bboxes, thresh = 0.6f, outPath = params.outputFolder)
            clsInd += 1
          }
        })
        logger.info(s"labeled images are saved to ${ params.outputFolder }")
      }
    }
  }
}
