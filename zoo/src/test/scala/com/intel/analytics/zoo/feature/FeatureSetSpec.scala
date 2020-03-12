/*
 * Copyright 2018 Analytics Zoo Authors.
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
package com.intel.analytics.zoo.feature

import java.util

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.zoo.common.{NNContext, Utils}
import com.intel.analytics.zoo.core.TFNetNative
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import jep.{JepConfig, NDArray, NamingConventionClassEnquirer, SharedInterpreter}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext, TaskContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer
import scala.tools.nsc.interpreter.JList


class FeatureSetSpec extends FlatSpec with Matchers with BeforeAndAfter {
  var sc : SparkContext = _

  before {
    val conf = new SparkConf().setAppName("Test Feature Set").setMaster("local[1]")
      .set("spark.executorEnv.PYTHONHOME","/home/ding/tensorflow_venv/venv-py3/lib/python3.5/site-packages/jep")
    sc = NNContext.initNNContext(conf)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "FeatureSet" should "iterate 2" in {
    import PythonLoaderFeatureSet._
    val interpRdd = getOrCreateInterpRdd()
    val nodeNumber = EngineRef.getNodeNumber()
    val preimports = s"""
                        |from pyspark.serializers import CloudPickleSerializer
                        |import numpy as np
                        |import pandas as pd
                        |""".stripMargin
    interpRdd.mapPartitions{iter =>
      val interp = iter.next()
      val partId = TaskContext.getPartitionId()
      require(partId < nodeNumber, s"partId($partId) should be" +
        s" smaller than nodeNumber(${nodeNumber})")
      interp.exec(preimports)
      Iterator.single(interp)
    }.count()

    import com.intel.analytics.zoo.common
    val path = Utils.listPaths("/home/ding/data/skt/raw_csv")
    val dataPath = sc.parallelize(path)

    val sampleRDD = interpRdd.zipPartitions(dataPath){(iterpIter, pathIter) =>
      val interp = iterpIter.next()
      import java.util.{ArrayList => JArrayList}
      val data = ArrayBuffer[JArrayList[util.Collection[AnyRef]]]()
      while(!pathIter.isEmpty) {
        val path = pathIter.next()
        interp.eval("from preprocess_test import parse_local_csv, get_feature_label_list")
        interp.eval(s"result = parse_local_csv('${path}')")
        interp.eval(s"result2 = get_feature_label_list(result)")
        data.append(interp.getValue("result2").asInstanceOf[JArrayList[util.Collection[AnyRef]]])
      }

      import scala.collection.JavaConverters._
      val record = data.toArray.flatMap(_.asScala)
      val sample = record.map {x =>
        val features = x.asScala.head.asInstanceOf[JList[NDArray[_]]].asScala.map(ndArrayToTensor(_))
        val label = ndArrayToTensor(x.asScala.last.asInstanceOf[NDArray[_]])
        Sample(features.toArray, label)
      }

      sample.iterator
    }

    val t2 = FeatureSet.rdd(sampleRDD)
    val t3 = 0
  }

  "FeatureSet" should "iterate in sequential order without shuffle" in {

    val rdd = sc.parallelize(0 until 10, numSlices = 1)

    val fs = FeatureSet.rdd(rdd, sequentialOrder = true, shuffle = false)

    val data = fs.data(train = true)
    val seq = for (i <- 0 until 10) yield {
      data.first()
    }

    assert(seq == (0 until 10))
    fs.unpersist()
  }


  "FeatureSet" should "iterate in sequential order with shuffle" in {
    val rdd = sc.parallelize(0 until 10, numSlices = 1)

    val fs = FeatureSet.rdd(rdd, sequentialOrder = true)
    fs.shuffle()
    val data = fs.data(train = true)
    val set = scala.collection.mutable.Set[Int]()
    set ++= (0 until 10)

    val firstRound = for (i <- 0 until 10) yield {
      val value = data.first()

      set -= value
    }

    assert(firstRound != (0 until 10))
    assert(set.isEmpty)
    fs.unpersist()
  }
}
