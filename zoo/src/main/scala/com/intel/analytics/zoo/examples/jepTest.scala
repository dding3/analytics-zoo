package com.intel.analytics.zoo.examples

import java.util

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.zoo.common.Utils
import com.intel.analytics.zoo.feature.PythonLoaderFeatureSet

object jepTest {
  def main(args: Array[String]): Unit = {
    import java.util.{ArrayList => JArrayList}

    val start = System.nanoTime()
    import scala.collection.JavaConverters._
    import com.intel.analytics.zoo.common
    import PythonLoaderFeatureSet._

    val path = Utils.listPaths(args(0))
    val pathStr = path.mkString(",")

    val interp = getOrCreateInterpreter(args(1))

    interp.eval("from preprocess import parse_csv2")
    interp.eval(s"result = parse_csv2('${pathStr}')")
    val t = interp.getValue("result")
//    val record = t.asInstanceOf[JArrayList[util.Collection[AnyRef]]].asScala
//
//    val sample = record.map {x =>
//      val iter = x.iterator()
//      val features = toArrayTensor(iter.next())
//      val labels = toArrayTensor(iter.next()).head
//      Sample(features, labels)
//    }
    val end = System.nanoTime()
    val time = (end - start) / 1e9
    println(s"Elapse $time s")  }

}
