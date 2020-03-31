package com.intel.analytics.zoo.examples

import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.FeatureSet

/**
  * Created by ding on 3/30/20.
  */
object jepSparkTest {
  def main(args: Array[String]): Unit = {
    val sc = NNContext.initNNContext("jepSparkTest")
    FeatureSet.pythonCSV(args(0), args(1), args(2).toBoolean)
  }
}
