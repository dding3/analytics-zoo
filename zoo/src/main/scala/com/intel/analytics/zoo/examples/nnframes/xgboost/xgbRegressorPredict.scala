package com.intel.analytics.zoo.examples.nnframes.xgboost

import com.intel.analytics.zoo.pipeline.nnframes.XGBRegressorModel
import org.apache.spark.ml.feature.{MinMaxScaler, VectorAssembler}
import org.apache.spark.sql.SparkSession

object xgbRegressorPredict {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().getOrCreate()
    val baseDataPath = args(0)
    val dataPath = args(1)
    val modelPath = args(2)
    val resPath = args(3)

    val baseData = spark.read.format("csv")
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .load(baseDataPath)
      .withColumnRenamed("src_eci", "CellID")
    val newData = baseData.na.drop()
    val data = spark.read.format("csv")
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .load(dataPath)


    val filterData = data.filter(data("Longitude") > 115.5 &&
      data("Longitude") < 117.5 && data("Latitude") > 39.5 &&
      data("Latitude") < 41)

    var joinData = filterData.join(newData, "CellID").select("base_lon","base_lat",
      "base_angle","NC1PCI","NC1Freq","NC2PCI","NC2Freq", "SCRSRP","NC1RSRP","NC2RSRP",
      "Longitude", "Latitude")

    val columns = Array("base_lon","base_lat","base_angle","NC1PCI","NC1Freq","NC2PCI", "NC2Freq",
      "SCRSRP","NC1RSRP","NC2RSRP")

    for(i <- 0 until columns.length) {
      val column = columns(i)
      val vectorAssembler = new VectorAssembler()
        .setInputCols(Array(column))
        .setOutputCol(column + "vec")
      joinData = vectorAssembler.transform(joinData)
      val scaler = new MinMaxScaler().setInputCol(column + "vec").setOutputCol(column + "scaled")
      val model = scaler.fit(joinData)
      joinData = model.transform(joinData)
    }

    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("base_lonscaled","base_latscaled",
        "base_anglescaled","NC1PCIscaled","NC1Freqscaled","NC2PCIscaled","NC2Freqscaled",
        "SCRSRPscaled","NC1RSRPscaled","NC2RSRPscaled"))
      .setOutputCol("features")
    val df = vectorAssembler.transform(joinData)

    val df3 = df.select("features", "Longitude", "Latitude")
//    val trainTest = df3.randomSplit(Array(0.8, 0.2))
//    val train = trainTest(0)
//    val test = trainTest(1).cache()

    val xgbRegressorModel0 = XGBRegressorModel.loadFromXGB(modelPath + "xgbregressor0.model")
    val xgbRegressorModel1 = XGBRegressorModel.loadFromXGB(modelPath + "xgbregressor1.model")

    xgbRegressorModel0.setPredictionCol("predict_Longitude")
    val predictY0 = xgbRegressorModel0.transform(df3)
      .select("Longitude", "predict_Longitude", "Latitude", "features")
      .cache()

    xgbRegressorModel1.setPredictionCol("predict_Latitude")
    val predictY1 = xgbRegressorModel1.transform(predictY0).cache()
    predictY1.show()

//    val result = predictY1.select("Longitude", "predict_Longitude", "Latitude", "predict_Latitude")
//    result
//      .write
//      .option("header","true")
//      .option("sep",",")
//      .mode("overwrite")
//      .csv(resPath + "result")
    println("End")
  }
}
