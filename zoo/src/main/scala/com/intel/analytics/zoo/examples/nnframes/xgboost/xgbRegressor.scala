//package com.intel.analytics.zoo.examples.nnframes.xgboost
//
//import com.intel.analytics.zoo.pipeline.nnframes.XGBRegressor
//import org.apache.spark.ml.feature.{MinMaxScaler, VectorAssembler}
//import org.apache.spark.sql.SparkSession
//
//object xgbRegressor {
//  def main(args: Array[String]): Unit = {
//    val spark = SparkSession.builder().getOrCreate()
//    val baseDataPath = args(0)
//    val dataPath = args(1)
//    val resPath = args(2)
//    val numRounds = args(3).toInt
//
//    val baseData = spark.read.format("csv")
//      .option("sep", ",")
//      .option("inferSchema", "true")
//      .option("header", "true")
//      .load(baseDataPath)
//      .withColumnRenamed("src_eci", "CellID")
//    val newData = baseData.na.drop()
//    val data = spark.read.format("csv")
//      .option("sep", ",")
//      .option("inferSchema", "true")
//      .option("header", "true")
//      .load(dataPath)
//
//
//    val filterData = data.filter(data("Longitude") > 115.5 &&
//      data("Longitude") < 117.5 && data("Latitude") > 39.5 &&
//      data("Latitude") < 41)
//
//    val joinData = filterData.join(newData, "CellID").select("base_lon","base_lat",
//      "base_angle","NC1PCI","NC1Freq","NC2PCI","NC2Freq", "SCRSRP","NC1RSRP","NC2RSRP",
//      "Longitude", "Latitude")
//
//    val vectorAssembler = new VectorAssembler()
//      .setInputCols(Array("base_lon","base_lat",
//        "base_angle","NC1PCI","NC1Freq","NC2PCI","NC2Freq", "SCRSRP","NC1RSRP","NC2RSRP"))
//      .setOutputCol("features_vec")
//    val df = vectorAssembler.transform(joinData)
//
//    val scaler = new MinMaxScaler().setInputCol("features_vec").setOutputCol("features")
//    val scalerModel = scaler.fit(df)
//    val scaledData = scalerModel.transform(df).cache()
//
//    val df3 = scaledData.select("features", "Longitude", "Latitude").cache()
//    val trainTest = df3.randomSplit(Array(0.8, 0.2))
//    val train = trainTest(0)
//    val test = trainTest(1)
//
//    val xgbRf0 = new XGBRegressor().
//      setLabelCol("Longitude")
//    val xgbRf1 = new XGBRegressor().
//      setLabelCol("Latitude")
//    xgbRf0.setNumRound(numRounds) // n_estimators
//    xgbRf0.setMaxDepth(50) // max_depth
//    xgbRf0.setNthread(1) // n_jobs
//    xgbRf0.setTreeMethod("hist") // tree_method
//    xgbRf0.setSeed(2)  // random_state
//    xgbRf0.setEta(0.1) //learing rate
//    xgbRf0.setMinChildWeight(1) //min_child_weight
//    xgbRf0.setSubsample(0.8) //subsample
//    xgbRf0.setColsampleBytree(0.8) //colsample_bytree
//    xgbRf0.setGamma(0) // gamma
//    xgbRf0.setAlpha(0) // reg_alpha
//    xgbRf0.setLambda(1) // reg_lambda
//    xgbRf0.setNumWorkers(4)
//    //    xgbRf0.setMaxDeltaStep(0)
//
//    xgbRf1.setNumRound(numRounds) // n_estimators
//    xgbRf1.setMaxDepth(50) // max_depth
//    xgbRf1.setNthread(1) // n_jobs
//    xgbRf1.setTreeMethod("hist") // tree_method
//    xgbRf1.setSeed(2)  // random_state
//    xgbRf1.setEta(0.1) //learing rate
//    xgbRf1.setMinChildWeight(1) //min_child_weight
//    xgbRf1.setSubsample(0.8) //subsample
//    xgbRf1.setColsampleBytree(0.8) //colsample_bytree
//    xgbRf1.setGamma(0) // gamma
//    xgbRf1.setAlpha(0) // reg_alpha
//    xgbRf1.setLambda(1) // reg_lambda
//    xgbRf1.setNumWorkers(4)
//
//    val xgbRegressorModel0 = xgbRf0.fit(train)
//    val xgbRegressorModel1 = xgbRf1.fit(train)
//
//    val predictY0 = xgbRegressorModel0.transform(test)
//      .select("Longitude", "prediction", "Latitude", "features")
//      .withColumnRenamed("prediction", "predict_Longitude").cache()
//
//    val predictY1 = xgbRegressorModel1.transform(predictY0)
//      .withColumnRenamed("prediction", "predict_Latitude").cache()
//
//    val result = predictY1.select("Longitude", "predict_Longitude", "Latitude", "predict_Latitude")
//    result
//    .write
//    .option("header","true")
//    .option("sep",",")
//    .mode("overwrite")
//    .csv(resPath + "result")
//    println("End3")
//  }
//}
