package com.intel.analytics.zoo.examples.nnframes.xgboost

import com.intel.analytics.zoo.pipeline.nnframes.XGBClassifierModel

object xgbClassifier {
  def main(args: Array[String]): Unit = {
    val num_classes = args(2).toInt
    import org.apache.spark.sql.SparkSession
    val spark = SparkSession.builder().getOrCreate()
    val filePath = args(1)

    val df = spark.read.format("csv")
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .load(filePath)

    val model_path = args(0)

    val voiceModelPath = model_path + "xgb_yuyin-18-16.model"
    val voiceModel = XGBClassifierModel.load(voiceModelPath, num_classes)
    voiceModel.setFeaturesCol(Array("分公司名称", "用户入网时间", "用户状态", "年龄", "性别", "用户星级",
    "是否集团成员", "城市农村用户", "是否欠费用户", "主套餐费用", "通话费用", "通话费用趋势", "VoLTE掉话率", "ESRVCC切换时延",
    "ESRVCC切换比例", "ESRVCC切换成功率", "VoLTE接续时长", "呼叫建立时长", "VoLTE接通率",
    "全程呼叫成功率", "VoLTE掉话率_diff",
    "ESRVCC切换时延_diff", "ESRVCC切换比例_diff", "ESRVCC切换成功率_diff",
    "VoLTE接续时长_diff", "呼叫建立时长_diff", "VoLTE接通率_diff", "全程呼叫成功率_diff"))
    voiceModel.setPredictionCol("voice")
    val voicePredictDF = voiceModel.transform(df).select("voice", "手机号码")

    val mobileModelPath = model_path + "xgb_shouji-18-16.model"
    val mobileModel = XGBClassifierModel.load(mobileModelPath, num_classes)
    mobileModel.setFeaturesCol(Array("分公司名称", "用户入网时间", "用户状态", "年龄", "性别", "用户星级",
    "是否集团成员", "城市农村用户", "主套餐费用", "流量费用",
    "流量费用趋势",
    "网页响应成功率", "网页响应时延", "网页显示成功率", "网页浏览成功率", "网页打开时长",
    "视频响应成功率", "视频响应时延", "视频平均每次播放卡顿次数", "视频播放成功率", "视频播放等待时长",
    "即时通信接入成功率", "即时通信接入时延", "下载速率", "上传速率",
    "网页响应成功率_diff", "网页响应时延_diff", "网页显示成功率_diff", "网页浏览成功率_diff",
    "网页打开时长_diff", "视频响应成功率_diff", "视频响应时延_diff", "视频平均每次播放卡顿次数_diff",
    "视频播放成功率_diff", "视频播放等待时长_diff",
    "即时通信接入成功率_diff", "即时通信接入时延_diff", "下载速率_diff", "上传速率_diff"))
    mobileModel.setPredictionCol("mobile")
    val mobilePredictDF = mobileModel.transform(df).select("mobile", "手机号码")

    val feeModelPath = model_path + "xgb_zifei-18-16.model"
    val feeModel = XGBClassifierModel.load(feeModelPath, num_classes)
    val feeDF = df.join(mobilePredictDF, "手机号码").join(voicePredictDF, "手机号码")
    feeDF.select("年龄", "性别", "用户入网时间", "用户星级", "是否集团成员", "城市农村用户", "是否欠费用户",
    "主套餐费用", "超套费用", "通话费用", "通话费用趋势", "流量费用", "流量费用趋势",
    "近3月的平均出账费用", "近3月的平均出账费用趋势", "近3月超套平均", "近3月月均欠费金额", "用户状态", "分公司名称",
    "voice", "mobile")
    feeModel.setFeaturesCol(Array("年龄", "性别", "用户入网时间", "用户星级", "是否集团成员", "城市农村用户", "是否欠费用户",
    "主套餐费用", "超套费用", "通话费用", "通话费用趋势", "流量费用", "流量费用趋势",
    "近3月的平均出账费用", "近3月的平均出账费用趋势", "近3月超套平均", "近3月月均欠费金额", "用户状态", "分公司名称", "mobile", "voice"))
    feeModel.setPredictionCol("fee")
    val feePredictDF = feeModel.transform(feeDF).select("fee", "voice", "mobile", "手机号码")

    val satisfyModelPath = model_path + "xgb_manyi-18-16.model"
    val satisfyModel = XGBClassifierModel.load(satisfyModelPath, num_classes)
    val satisfyDF = df.join(feePredictDF, "手机号码")
    satisfyModel.setFeaturesCol(Array("年龄", "性别", "用户入网时间", "用户星级", "voice", "mobile",
    "fee"))
    satisfyModel.setPredictionCol("satisify")
    val satisfyPredictDF = satisfyModel.transform(satisfyDF).select("手机号码", "satisify", "fee", "voice", "mobile")
    satisfyPredictDF.count()
    println("End...")
  }
}