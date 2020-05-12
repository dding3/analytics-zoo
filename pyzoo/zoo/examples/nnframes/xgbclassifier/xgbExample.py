from zoo.common.nncontext import *
from zoo.pipeline.nnframes import *

from optparse import OptionParser
import sys

def inference(csv_path, model_path, num_classes):
    from pyspark.sql import SparkSession

    spark = SparkSession \
        .builder \
        .getOrCreate()
    df = spark.read.csv(csv_path, sep=",", inferSchema=True, header=True)

    voiceModelPath = model_path + "xgb_yuyin-18-16.model"
    voiceModel = XGBClassifierModel.loadModel(voiceModelPath, num_classes)
    voiceModel.setFeaturesCol(['分公司名称', '用户入网时间', '用户状态', '年龄', '性别', '用户星级',
                               '是否集团成员', '城市农村用户', '是否欠费用户', '主套餐费用', '通话费用', '通话费用趋势', 'VoLTE掉话率', 'ESRVCC切换时延',
                               'ESRVCC切换比例', 'ESRVCC切换成功率', 'VoLTE接续时长', '呼叫建立时长', 'VoLTE接通率',
                               '全程呼叫成功率', 'VoLTE掉话率_diff',
                               'ESRVCC切换时延_diff', 'ESRVCC切换比例_diff', 'ESRVCC切换成功率_diff',
                               'VoLTE接续时长_diff', '呼叫建立时长_diff', 'VoLTE接通率_diff', '全程呼叫成功率_diff'])
    voiceModel.setPredictionCol("voice")
    voicePredictDF = voiceModel.transform(df).select("voice", "手机号码")

    mobileModelPath = model_path + "xgb_shouji-18-16.model"
    mobileModel = XGBClassifierModel.loadModel(mobileModelPath, num_classes)
    mobileModel.setFeaturesCol(['分公司名称', '用户入网时间', '用户状态', '年龄', '性别', '用户星级',
                                '是否集团成员', '城市农村用户', '主套餐费用', '流量费用',
                                '流量费用趋势',
                                '网页响应成功率', '网页响应时延', '网页显示成功率', '网页浏览成功率', '网页打开时长',
                                '视频响应成功率', '视频响应时延', '视频平均每次播放卡顿次数', '视频播放成功率', '视频播放等待时长',
                                '即时通信接入成功率', '即时通信接入时延', '下载速率', '上传速率',
                                '网页响应成功率_diff', '网页响应时延_diff', '网页显示成功率_diff', '网页浏览成功率_diff',
                                '网页打开时长_diff', '视频响应成功率_diff', '视频响应时延_diff', '视频平均每次播放卡顿次数_diff',
                                '视频播放成功率_diff', '视频播放等待时长_diff',
                                '即时通信接入成功率_diff', '即时通信接入时延_diff', '下载速率_diff', '上传速率_diff'])
    mobileModel.setPredictionCol("mobile")
    mobilePredictDF = mobileModel.transform(df).select("mobile", "手机号码")

    feeModelPath = model_path + "xgb_zifei-18-16.model"
    feeModel = XGBClassifierModel.loadModel(feeModelPath, num_classes)
    feeDF = df.join(mobilePredictDF, "手机号码").join(voicePredictDF, "手机号码")
    feeDF.select('年龄', '性别', '用户入网时间', '用户星级', '是否集团成员', '城市农村用户', '是否欠费用户',
                 '主套餐费用', '超套费用', '通话费用', '通话费用趋势', '流量费用', '流量费用趋势',
                 '近3月的平均出账费用', '近3月的平均出账费用趋势', '近3月超套平均', '近3月月均欠费金额', '用户状态', '分公司名称',
                 "voice", "mobile")
    feeModel.setFeaturesCol(['年龄', '性别', '用户入网时间', '用户星级', '是否集团成员', '城市农村用户', '是否欠费用户',
                             '主套餐费用', '超套费用', '通话费用', '通话费用趋势', '流量费用', '流量费用趋势',
                             '近3月的平均出账费用', '近3月的平均出账费用趋势', '近3月超套平均', '近3月月均欠费金额', '用户状态', '分公司名称', "mobile", "voice"])
    feeModel.setPredictionCol("fee")
    feePredictDF = feeModel.transform(feeDF).select("fee", "voice", "mobile", "手机号码")

    satisfyModelPath = model_path + "xgb_manyi-18-16.model"
    satisfyModel = XGBClassifierModel.loadModel(satisfyModelPath, num_classes)
    satisfyDF = df.join(feePredictDF, "手机号码")
    satisfyModel.setFeaturesCol(['年龄', '性别', '用户入网时间', '用户星级', "voice", "mobile",
                                 "fee"])
    satisfyModel.setPredictionCol("satisify")
    satisfyPredictDF = satisfyModel.transform(satisfyDF) \
        .select("手机号码", "satisify", "fee", "voice", "mobile")

    return satisfyPredictDF

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-m", dest="model_path",
                      help="Required. pretrained model path.")
    parser.add_option("-f", dest="file_path",
                      help="training data path.")
    parser.add_option("-n", dest="num_classes",
                      help="number of classification classes.")

    (options, args) = parser.parse_args(sys.argv)

    if not options.model_path:
        parser.print_help()
        parser.error('model_path is required')

    if not options.file_path:
        parser.print_help()
        parser.error('file_path is required')

    file_path = options.file_path
    model_path = options.model_path
    num_classes = int(options.num_classes)


predictionDF = inference(file_path, model_path, num_classes)
predictionDF.count()

print("finished...")