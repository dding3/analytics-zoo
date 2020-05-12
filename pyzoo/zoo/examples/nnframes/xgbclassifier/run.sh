export ANALYTICS_ZOO_HOME=/root/dingding/analytics-zoo
export SPARK_HOME=/root/dingding/spark-2.4.3-bin-hadoop2.7

${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
 --master yarn \
 --deploy-mode client \
 --driver-memory 20g \
 --executor-memory 70g \
 --executor-cores 32 \
 --num-executors 5 \
 --py-files ${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_0.10.0-spark_2.4.3-0.8.0-SNAPSHOT-python-api.zip \
xgbExample.py -m modelPath -f dataPath -n 3
