# Demo Setup Guide

## 1 Install Dependency Packages

###  Installation Steps
* (Optional) (Mac) You may want to install Homebrew as a convinient package manager
```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```
* Install Java and Spark
   * Install Java on OSX following the guide https://java.com/en/download/help/mac_install.xml
   * Install Spark on OSX
* (Optional) (Mac) Install Python. Python is shipped with MacOS, but you may want to install updates using Homebrew
```
brew install python
```
* Instal pip - the package manager for python
```
sudo easy_install pip
```
* Install required python libraries using Pip
```
pip install numpy scipy pandas scikit-learn matplotlib seaborn jupyter tensorboard wordcloud 
```
* If you have installed jupyter using pip3, you can use below command to install the python 2 kernel
```
python2 -m pip install ipykernel
python2 -m ipykernel install --user
```
###  Required Package Versions
* Demo is tested to work with  
   * jdk8
   * spark-1.6.0
   * python-2.7.x (pip2.7)
   * tensorboard-1.0.0a4 and tensorboard-1.0.0a5

## 2 Start Jupyter Server and Tensorboard Server

* Create start_tensorboard.sh, copy and paste the contents below.  In current demos, the log files are written into "/tmp/bigdl_summaries" by default . So we set tensorboard logdir to the directory so that it can read the events files and produce charts. If you changed the directory where the logs are written to, change the --logdir settings when starting tensorboard server.
```
tensorboard --logdir=/tmp/bigdl_summaries
```
* Create start_notebook.sh, copy and paste the contents below, and edit SPARK_HOME, BigDL_HOME accordingly. Change other parameter settings as you need. 
```
#!/bin/bash

#setup pathes
SPARK_HOME=/Users/bigdl/spark-1.6.0-bin-hadoop2.6/ 
BigDL_HOME=/Users/bigdl/BigDL                                                                                     

#use local mode or cluster mode
#MASTER=spark://xxxx:7077 
MASTER="local[4]"

PYTHON_API_ZIP_PATH=${BigDL_HOME}/dist/lib/bigdl-0.1.0-SNAPSHOT-python-api.zip
BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar

export PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH
export IPYTHON_OPTS="notebook --notebook-dir=./  --ip=* --no-browser --NotebookApp.token=''"

${SPARK_HOME}/bin/pyspark \
    --master ${MASTER} \
    --driver-cores 4  \
   --driver-memory 10g  \
   --total-executor-cores 4  \
   --executor-cores 1  \
   --executor-memory 10g \
   --conf spark.akka.frameSize=64 \
    --py-files ${PYTHON_API_ZIP_PATH} \
    --jars ${BigDL_JAR_PATH} \
    --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
    --conf spark.executor.extraClassPath=bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar
```
* Put start_notebook.sh and start_tensorboard.sh in home directory and execute them in bash. 
    * the root directory in your notebook view is the directory where you execute the script start_notebook. 
## 3 Run Demo
* Open a browser - Suggest Chrome or Firefox or Safari
* Access notebook client at address http://localhost:8888, open the example ipynb files and execute. 
* Access tensorboard client at address http://localhost:6006 and check the charts

## 4 Trouble Shooting
* Make sure the mac is connected to Internet. If not, I found spark local instance can’t be launch due to a hostname not found error.
