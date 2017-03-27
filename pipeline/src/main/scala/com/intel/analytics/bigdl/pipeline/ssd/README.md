# SSD: Single Shot MultiBox Detector

[SSD](https://research.google.com/pubs/pub44872.html) is one of the state-of-the-art
 object detection pipeline.

Currently Bigdl has added support for ssd with vgg and alexnet as base model,
with 300*300 or 516*516 input resolution.

## Prepare the dataset

Download the test dataset

```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```

Extract all of these tars into one directory named ```VOCdevkit```

```
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
```

It should have this basic structure

```
$VOCdevkit/                           # development kit
$VOCdevkit/VOCcode/                   # VOC utility code
$VOCdevkit/VOC2007                    # image sets, annotations, etc.
# ... and several other directories ...
```

## Generate Sequence Files

Get the bigdl.sh script 
```
wget https://raw.githubusercontent.com/intel-analytics/BigDL/master/scripts/bigdl.sh
```

### convert labeled pascal voc dataset

```bash
dist/bin/bigdl.sh --
java -cp pipeline-0.1-SNAPSHOT-jar-with-dependencies.jar:spark-assembly-1.5.1-hadoop2.6.0.jar \
         com.intel.analytics.bigdl.pipeline.common.dataset.RoiImageSeqGenerator \
     -f $VOCdevkit -o output -i voc_2007_test
```

where ```-f``` is your devkit folder, ```-o``` is the output folder, and ```-i``` is the imageset name.

note that a spark jar is needed as dependency.

### convert unlabeled image folder
```bash
dist/bin/bigdl.sh --
java -cp pipeline-0.1-SNAPSHOT-jar-with-dependencies.jar:spark-assembly-1.5.1-hadoop2.6.0.jar \
         com.intel.analytics.bigdl.pipeline.common.dataset.RoiImageSeqGenerator \
     -f imageFolder -o output
```

where ```-f``` is your image folder, ```-o``` is the output folder

## Download pretrained model

https://github.com/weiliu89/caffe/tree/ssd#models

07+12: SSD300, SSD512

## Run the demo
Example command for running in Spark cluster (yarn)

```
dist/bin/bigdl.sh -- spark-submit \
--master yarn \
--deploy-mode client \
--executor-cores 28 \
--num-executors 2 \
--driver-memory 128g \
--executor-memory 128g \
--class com.intel.analytics.bigdl.pipeline.ssd.example.Demo \
pipeline-0.1-SNAPSHOT-jar-with-dependencies.jar \
-f $demoDataFolder \
--folderType seq \
-o demotestall \
--caffeDefPath data/models/VGGNet/VOC0712/SSD_300x300/test.prototxt \
--caffeModelPath data/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel \
-t vgg16 \
--classname data/models/VGGNet/VOC0712/classname.txt \
-v false \
-b 4 \
-r 300
```

In the above commands

* -f: where you put your demo data
* --folderType: It can be seq/local
* -o: where you put your demo output data
* --caffeDefPath: caffe network definition prototxt file path
* --caffeModelPath: caffe serialized model file path
* -t: network type, it can be vgg16 or alexnet
* --classname: file that store detection class names, one line for one class
* -v: whether to visualize detections
* -b: batch size
* -r: input resolution, 300 or 512

## Run the test

```
dist/bin/bigdl.sh -- spark-submit \
--master yarn \
--deploy-mode client \
--executor-cores 28 \
--num-executors 2 \
--driver-memory 128g \
--executor-memory 128g \
--class com.intel.analytics.bigdl.pipeline.ssd.example.Test \
pipeline-0.1-SNAPSHOT-jar-with-dependencies.jar \
-f $voc_test_data \
--caffeDefPath data/models/VGGNet/VOC0712/SSD_300x300/test.prototxt \
--caffeModelPath data/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel \
-t vgg16 \
--nclass 21 \
-i voc_2007_test \
-b 4 \
-r 300
```

In the above commands

* -f: where you put your demo data
* --caffeDefPath: caffe network definition prototxt file path
* --caffeModelPath: caffe serialized model file path
* -t: network type, it can be vgg16 or alexnet
* --nclass: number of detection classes
* -i: image set name with the format ```voc_${year}_${imageset}```, e.g. voc_2007_test
* -b: batch size
* -r: input resolution, 300 or 512