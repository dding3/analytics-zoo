/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.bigdl.pipeline.common.dataset

import java.io.File
import java.nio.file.Paths

import com.intel.analytics.bigdl.pipeline.common.dataset.roiimage.{RoiImagePath, Target}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

import scala.Array._
import scala.io.Source
import scala.xml.XML

/**
 * Parse the pascal voc dataset, load images and annotations
 * @param year       the year of dataset
 * @param imageSet   train, val, test, etc
 * @param devkitPath dataset folder
 */
class PascalVoc(val year: String = "2007", val imageSet: String,
  devkitPath: String)
  extends Imdb {
  override val name = "voc_" + year + "_" + imageSet
  val dataPath = Paths.get(devkitPath, "/VOC" + year).toString
  override val classes = PascalVoc.classes
  val classToInd = (classes zip (Stream from 1).map(_.toFloat)).toMap
  imageIndex = loadImageSetIndex()

  assert(new File(devkitPath).exists(),
    "VOCdevkit path does not exist: " + devkitPath)
  assert(new File(dataPath).exists(),
    "Path does not exist: {}" + dataPath)

  /**
   * Return the absolute path to image i in the image sequence.
   * @param i seq index
   * @return image path
   */
  def imagePathAt(i: Int): String = imagePathFromIndex(imageIndex(i))

  /**
   * Construct an image path from the image"s "index" identifier.
   * @param index e.g. 000001
   * @return image path
   */
  def imagePathFromIndex(index: String): String = s"$dataPath/JPEGImages/$index.jpg"


  /**
   * Load the indexes listed in this dataset's image set file.
   *
   * Example path to image set file: devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
   * @return
   */
  def loadImageSetIndex(): Array[String] = {
    val imageSetFile = Paths.get(dataPath, s"/ImageSets/Main/$imageSet.txt").toFile
    assert(imageSetFile.exists(), "Path does not exist " + imageSetFile.getAbsolutePath)
    Source.fromFile(imageSetFile).getLines().map(line => line.trim).toArray
  }

  def annotationPath(index: String): String = {
    dataPath + "/Annotations/" + index + ".xml"
  }

  /**
   * Load image and bounding boxes info from XML file in the PASCAL VOC
   * format.
   */
  def loadAnnotation(index: String): RoiImagePath = {
    val annot = PascalVoc.loadAnnotation(annotationPath(index))
    val classes = annot.classNames.map(classToInd)
    val gtClasses = Tensor(Storage(classes ++ annot.difficults)).resize(2, classes.length)
    RoiImagePath(imagePathFromIndex(index), Target(gtClasses, annot.boxes))
  }

  /**
   * This function loads/saves from/to a cache file to speed up future calls.
   * @return the database of ground-truth regions of interest.
   */
  def loadRoidb: Array[RoiImagePath] = {
    roidb = imageIndex.map(index => loadAnnotation(index))
    roidb
  }
}

object PascalVoc {
  val classes = Array[String](
    "__background__", // always index 0
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
  )

  def loadAnnotation(path: String): ImageObjects = {
    val xml = XML.loadFile(path)
    val objs = xml \\ "object"

    val boxes = Tensor[Float](objs.length, 4)
    val boxesArr = boxes.storage().array()
    val classNames = new Array[String](objs.length)
    val difficults = new Array[Float](objs.length)
    // Load object bounding boxes into a data frame.
    var ix = 0
    while (ix < objs.length) {
      val obj = objs(ix)
      val bndbox = obj \ "bndbox"
      boxesArr(ix * 4) = (bndbox \ "xmin").text.toFloat
      boxesArr(ix * 4 + 1) = (bndbox \ "ymin").text.toFloat
      boxesArr(ix * 4 + 2) = (bndbox \ "xmax").text.toFloat
      boxesArr(ix * 4 + 3) = (bndbox \ "ymax").text.toFloat
      classNames(ix) = (obj \ "name").text
      difficults(ix) = (obj \ "difficult").text.toFloat
      ix += 1
    }
    ImageObjects(classNames, boxes, difficults)
  }

  /**
   * Image objects that describe the annotations
   * @param classNames class names N
   * @param difficults whether objects are difficult N
   * @param boxes      bounding boxes N * 4
   */
  case class ImageObjects(classNames: Array[String],
    boxes: Tensor[Float], difficults: Array[Float]) {
    require(classNames.length == difficults.length && classNames.length == boxes.size(1)
      && boxes.size(2) == 4)
  }

}
