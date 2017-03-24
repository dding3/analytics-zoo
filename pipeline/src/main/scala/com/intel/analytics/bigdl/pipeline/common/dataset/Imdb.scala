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

import com.intel.analytics.bigdl.pipeline.common.dataset.roiimage.RoiImagePath

abstract class Imdb(val cacheFolder: String = "data/cache") {
  val name: String
  val classes: Array[String]
  var roidb: Array[RoiImagePath] = _
  protected var imageIndex: Array[String] = _

  def getRoidb(useFlipped: Boolean): Array[RoiImagePath] = {
    if (roidb != null && roidb.length > 0) return roidb
    roidb = loadRoidb
    if (useFlipped) {
      appendFlippedImages()
    }
    roidb
  }

  protected def loadRoidb: Array[RoiImagePath]

  def numClasses: Int = classes.length

  def numImages: Int = imageIndex.length

  protected def imagePathAt(i: Int): String

  def imagePathFromIndex(index: String): String

  def appendFlippedImages(): Unit = throw new UnsupportedOperationException

  def loadAnnotation(index: String): RoiImagePath
}


object Imdb {
  /**
   * Get an imdb (image database) by name
   * @param name
   * @param devkitPath
   * @return
   */
  def getImdb(name: String, devkitPath: String): Imdb = {
    val items = name.split("_")
    if (items.length != 3) throw new Exception("dataset name error")
    if (items(0) == "voc") {
      new PascalVoc(items(1), items(2), devkitPath)
    } else {
      throw new UnsupportedOperationException("unsupported dataset")
    }
  }
}
