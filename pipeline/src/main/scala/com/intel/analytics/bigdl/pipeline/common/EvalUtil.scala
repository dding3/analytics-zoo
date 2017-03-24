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

package com.intel.analytics.bigdl.pipeline.common

import com.intel.analytics.bigdl.pipeline.common.dataset.roiimage.Target
import com.intel.analytics.bigdl.tensor.Tensor

import scala.collection.mutable.ArrayBuffer

object EvalUtil {

  /**
   * cumulate sum
   * @param arr array of values
   * @return cumulated sums
   */
  private def cumsum(arr: Array[Int]): Array[Int] = {
    var sum = 0
    arr.map { x => sum += x; sum }
  }

  /**
   * Compute VOC AP given precision and recall
   * @param rec         recall
   * @param prec        precision
   * @param use07metric whether use 07 11 point method
   * @return average precision
   */
  private def vocAp(rec: Array[Double], prec: Array[Double], use07metric: Boolean): Double = {
    var ap = 0.0
    if (use07metric) {
      // 11 point metric
      var t = 0.0
      while (t <= 1) {
        val ps = (rec zip prec).filter(x => x._1 > t).map(_._2)
        val p = if (ps.length == 0) 0 else ps.max
        ap = ap + p / 11.0
        t += 0.1
      }
    } else {
      // correct AP calculation
      // first append sentinel values at the end
      val mrec = new Array[Double](rec.length + 2)
      mrec(mrec.length - 1) = 1.0
      rec.copyToArray(mrec, 1)
      val mpre = new Array[Double](prec.length + 2)
      prec.copyToArray(mpre, 1)

      // compute the precision envelope
      var i = mpre.length - 1
      while (i > 0) {
        mpre(i - 1) = Math.max(mpre(i - 1), mpre(i))
        i -= 1
      }
      // to calculate area under PR curve, look for points
      // where X axis (recall) changes value
      val inds = (mrec.slice(1, mrec.length) zip mrec.slice(0, mrec.length - 1)).map(
        x => x._1 != x._2).zipWithIndex.map(x => x._2)

      // and sum (\Delta recall) * prec
      ap = inds.map(i => (mrec(i + 1) - mrec(i)) * mpre(i + 1)).sum
    }
    ap
  }

  def evaluateBatch(results: Array[Array[Target]], gt: Tensor[Float], gtAreas: Tensor[Float],
    clsInd: Int, ovThresh: Double = 0.5,
    use07Metric: Boolean = false): (Int, Array[(Float, Int, Int)]) = {
    require(gt.size(2) == 7)
    // extract gt objects for this class
    val num = results.length
    val labelGts = new Array[(Array[(Int, Float)], Array[Boolean])](num)

    val imgToDetectInds = new ArrayBuffer[(Int, Int)]()

    var npos = 0
    var i = 1
    val labelGtInds = new Array[ArrayBuffer[(Int, Float)]](num)
    // var imgId = -1
    // assume the image ids are labeled from 0 for each batch
    // (imgId, label, diff, x1, y1, x2, y2)
    while (i <= gt.size(1)) {
      val imgId = gt.valueAt(i, 1).toInt
      if (gt.valueAt(i, 2) == clsInd + 1) {
        if (labelGtInds(imgId) == null) {
          labelGtInds(imgId) = new ArrayBuffer[(Int, Float)]()
        }
        if (gt.valueAt(i, 3) == 0) {
          npos += 1
        }
        labelGtInds(imgId).append((i, gt.valueAt(i, 3)))
      }
      i += 1
    }

    i = 0
    while (i < labelGtInds.length) {
      if (labelGtInds(i) != null) {
        val det = new Array[Boolean](labelGtInds(i).length)
        labelGts(i) = (labelGtInds(i).toArray, det)
      }
      i += 1
    }

    var imgInd = 0
    while (imgInd < num) {
      val output = results(imgInd)(clsInd)
      if (output != null && output.classes.nElement() != 0) {
        var i = 1
        while (i <= output.classes.size(1)) {
          imgToDetectInds.append((imgInd, i))
          i += 1
        }
      }
      imgInd += 1
    }

    val gtBoxes = gt.narrow(2, 4, 4)
    val out = imgToDetectInds.map(box => {
      var tp = 0
      var fp = 0
      val labeledGt = labelGts(box._1)
      val (ovmax, jmax) = if (labeledGt == null) (-1f, -1)
      else getMaxOverlap(gtBoxes, gtAreas, labeledGt._1.map(_._1),
        results(box._1)(clsInd).bboxes(box._2))
      if (ovmax > ovThresh) {
        if (labeledGt._1(jmax)._2 == 0) {
          if (!labeledGt._2(jmax)) {
            tp = 1
            labeledGt._2(jmax) = true
          } else {
            fp = 1
          }
        }
      } else {
        fp = 1
      }
      (results(box._1)(clsInd).classes.valueAt(box._2), tp, fp)
    }).toArray
    (npos, out)
  }

  // todo: reuse nms maxoverlap
  private def getMaxOverlap(gtBboxes: Tensor[Float], gtAreas: Tensor[Float], gtInds: Array[Int],
    bbox: Tensor[Float]): (Float, Int) = {
    var maxOverlap = Float.MinValue
    var maxInd = -1
    if (gtInds != null && gtInds.length > 0) {
      var i = 1
      while (i <= gtInds.length) {
        val r = gtInds(i - 1)
        val ixmin = Math.max(gtBboxes.valueAt(r, 1), bbox.valueAt(1))
        val iymin = Math.max(gtBboxes.valueAt(r, 2), bbox.valueAt(2))
        val ixmax = Math.min(gtBboxes.valueAt(r, 3), bbox.valueAt(3))
        val iymax = Math.min(gtBboxes.valueAt(r, 4), bbox.valueAt(4))
        val inter = Math.max(ixmax - ixmin + 1, 0) * Math.max(iymax - iymin + 1, 0)
        val bbArea = (bbox.valueAt(3) - bbox.valueAt(1) + 1f) *
          (bbox.valueAt(4) - bbox.valueAt(2) + 1f)
        val overlap = inter / (gtAreas.valueAt(r) - inter + bbArea)
        if (maxOverlap < overlap) {
          maxOverlap = overlap
          maxInd = i - 1
        }
        i += 1
      }
    }
    (maxOverlap, maxInd)
  }


  def map(results: Array[(Float, Int, Int)], use07Metric: Boolean, npos: Int = -1)
  : Double = {
    val num = if (npos > 0) npos else results.length
    val sortedResults = results.sortBy(-_._1)
    val fp = sortedResults.map(_._3)
    val tp = sortedResults.map(_._2)
    // compute precision recall
    val cumfp = cumsum(fp)
    val cumtp = cumsum(tp)
    val rec = cumtp.map(x => x / num.toDouble)
    // avoid divide by zero in case the first detection matches a difficult
    // ground truth
    val prec = (cumtp zip (cumtp zip cumfp).map(x => x._1 + x._2)
      .map(x => Math.max(x, 2.2204460492503131e-16)))
      .map(x => x._1 / x._2)
    vocAp(rec, prec, use07Metric)
  }
}

