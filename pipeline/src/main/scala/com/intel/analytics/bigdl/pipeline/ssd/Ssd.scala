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

package com.intel.analytics.bigdl.pipeline.ssd

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.pipeline.ssd.Ssd._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.pipeline.common.nn.{PriorBox, Normalize => SSDNormalize}
import com.intel.analytics.bigdl.numeric.NumericFloat

/**
 *
 * @param nInput
 * @param nPreds
 * @param minSizes
 * @param maxSizes
 * @param aspectRatios
 * @param isFlip
 * @param isClip
 * @param variances
 * @param step
 */
case class ComponetParam(nInput: Int, nPreds: Int, minSizes: Array[Float],
  maxSizes: Array[Float], aspectRatios: Array[Float], isFlip: Boolean,
  isClip: Boolean, variances: Array[Float], step: Int = 0)

class Ssd(numClasses: Int, resolution: Int,
  basePart1: Sequential[Float], basePart2: Sequential[Float],
  params: Map[String, ComponetParam],
  isLastPool: Boolean, normScale: Float) {

  def apply(): Module[Float] = {
    val model = Sequential()
    model.add(ConcatTable()
      .add(Sequential()
        .add(basePart1)
        .add(ConcatTable()
          .add(SSDNormalize(2, scale = normScale).setName("conv4_3_norm"))
          .add(basePart2
            .add(SpatialDilatedConvolution(params("conv4_3_norm").nInput,
              1024, 3, 3, 1, 1, 6, 6, 6, 6).setName("fc6"))
            .add(ReLU(true).setName("relu6"))
            .add(SpatialConvolution(1024, 1024, 1, 1).setName("fc7"))
            .add(ReLU(true).setName("relu7")))))
      .add(Identity()))

    val conv4_3NormMboxPriorbox =
      Sequential()
        .add(ConcatTable()
          .add(selectTensor(1, 1).setName("conv4_3_norm_out"))
          .add(SelectTable(2).setName("data")))
        .add(getPriorBox("conv4_3_norm"))

    val conv4_3data = selectTensor(1, 1)


    def fc7data = selectTensor(1, 2)
    val fc7MboxPriorbox =
      Sequential()
        .add(ConcatTable()
          .add(fc7data)
          .add(SelectTable(2).setName("data")))
        .add(getPriorBox("fc7"))



    val featureComponents = ConcatTable()
      .add(getConcatOutput(conv4_3data, "conv4_3_norm", params("conv4_3_norm").nInput,
        params("conv4_3_norm").nPreds, conv4_3NormMboxPriorbox))
      .add(getConcatOutput(fc7data, "fc7", 1024, params("fc7").nPreds, fc7MboxPriorbox))

    model.add(featureComponents)

    val com6 = addFeatureComponent6(featureComponents)
    val com7 = addFeatureComponent7(com6)
    val com8 = addFeatureComponent8(com7)
    if (isLastPool) {
      addFeatureComponentPool6(com8)
    } else {
      val com9 = addFeatureComponent9(com8)
      if (resolution == 512) {
        addFeatureComponent10(com9)
      }
    }
    concatResults(model, numClasses)
  }

  def selectResults(start: Int, dim: Int, nInputDims: Int, name: String): Module[Float] = {
    val con = ConcatTable()
    var i = start
    while (i <= 18) {
      con.add(SelectTable(i))
      i += 3
    }
    Sequential().add(con).add(JoinTable(dim, nInputDims).setName(name))
  }

  def addComponet(component: Sequential[Float], conv: Sequential[Float], name: String)
  : ConcatTable[Float] = {
    val param = params(name)
    // connect neighbor feature components
    val connection = ConcatTable()
      .add(getConcatOutput(SelectTable[Float](1), name, param.nInput, param.nPreds,
        getPriorBox(name)))

    component
      .add(ParallelTable()
        .add(conv)
        .add(Identity().setName("data")))
      .add(connection)
    connection
  }


  def getConcatOutput(input: Module[Float], name: String, nInput: Int, numPreds: Int,
    priorBox: Module[Float]): ConcatTable[Float] = {
    ConcatTable()
      .add(getLocConfComponent(input, name, nInput, numPreds * numClasses, "conf"))
      .add(getLocConfComponent(input, name, nInput, numPreds * 4, "loc"))
      .add(priorBox)
  }


  def getLocConfComponent(input: Module[Float], name: String, nInput: Int,
    nOutput: Int, typeName: String): Sequential[Float] = {
    Sequential()
      .add(input)
      .add(SpatialConvolution(nInput, nOutput, 3, 3, 1, 1, 1, 1)
        .setName(s"${ name }_mbox_$typeName"))
      .add(Transpose(Array((2, 3), (3, 4))).setName(s"${ name }_mbox_${ typeName }_perm"))
      .add(InferReshape(Array(0, -1)).setName(s"${ name }_mbox_${ typeName }_flat"))
  }


  /**
   * select tensor from nested tables
   * @param depths a serious of depth to use when fetching certain tensor
   * @return a wanted tensor
   */
  def selectTensor(depths: Int*): Sequential[Float] = {
    val module = Sequential()
    depths.slice(0, depths.length - 1).foreach(depth =>
      module.add(SelectTable(depth)))
    module.add(SelectTable(depths(depths.length - 1)))
  }

  def concatResults(model: Sequential[Float], numClasses: Int): Module[Float] = {
    model
      .add(FlattenTable())
      .add(ConcatTable()
        .add(selectResults(2, 1, 1, "mbox_loc"))
        .add(Sequential()
          .add(selectResults(1, 1, 1, "mbox_conf"))
          .add(InferReshape(Array(0, -1, numClasses)).setName("mbox_conf_reshape"))
          .add(TimeDistributed[Float](SoftMax()).setName("mbox_conf_softmax"))
          .add(InferReshape(Array(0, -1)).setName("mbox_conf_flatten")))
        .add(selectResults(3, 2, 2, "mbox_priorbox")))
    model
  }

  def addFeatureComponent6(connection: ConcatTable[Float])
  : ConcatTable[Float] = {
    val component = Sequential()
    connection.add(component)
    val fc6 = Sequential()
    addConvRelu(fc6, (1024, 256, 1, 1, 0), "6_1")
    addConvRelu(fc6, (256, 512, 3, 2, 1), "6_2")
    component.add(ConcatTable()
      .add(selectTensor(1, 2))
      .add(SelectTable(2)))
    addComponet(component, fc6, "conv6_2")
  }

  def addFeatureComponent7(connection: ConcatTable[Float])
  : ConcatTable[Float] = {
    val component = Sequential()
    connection.add(component)
    val c7 = Sequential()
    addConvRelu(c7, (512, 128, 1, 1, 0), "7_1")
    addConvRelu(c7, (128, 256, 3, 2, 1), "7_2")

    addComponet(component, c7, "conv7_2")
  }

  def addFeatureComponent8(connection: ConcatTable[Float])
  : ConcatTable[Float] = {
    val component = Sequential()
    connection.add(component)
    val c8 = Sequential()
    addConvRelu(c8, (256, 128, 1, 1, 0), "8_1")
    if (isLastPool || resolution == 512) {
      addConvRelu(c8, (128, 256, 3, 2, 1), "8_2")
    } else {
      addConvRelu(c8, (128, 256, 3, 1, 0), "8_2")
    }

    addComponet(component, c8, "conv8_2")
  }

  def addFeatureComponent9(connection: ConcatTable[Float]): ConcatTable[Float] = {
    val component = Sequential()
    connection.add(component)
    val c9 = Sequential()
    addConvRelu(c9, (256, 128, 1, 1, 0), "9_1")
    if (resolution == 512) {
      addConvRelu(c9, (128, 256, 3, 2, 1), "9_2")
    } else {
      addConvRelu(c9, (128, 256, 3, 1, 0), "9_2")
    }

    val name = "conv9_2"
    if (resolution == 300) {
      val param = params(name)
      component.add(Identity())
        .add(ParallelTable()
          .add(c9)
          .add(Identity().setName("data")))
        .add(getConcatOutput(SelectTable[Float](1), name, param.nInput,
          param.nPreds, getPriorBox(name)))
      null
    } else {
      addComponet(component, c9, name)
    }
  }

  def addFeatureComponent10(connection: ConcatTable[Float]): Unit = {
    val component = Sequential()
    connection.add(component)
    val c10 = Sequential()
    addConvRelu(c10, (256, 128, 1, 1, 0), "10_1")
    addConvRelu(c10, (128, 256, 4, 1, 1), "10_2")

    val name = "conv10_2"
    require(params.contains(name))
    val param = params(name)
    component.add(Identity())
      .add(ParallelTable()
        .add(c10)
        .add(Identity().setName("data")))
      .add(getConcatOutput(SelectTable[Float](1), name, param.nInput,
        param.nPreds, getPriorBox(name)))
  }

  def addFeatureComponentPool6(connection: ConcatTable[Float])
  : Unit = {
    val component = Sequential()
    connection.add(component)

    val name = "pool6"
    val pool6priorbox = getPriorBox(name)

    val param = params(name)

    component.add(Identity())
      .add(ParallelTable()
        .add(SpatialAveragePooling(3, 3).setName(name))
        .add(Identity().setName("data")))
      .add(getConcatOutput(SelectTable[Float](1), name,
        param.nInput, param.nPreds, pool6priorbox))
  }

  private def getPriorBox(name: String): PriorBox[Float] = {
    val param = params(name)
    PriorBox[Float](minSizes = param.minSizes, maxSizes = param.maxSizes,
      _aspectRatios = param.aspectRatios, isFlip = param.isFlip, isClip = param.isClip,
      variances = param.variances, step = param.step, offset = 0.5f)
      .setName(s"${ name }_mbox_priorbox")
  }

}

object Ssd {
  def apply(numClasses: Int, resolution: Int,
    basePart1: Sequential[Float], basePart2: Sequential[Float],
    params: Map[String, ComponetParam],
    normScale: Float, isLastPool: Boolean): Module[Float] =
    new Ssd(numClasses, resolution, basePart1, basePart2, params, isLastPool, normScale).apply()

  def addConvRelu(model: Sequential[Float], p: (Int, Int, Int, Int, Int),
    name: String, prefix: String = "conv", nGroup: Int = 1): Unit = {
    model.add(new SpatialConvolution(p._1, p._2, p._3, p._3, p._4, p._4,
      p._5, p._5, nGroup = nGroup).setName(s"$prefix$name"))
    model.add(ReLU(true).setName(s"relu$name"))
  }
}
