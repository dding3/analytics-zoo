/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.{nn => bnn}
import com.intel.analytics.bigdl.nn.RandomNormal
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{MultiShape, Shape}
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.autograd.{AutoGrad, Constant, Parameter, Variable}
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.models.{Model, Sequential}

import scala.reflect.ClassTag

/**
 * [[TransformerLayer]] A self attention keras like layer
 * Input is a Tensor with shape [batch, seqLen, 2].
 * [:, :, 1] represents token id
 * [:, :, 2] represents postions in the sentence
 * Output is a Tensor which output the states of Transformer layer
 * @param nBlock block number
 * @param hiddenPDrop drop probability of projection
 * @param attnPDrop drop probability of attention
 * @param nHead head number
 * @param maskAttention whether unidirectional or bidirectional
 * @param outputAllBlock whether output all blocks' output
 * @param embeddingLayer embedding layer
 * @param inputShape input shape, default is null
 */
class TransformerLayer[T: ClassTag](
  val nBlock: Int,
  val hiddenPDrop: Double,
  val attnPDrop: Double,
  val nHead: Int,
  val maskAttention: Boolean,
  val outputAllBlock: Boolean,
  val embeddingLayer: KerasLayer[Activity, Tensor[T], T],
  var inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Activity, Activity, T](KerasUtils.addBatch(inputShape))
    with Net {

  var seqLen: Int = 0
  override def doBuild(inputShape: Shape): AbstractModule[Activity, Activity, T] = {
    require(inputShape.isInstanceOf[MultiShape], "TransformerLayer input must" +
      " be a multiple shape")
    val _inputShape = KerasUtils.removeBatch(inputShape).toMulti()

    val wordInput = Variable(_inputShape(0))
    val positionInput = Variable(_inputShape(1))

    seqLen = _inputShape.head.toSingle().head

    require(embeddingLayer.isInstanceOf[Net], "use layers from" +
      "com.intel.analytics.zoo.pipeline.api.keras and operators from" +
      " com.intel.analytics.zoo.pipeline.api.autograd to construct the embedding layer")
    val embedding = embeddingLayer.asInstanceOf[Net]
    val e = embedding.from(wordInput, positionInput)
    val hiddenSize = e.getOutputShape().toSingle().last

    val nextInput: Variable[T] = e
    val modelOutputSize = nBlock
    val modelOutput = new Array[Variable[T]](modelOutputSize)
    modelOutput(0) = block(nextInput, hiddenSize)

    for (i <- 1 until nBlock) {
      val output = block(modelOutput(i - 1), hiddenSize)
      modelOutput(i) = output
    }

    val model = if (outputAllBlock) {
      Model(Array(wordInput, positionInput), modelOutput)
    } else Model(Array(wordInput, positionInput), modelOutput.last)

    model.asInstanceOf[AbstractModule[Activity, Activity, T]]
  }

  def block(x: Variable[T], hiddenSize: Int, attention_mask: Variable[T] = null,
            eplision: Double = 1e-5): Variable[T] = {
    // g, b for layerNorm
    val g = Parameter[T](Shape(1, hiddenSize),
      initWeight = Tensor.ones[T](hiddenSize).view(1, hiddenSize))
    val b = Parameter[T](Shape(1, hiddenSize),
      initWeight = Tensor[T](hiddenSize).view(1, hiddenSize))

    // g, b for layerNorm
    val g2 = Parameter[T](Shape(1, hiddenSize),
      initWeight = Tensor.ones[T](hiddenSize).view(1, hiddenSize))
    val b2 = Parameter[T](Shape(1, hiddenSize),
      initWeight = Tensor[T](hiddenSize).view(1, hiddenSize))
    val a = multiHeadSelfAttention(x, hiddenSize, attention_mask)
    val n = TransformerLayer.layerNorm(x + a, eplision, weight = g, bias = b)
    val m = mlp(n, hiddenSize)
    val h = TransformerLayer.layerNorm(n + m, eplision, weight = g2, bias = b2)
    h
  }

  def mlp(x: Variable[T], hiddenSize: Int): Variable[T] = {
    val h = new Convolution1D(hiddenSize * 4, 1, init = RandomNormal(0.0, 0.02)).from(x)
    val a = gelu(h)
    val h2 = new Convolution1D(hiddenSize, 1, init = RandomNormal(0.0, 0.02)).from(a)
    Dropout(hiddenPDrop).from(h2)
  }

  def multiHeadSelfAttention(x: Variable[T], hiddenSize: Int,
    attention_mask: Variable[T] = null): Variable[T] = {
    val c = new Convolution1D(hiddenSize * 3, 1, init = RandomNormal(0.0, 0.02)).from(x)
    val query = c.slice(2, 0, hiddenSize)
    val key = c.slice(2, hiddenSize, hiddenSize)
    val value = c.slice(2, hiddenSize * 2, hiddenSize)
    val q = splitHeads(query, nHead)
    val k = splitHeads(key, nHead, k = true)
    val v = splitHeads(value, nHead)
    val a = attn(q, k, v, true) // m: (batch, nhead, seqLen, hiddenSize/nhead)
    val m = mergeHeads(a) // m: (batch, seqLen, hiddenSize)
    val n = new Convolution1D(hiddenSize, 1, init = RandomNormal(0.0, 0.02))
      .from(m) // n: (batch, seqLen, hiddenSize)
    Dropout(hiddenPDrop).from(n)
  }

  lazy val maskValue = if (maskAttention) {
    val data = KerasUtils.tril(Tensor.ones(seqLen, seqLen)).view(1, seqLen, seqLen)
    new Constant[T](data)
  } else null

  // scale shoule be set in Attention
  def attn(q: Variable[T], k: Variable[T], v: Variable[T], scale: Boolean = false,
           attention_mask: Variable[T] = null): Variable[T] = {
    // q, v:(batch, nHead, seqLen, hiddenSize/nHead)
    // k:(batch, nHead, hiddenSize/nHead, seqLen)
    var w = AutoGrad.mm(q, k) // w: (batch, nHead, seqLen, seqLen)
    if (scale) w = w / scala.math.sqrt(v.getOutputShape().toSingle().toArray.last)

    // mask attention
    if (maskAttention) {
      if (attention_mask != null) {
        w = w + attention_mask
      } else {
        w = w * maskValue + (maskValue * (-1) + 1) * -1e9
      }
    }

    w = Activation[Float]("softmax").from(w)
    w = Dropout(attnPDrop).from(w)

    AutoGrad.mm(w, v)
  }

  def gelu(x: Variable[T]): Variable[T] = {
    x * 0.5 * (Activation("tanh").from((AutoGrad.square(x) * x * 0.044715 + x)
      * (scala.math.sqrt(2 / scala.math.Pi))) + 1)
  }

  def splitHeads(x: Variable[T], nHead: Int, k: Boolean = false): Variable[T] = {
    val sizes = x.getOutputShape().toSingle().toArray
    val newSizes = sizes.drop(1).dropRight(1) ++ Array(nHead, sizes.last / nHead)
    val r = Reshape(newSizes).from(x)
    if (k) Permute(Array(2, 3, 1)).from(r)
    else Permute(Array(2, 1, 3)).from(r)
  }

  def mergeHeads(x: Variable[T]): Variable[T] = {
    val p = AutoGrad.contiguous(Permute[T](Array(2, 1, 3)).from(x))
    val sizes = p.getOutputShape().toSingle().toArray
    Reshape(sizes.drop(1).dropRight(2) ++ Array(sizes.last * sizes(sizes.length - 2))).from(p)
  }
}

object TransformerLayer {
  /**
   * [[TransformerLayer]] A self attention keras like layer
   * @param vocab vocabulary size of training data, default is 40990
   * @param seqLen max sequence length of training data, default is 77
   * @param nBlock block number, default is 12
   * @param residPdrop drop probability of projection, default is 0.1
   * @param attnPdrop drop probability of attention, default is 0.1
   * @param nHead head number, default is 12
   * @param hiddenSize is also embedding size
   * @param embeddingDrop drop probability of embedding layer, default is 0.1
   * @param maskAttention whether unidirectional or bidirectional, default is true(unidirectional)
   * @param outputAllBlock whether output all blocks' output
   */
  def apply[@specialized(Float, Double) T: ClassTag](
    vocab: Int = 40990,
    seqLen: Int = 77,
    nBlock: Int = 12,
    residPdrop: Double = 0.1,
    attnPdrop: Double = 0.1,
    nHead: Int = 12,
    hiddenSize: Int = 768,
    embeddingDrop: Double = 0,
    maskAttention: Boolean = true,
    outputAllBlock: Boolean = true)(implicit ev: TensorNumeric[T]): TransformerLayer[T] = {
    require(hiddenSize > 0, "hiddenSize must be great" +
      "than 0 with default embedding layer")
    val wordInput = InputLayer[T](inputShape = Shape(seqLen))
    val postionInput = InputLayer[T](inputShape = Shape(seqLen))
    val embeddingLayer = Sequential[T]()
      .add(Merge(layers = List(wordInput, postionInput), mode = "concat"))
      .add(Reshape(Array(seqLen * 2)))
      .add(Embedding(vocab, hiddenSize, inputLength = seqLen * 2))
      .add(Dropout(embeddingDrop))
      .add(Reshape(Array(seqLen, 2, hiddenSize)))
      .add(new KerasLayerWrapper[T](bnn.Sum[T](dimension = 3,
        squeeze = true).asInstanceOf[AbstractModule[Activity, Activity, T]]))
    new TransformerLayer[T](nBlock,
      residPdrop, attnPdrop, nHead, maskAttention, outputAllBlock,
      embeddingLayer.asInstanceOf[KerasLayer[Activity, Tensor[T], T]])
  }

  /**
   * [[TransformerLayer]] A self attention keras like layer
   * @param nBlock block number
   * @param residPdrop drop probability of projection
   * @param attnPdrop drop probability of attention
   * @param nHead head number
   * @param maskAttention whether unidirectional or bidirectional
   * @param outputAllBlock whether output all blocks' output
   * @param embeddingLayer embedding layer
   */
  def apply[@specialized(Float, Double) T: ClassTag](
    nBlock: Int,
    residPdrop: Double,
    attnPdrop: Double,
    nHead: Int,
    maskAttention: Boolean,
    outputAllBlock: Boolean,
    embeddingLayer: KerasLayer[Activity, Tensor[T], T])
    (implicit ev: TensorNumeric[T]): TransformerLayer[T] = {
    new TransformerLayer[T](nBlock, residPdrop, attnPdrop, nHead,
      maskAttention, outputAllBlock, embeddingLayer = embeddingLayer)
  }

  def layerNorm[@specialized(Float, Double) T: ClassTag](x: Variable[T],
    e: Double = 1e-5, weight: Parameter[T], bias: Parameter[T])
    (implicit ev: TensorNumeric[T]): Variable[T] = {
    val sizes = x.getOutputShape().toSingle().toArray
    val u = AutoGrad.mean(x, sizes.size - 1, true)
    val s = AutoGrad.mean(AutoGrad.square(x - u), sizes.size -1, true)
    val y = (x - u) / AutoGrad.sqrt(s + e)
    y * weight + bias
  }
}
