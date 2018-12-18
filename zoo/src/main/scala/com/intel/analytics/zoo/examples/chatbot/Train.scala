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

package com.intel.analytics.zoo.examples.chatbot

import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.dataset.text.utils.SentenceToken
import com.intel.analytics.bigdl.dataset.text._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.nn.{InternalClassNLLCriterion, LookupTable, RandomUniform, TimeDistributedMaskCriterion}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Adam, OptimMethod}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.zoo.common.{NNContext, ZooDictionary}
import com.intel.analytics.zoo.models.seq2seq.{Decoder, Encoder, Seq2seq}
import com.intel.analytics.zoo.pipeline.api.keras.layers.internal.InternalMax
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import org.apache.log4j.{Level, Logger}

import scala.collection.Iterator
import scala.io.Source
import scala.reflect.ClassTag

object Train {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  import Utils._

  val logger = Logger.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {
      val sc = NNContext.initNNContext("Chatbot Example")

      val idx2w = Source
        .fromFile(param.dataFolder + "idx2w.csv", "UTF-8")
        .getLines
        .map(x => {
          val split = x.split(",")
          (split(0).toInt, if (split.length < 2) "" else split(1))
        })
        .toMap

      val w2idx = Source
        .fromFile(param.dataFolder + "w2idx.csv", "UTF-8")
        .getLines
        .map(x => {
          val split = x.split(",")
          (split(0), split(1).toInt)
        })
        .toMap

      val dictionary = new ZooDictionary(idx2w, w2idx)
      val vocabSize = dictionary.getVocabSize()
      // we are sure _ will not used in the training data so we can use it as padding
      val padId = dictionary.getIndex("_") + 1

      val chat1 = Source
        .fromFile(param.dataFolder + "chat1_1.txt", "UTF-8")
        .getLines
        .toList
        .map(_.split(",").map(_.toInt))

      val chat2List = Source
        .fromFile(param.dataFolder + "chat2_1.txt", "UTF-8")
        .getLines
        .toList
        .toIterator

      val chat2 = SentenceIdxBiPadding(dictionary = dictionary)
        .apply(chat2List).map(_.split(",").map(_.toInt))
        .toList

      val tokens = sc.parallelize(chat1.zip(chat2))

      val trainRDD = tokens

      val trainSet = trainRDD
        .map(chatIdxToLabeledChat(_))
        .map(labeledChatToSample(_))

//      RNG.setSeed(50)
      val stdv = 1.0 / math.sqrt(param.embedDim)

//      val embEnc =
//        new Embedding(vocabSize, param.embedDim, maskZero = true,
//          paddingValue = padId, init = RandomUniform(-stdv, stdv))
//      val embDec =
//        new Embedding(vocabSize, param.embedDim, maskZero = true,
//          paddingValue = padId, init = RandomUniform(-stdv, stdv))
//      val embEncW = embEnc.parameters()._1
//      val embDecW = embDec.parameters()._1
//      val embEncG = embEnc.parameters()._2
//      val embDecG = embDec.parameters()._2
//      for (i <- 0 until embEncW.size) {
//        embEncW(i).set(embDecW(i))
//        embEncG(i).set(embDecG(i))
//      }

      val wInit = RandomUniform(-stdv, stdv)
      val bInit = RandomUniform(-stdv, stdv)

      val enclookuptable = LookupTable(
        vocabSize,
        param.embedDim,
        paddingValue = padId,
        maskZero = true
      ).setInitMethod(wInit, bInit)

      val declookuptable = LookupTable(
        vocabSize,
        param.embedDim,
        paddingValue = padId,
        maskZero = true
      ).setInitMethod(wInit, bInit)

      declookuptable.weight.set(enclookuptable.weight)
      declookuptable.gradWeight.set(enclookuptable.gradWeight)

      val embEnc =
        new KerasLayerWrapper[Float](enclookuptable
          .asInstanceOf[AbstractModule[Activity, Activity, Float]])
        .asInstanceOf[KerasLayer[Tensor[Float], Tensor[Float], Float]]

      val embDec =
        new KerasLayerWrapper[Float](declookuptable
          .asInstanceOf[AbstractModule[Activity, Activity, Float]])
          .asInstanceOf[KerasLayer[Tensor[Float], Tensor[Float], Float]]

      val encoder = Encoder[Float]("lstm", 3, param.embedDim,
        embEnc)
      val decoder = Decoder[Float]("lstm", 3, param.embedDim,
        embDec)

      val generator = Sequential[Float]()
      // need fix in checkWithCurrentInputShape KerasLayer.scala to take account -1
      generator.add(TimeDistributed[Float](Dense(vocabSize),
        Shape(Array(2, param.embedDim))))
      generator.add(TimeDistributed[Float](Activation("log_softmax")))

      val model = Seq2seq(encoder, decoder, Shape(Array(-1)),
        Shape(Array(-1)), generator = generator)

      val optimMethod = if (param.stateSnapshot.isDefined) {
        OptimMethod.load[Float](param.stateSnapshot.get)
      } else {
        new Adam[Float](learningRate = 0.0001)
      }

      model.compile(
        optimizer = optimMethod,
        loss = TimeDistributedMaskCriterion(
          InternalClassNLLCriterion(paddingValue = padId),
          paddingValue = padId
        ))
      if (param.checkpoint.isDefined) {
        model.setCheckPoint(param.checkpoint.get, param.overWriteCheckpoint)
      }

      val seeds = Array("happy birthday have a nice day",
        "donald trump won last nights presidential debate according to snap online polls")
//val seeds = Array("happy I am light rain")

      var i = 1
      while (i <= param.nEpochs) {
        model.fit(
          trainSet, batchSize = param.batchSize,
          nbEpoch = i)

        for (seed <- seeds) {
          println("Query> " + seed)
          val evenToken = SentenceTokenizer().apply(Array(seed).toIterator).toArray
          val oddToken = (SentenceBiPadding() -> SentenceTokenizer())
            .apply(Array("").toIterator).toArray
          val labeledChat = evenToken.zip(oddToken)
            .map(buildInferInput(dictionary, _)).apply(0)

          val sent1 = Tensor(Storage(labeledChat._1), 1, Array(1, labeledChat._1.length))
          val sent2 = Tensor(Storage(labeledChat._2), 1, Array(1, labeledChat._2.length))

          val featDim = 3
          val end = dictionary.getIndex(SentenceToken.end) + 1
          val endSign = Tensor(Array(end.toFloat), Array(1))

          val layers = new KerasLayerWrapper[Float](
            InternalMax(dim = featDim, returnValue = false)
              .asInstanceOf[AbstractModule[Activity, Activity, Float]])
            .asInstanceOf[KerasLayer[Tensor[Float], Tensor[Float], Float]]

          val infer = model.inference(T(sent1, sent2), maxSeqLen = 30,
            stopSign = endSign, buildOutput = layers)

          val predArray = new Array[Float](infer.nElement())
          Array.copy(infer.storage().array(), infer.storageOffset() - 1,
            predArray, 0, infer.nElement())
          val result = predArray.grouped(infer.size(Seq2seq.timeDim)).toArray[Array[Float]]
            .map(x => x.map(t => dictionary.getWord(t - 1)))
          println(result.map(x => x.mkString(" ")).mkString("\n"))
        }
        model.clearState()
        i += 1
      }
      sc.stop()
    })
  }


  def buildInferInput[T: ClassTag](
    dictionary: Dictionary,
    chat: (Array[String], Array[String]))(implicit ev: TensorNumeric[T])
  : (Array[T], Array[T]) = {
//    val (indices1, indices2) =
//      (chat._1.map(x => ev.fromType[Int](dictionary.getIndex(x))),
//        chat._2.map(x => ev.fromType[Int](dictionary.getIndex(x))))
//    (indices1, indices2.take(indices2.length - 1))

    val (indices1, indices2) =
      (chat._1.map(x => ev.fromType[Int](dictionary.getIndex(x) + 1)),
        chat._2.map(x => ev.fromType[Int](dictionary.getIndex(x) + 1)))
    (indices1, indices2.take(indices2.length - 1))
  }

  def chatIdxToLabeledChat[T: ClassTag](
    chat: (Array[Int], Array[Int]))(implicit ev: TensorNumeric[T])
  : (Array[T], Array[T], Array[T]) = {
//    val evOne = ev.fromType(1)
//    val (indices1, indices2) =
//    (chat._1.map(x => ev.fromType[Int](x)),
//      chat._2.map(x => ev.fromType[Int](x)))
//    // raw data is 0 based, and we need change it to 1 based.
//    // Embeddding will add 1 for feature, we need manually add 1 for label
//    val label = indices2.drop(1).clone().map(x => ev.plus(x, evOne))
//    (indices1, indices2.take(indices2.length - 1), label)
    val (indices1, indices2) =
    (chat._1.map(x => ev.fromType[Int](x + 1)),
      chat._2.map(x => ev.fromType[Int](x + 1)))
    val label = indices2.drop(1)
    (indices1, indices2.take(indices2.length - 1), label)
  }

  class SentenceIdxBiPadding(
    start: Option[String] = None,
    end: Option[String] = None,
    dictionary: Dictionary
  )
    extends Transformer[String, String] {

    val sentenceStart = dictionary.getIndex(start.getOrElse(SentenceToken.start))
    val sentenceEnd = dictionary.getIndex(end.getOrElse(SentenceToken.end))

    override def apply(prev: Iterator[String]): Iterator[String] = {
      prev.map(x => {
        val index = x.indexOf(",0")
        val sentence = if (index == -1) sentenceStart + "," + x + "," + sentenceEnd
        else sentenceStart + "," + x.slice(0, index) + "," + sentenceEnd + x.slice(index, x.length)
        sentence
      })
    }
  }

  object SentenceIdxBiPadding {
    def apply(
      start: Option[String] = None,
      end: Option[String] = None,
      dictionary: Dictionary
    ): SentenceIdxBiPadding = new SentenceIdxBiPadding(start, end, dictionary)
  }


  def labeledChatToSample[T: ClassTag](
    labeledChat: (Array[T], Array[T], Array[T]))
    (implicit ev: TensorNumeric[T]): Sample[T] = {

    val sentence1: Tensor[T] = Tensor(Storage(labeledChat._1))
    val sentence2: Tensor[T] = Tensor(Storage(labeledChat._2))
    val label: Tensor[T] = Tensor(Storage(labeledChat._3))

    Sample(featureTensors = Array(sentence1, sentence2), labelTensor = label)
  }
}
