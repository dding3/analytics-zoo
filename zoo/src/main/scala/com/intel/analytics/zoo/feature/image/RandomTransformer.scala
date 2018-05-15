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
package com.intel.analytics.zoo.feature.image

import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature, augmentation}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.zoo.feature.common.ImageProcessing

/**
 * It is a wrapper for transformers to control the transform probability
 * @param transformer transformer to apply randomness
 * @param maxProb max prob
 */
class RandomTransformer(transformer: ImageProcessing, maxProb: Double) extends ImageProcessing {
//  override def transform(prev: ImageFeature): ImageFeature = {
//    if (RNG.uniform(0, 1) < maxProb) {
//      transformer.transform(prev)
//    }
//    prev
//  }
//
//  override def toString: String = {
//    s"Random[${transformer.getClass.getCanonicalName}, $maxProb]"
//  }

  private val internalCrop = augmentation.RandomTransformer(transformer, maxProb)
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalCrop.apply(prev)
  }
}

object RandomTransformer {
  def apply(transformer: ImageProcessing, maxProb: Double): RandomTransformer =
    new RandomTransformer(transformer, maxProb)
}
