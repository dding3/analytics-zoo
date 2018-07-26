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

package com.intel.analytics.bigdl.nn

import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.tensor.Tensor

import scala.math._

class InternalClassNLLCriterionSpec extends FlatSpec with Matchers {
  "A ClassNLL Criterion with sizeAverage False" should "generate correct output and grad" in {
    val criterion = new InternalClassNLLCriterion[Double](null)
    val input = Tensor[Double](3, 3)
    input(Array(1, 1)) = -1.10821131127
    input(Array(1, 2)) = -0.92179085988591
    input(Array(1, 3)) = -1.3017876357682
    input(Array(2, 1)) = -0.72992115377362
    input(Array(2, 2)) = -1.2817109257719
    input(Array(2, 3)) = -1.4250730090114
    input(Array(3, 1)) = -1.1074577039332
    input(Array(3, 2)) = -1.0506933510994
    input(Array(3, 3)) = -1.1397251596433
    val target = Tensor[Double](3)
    target(Array(1)) = 1
    target(Array(2)) = 2
    target(Array(3)) = 3
    val expectedOutput = 3.5296473966852
    val expectedGrad = Tensor[Double](3, 3)
    expectedGrad(Array(1, 1)) = -1
    expectedGrad(Array(1, 2)) = 0
    expectedGrad(Array(1, 3)) = 0
    expectedGrad(Array(2, 1)) = 0
    expectedGrad(Array(2, 2)) = -1
    expectedGrad(Array(2, 3)) = 0
    expectedGrad(Array(3, 1)) = 0
    expectedGrad(Array(3, 2)) = 0
    expectedGrad(Array(3, 3)) = -1
    val output = criterion.forward(input, target)
    val gradInput = criterion.backward(input, target)
    assert(abs(expectedOutput - output) < 1e-6)
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
  }

  "A ClassNLL Criterion with weight" should "generate correct output and grad" in {
    val weight = Tensor[Double](3)
    weight(Array(1)) = 0.35054216370918
    weight(Array(2)) = 0.76185464672744
    weight(Array(3)) = 0.66953149507754
    val criterion = new InternalClassNLLCriterion[Double](weight)
    val input = Tensor[Double](3, 3)
    input(Array(1, 1)) = -1.1894985426003
    input(Array(1, 2)) = -1.1789041748521
    input(Array(1, 3)) = -0.94672288864566
    input(Array(2, 1)) = -0.70491562360676
    input(Array(2, 2)) = -1.3937761580642
    input(Array(2, 3)) = -1.3559084361956
    input(Array(3, 1)) = -1.0404241993415
    input(Array(3, 2)) = -1.0287857984981
    input(Array(3, 3)) = -1.240448289816
    val target = Tensor[Double](3)
    target(Array(1)) = 1
    target(Array(2)) = 2
    target(Array(3)) = 3
    val expectedOutput = 2.309343433418
    val expectedGrad = Tensor[Double](3, 3)
    expectedGrad(Array(1, 1)) = -0.35054216370918
    expectedGrad(Array(1, 2)) = 0
    expectedGrad(Array(1, 3)) = 0
    expectedGrad(Array(2, 1)) = 0
    expectedGrad(Array(2, 2)) = -0.76185464672744
    expectedGrad(Array(2, 3)) = 0
    expectedGrad(Array(3, 1)) = 0
    expectedGrad(Array(3, 2)) = 0
    expectedGrad(Array(3, 3)) = -0.66953149507754
    val output = criterion.forward(input, target)
    val gradInput = criterion.backward(input, target)
    assert(abs(expectedOutput - output) < 1e-6)
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
  }
}
