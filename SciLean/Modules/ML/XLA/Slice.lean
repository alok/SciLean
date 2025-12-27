import SciLean.Modules.ML.XLA.XlaTensorIndex
import SciLean.VersoPrelude
import Lean

namespace SciLean

/-!
# StableHLO function: {lit}`slice`

Spec: {lit}`https://github.com/openxla/stablehlo/blob/main/docs/spec.md`

## Semantics

Extracts a slice from the {lit}`operand` using statically-computed starting indices
and produces a {lit}`result` tensor. {lit}`start_indices` contain the starting indices of
the slice for each dimension, {lit}`limit_indices` contain the ending indices
(exclusive) for the slice for each dimension, and {lit}`strides` contain the strides
for each dimension.

More formally, {lit}`result[result_index] = operand[operand_index]` where
{lit}`operand_index = start_indices + result_index * strides`.

## Inputs

| Label | Name              | Type                                         | Constraints      |
|-------|-------------------|----------------------------------------------|------------------|
| (I1)  | {lit}`operand`       | tensor or per-tensor quantized tensor        | (C1-C3), (C5)    |
| (I2)  | {lit}`start_indices` | 1-dimensional tensor constant of type {lit}`si64` | (C2), (C3), (C5) |
| (I3)  | {lit}`limit_indices` | 1-dimensional tensor constant of type {lit}`si64` | (C2), (C3), (C5) |
| (I4)  | {lit}`strides`       | 1-dimensional tensor constant of type {lit}`si64` | (C2), (C4)       |

## Outputs

| Name       | Type                                  | Constraints |
|------------|---------------------------------------|-------------|
| {lit}`result` | tensor or per-tensor quantized tensor | (C1), (C5)  |

## Constraints

* (C1) {lit}`element_type(operand) = element_type(result)`.
* (C2) {lit}`size(start_indices) = size(limit_indices) = size(strides) = rank(operand)`.
* (C3) {lit}`0 <= start_indices <= limit_indices <= shape(operand)`.
* (C4) {lit}`0 < strides`.
* (C5) {lit}`shape(result) = ceil((limit_indices - start_indices) / strides)`.

## Examples

```nonLeanCode
// %operand: [
//            [0, 0, 0, 0],
//            [0, 0, 1, 1],
//            [0, 0, 1, 1]
//           ]
%result = "stablehlo.slice"(%operand) {
  start_indices = array<i64: 1, 2>,
  limit_indices = array<i64: 3, 4>,
  strides = array<i64: 1, 1>
} : (tensor<3x4xi64>) -> tensor<2x2xi64>
// % result: [
//            [1, 1],
//            [1, 1]
//           ]
```

More Examples: {lit}`https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/slice.mlir`
-/

namespace Slice

def slice.outShape {r}
    (start_indices limit_indices : ArrayN ℤ r)
    (strides : ArrayN ℕ+ r) : ArrayN ℤ r :=
      .ofFn fun i => Rat.ceil <|
        (Rat.ofInt (limit_indices[i] - start_indices[i])) / (Rat.ofInt (Int.ofNat (strides[i])))
        -- this should be equal to which might be easier to reason about
        -- (limit_indices[i] - start_indices[i] + (strides[i] - 1)).fdiv strides[i]

structure Args {r} (inDims : Dims r) where
  start_indices : ArrayN ℤ r
  limit_indices : ArrayN ℤ r
  strides : ArrayN ℕ+ r

def Args.outShape {r} {inDims : Dims r} (args : Args inDims) : Dims r :=
      .ofFn fun i => Rat.ceil <|
        (Rat.ofInt (args.limit_indices[i] - args.start_indices[i])) / (Rat.ofInt (Int.ofNat (args.strides[i])))
        -- this should be equal to which might be easier to reason about
        -- (limit_indices[i] - start_indices[i] + (strides[i] - 1)).fdiv strides[i]

structure Preconditions {r} {inDims : Dims r} (args : Args inDims) where
  c1 : True
  c2 : args.start_indices.size = r ∧ args.limit_indices.size = r ∧ args.strides.size = r ∧ inDims.size = r
  c3 : ∀ d : Fin r, 0 ≤ args.start_indices[d] ∧ args.start_indices[d] ≤ args.limit_indices[d]
  c4 : True

structure Postconditions {r} {inDims : Dims r} (args : Args inDims) (outDims : Dims r) where
  c5 : outDims = args.outShape

end Slice

open Slice in
def slice
    {r} {inDims outDims : Dims r}
    (operand : XlaTensorIndex inDims → R)
    (args : Args inDims)
    (_h : Preconditions args)
    (_houtDims : outDims = args.outShape := by infer_var) :
    XlaTensorIndex outDims → R :=
  fun result_index =>
    let operand_index := args.start_indices + result_index.1 * args.strides
    operand ⟨operand_index, by sorry_proof⟩
