import SciLean.Modules.ML.XLA.XlaTensorIndex
import SciLean.VersoPrelude

/-!
# StableHLO function: {lit}`concatenate`

## Semantics

Concatenates {lit}`inputs` along {lit}`dimension` in the same order as the given
arguments and produces a {lit}`result` tensor. More formally,
{lit}`result[i0, ..., id, ..., iR-1] = inputs[k][i0, ..., kd, ..., iR-1]`, where:

1. {lit}`id = d0 + ... + dk-1 + kd`.
1. {lit}`d` is equal to {lit}`dimension`, and {lit}`d0`, ... are {lit}`d`th dimension sizes
   of {lit}`inputs`.

## Inputs

| Label | Name           | Type                                                       | Constraints      |
|-------|----------------|------------------------------------------------------------|------------------|
| (I1)  | {lit}`inputs`    | variadic number of tensors or per-tensor quantized tensors | (C1-C6)          |
| (I2)  | {lit}`dimension` | constant of type {lit}`si64`                                    | (C2), (C4), (C6) |

## Outputs

| Name       | Type                                  | Constraints |
|------------|---------------------------------------|-------------|
| {lit}`result` | tensor or per-tensor quantized tensor | (C5-C6)     |

## Constraints

* (C1) {lit}`same(element_type(inputs...))`.
* (C2) {lit}`same(shape(inputs...))` except for {lit}`dim(inputs..., dimension)`.
* (C3) {lit}`0 < size(inputs)`.
* (C4) {lit}`0 <= dimension < rank(inputs[0])`.
* (C5) {lit}`element_type(result) = element_type(inputs[0])`.
* (C6) {lit}`shape(result) = shape(inputs[0])` except for:
  * {lit}`dim(result, dimension) = dim(inputs[0], dimension) + ...`.

## Examples

```nonLeanCode
// %input0: [[1, 2], [3, 4], [5, 6]]
// %input1: [[7, 8]]
%result = "stablehlo.concatenate"(%input0, %input1) {
  dimension = 0 : i64
} : (tensor<3x2xi64>, tensor<1x2xi64>) -> tensor<4x2xi64>
// %result: [[1, 2], [3, 4], [5, 6], [7, 8]]
```

More Examples: {lit}`https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/concatenate.mlir`
-/

namespace SciLean


namespace Concatenate

structure Args {r k} (inDims : Fin k → Dims r) where
  dimension : Fin r

def Args.outShape {r k} {inDims : Fin k → Dims r} (args : Args inDims) : Dims r :=
  .ofFn fun d =>
    if d = args.dimension then
      ∑ i, (inDims i)[d]
    else if h : 0 < k then
      (inDims ⟨0, by linarith⟩)[d]
    else
      0

structure Preconditions {r k} {inDims : Fin k → Dims r} (args : Args inDims) : Prop where
  c1 : True
  c2 : True
  c3 : True
  c4 : True
  c5 : True

structure Postconditions {r k} {inDims : Fin k → Dims r} (args : Args inDims) (outDims : Dims r) : Prop where
  c6 : ∀ d,
    if d = args.dimension then
      outDims[d] = ∑ i, (inDims i)[d]
    else
      ∀ i, outDims[d] = (inDims i)[d]

theorem postconditions_true {r k} {inDims : Fin k → Dims r} (args : Args inDims) (_h : Preconditions args) :
    Postconditions args args.outShape := by
  sorry_proof

def Args.indexMap {r k} {inDims : Fin k → Dims r} (args : Args inDims)
  (h : Preconditions args)
  {outDims} (houtShape : outDims = args.outShape := by infer_var) :
  (i : Fin k) × XlaTensorIndex (inDims i)
  ≃
  XlaTensorIndex outDims := sorry


end Concatenate


open Concatenate in
def concatenate {r k} {inDims : Fin k → Dims r} {outDims : Dims r}
    (inputs : (i : Fin k) → XlaTensorIndex (inDims i) → R)
    (args : Args inDims)
    (h : Preconditions args)
    (houtDims : outDims = args.outShape := by first | (try simp_all) | (try infer_var)) :
    XlaTensorIndex outDims → R :=
  fun i =>
    let ⟨i,j⟩ := (args.indexMap h (by simp_all)).symm i
    inputs i j
