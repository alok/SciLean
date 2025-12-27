import SciLean.Modules.ML.XLA.XlaTensorIndex
import SciLean.VersoPrelude

namespace SciLean

/-!
# StableHLO function: {lit}`transpose`

Spec: {lit}`https://github.com/openxla/stablehlo/blob/main/docs/spec.md`

## Semantics

Permutes the dimensions of the {lit}`operand` tensor using {lit}`permutation` and
produces the {lit}`result` tensor. More formally, {lit}`result[result_index] = operand[operand_index]`
where {lit}`result_index[d] = operand_index[permutation[d]]`.

## Inputs

| Label | Name              | Type                                         | Constraints |
|-------|-------------------|----------------------------------------------|-------------|
| (I1)  | {lit}`operand`     | tensor or quantized tensor                   | (C1-C4)     |
| (I2)  | {lit}`permutation` | 1-dimensional tensor constant of type {lit}`si64` | (C2-C4) |

## Outputs

| Name       | Type                       | Constraints   |
|------------|----------------------------|---------------|
| {lit}`result` | tensor or quantized tensor | (C1), (C3-C4) |

## Constraints

* (C1) {lit}`element_type(result)` is given by:
  * {lit}`element_type(operand)`, if {lit}`!is_per_axis_quantized(operand)`.
  * {lit}`element_type(operand)` except that {lit}`quantization_dimension(operand)` and
    {lit}`quantization_dimension(result)` may differ, otherwise.
* (C2) {lit}`permutation` is a permutation of {lit}`range(rank(operand))`.
* (C3) {lit}`shape(result) = dim(operand, permutation...)`.
* (C4) If {lit}`is_per_axis_quantized(result)`, then
  {lit}`quantization_dimension(operand) = permutation(quantization_dimension(result))`.

## Examples

```nonLeanCode
// %operand: [
//            [[1,2], [3,4], [5,6]],
//            [[7,8], [9,10], [11,12]]
//           ]
%result = "stablehlo.transpose"(%operand) {
  permutation = array<i64: 2, 1, 0>
} : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
// %result: [
//           [[1,7], [3,9], [5,11]],
//           [[2,8], [4,10], [6,12]]
//          ]
```

More Examples: {lit}`https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/transpose.mlir`
-/


def transpose.outDims {r}
    (_permutation : ArrayN ℤ r)
    (inDims : Dims r) : Dims r := inDims

def transpose {r} {inDims outDims : Dims r}
    (operand : XlaTensorIndex inDims → R)
    (_permutation : ArrayN ℤ r)
    (_houtDims : outDims = transpose.outDims _permutation inDims := by infer_var) :
    XlaTensorIndex outDims → R :=
  fun result_index =>
    operand ⟨result_index.val, by sorry_proof⟩
end SciLean
