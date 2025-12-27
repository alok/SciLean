import SciLean.Modules.ML.XLA.XlaTensorIndex
import SciLean.VersoPrelude
import SciLean.Meta.GenerateFunTrans
import SciLean.Meta.GenerateFunProp
import SciLean.Analysis.AdjointSpace.Basic
import SciLean.Analysis.Scalar.Basic

import Mathlib.Tactic.ProxyType

namespace SciLean

/-! StableHLO function: {lit}`pad`

Spec: (source: https://github.com/openxla/stablehlo/blob/main/docs/spec.md)

# pad

## Semantics

Expands {lit}`operand` by padding around the tensor as well as between the elements
of the tensor with the given {lit}`padding_value`.

{lit}`edge_padding_low` and {lit}`edge_padding_high` specify the amount of padding added
at the low-end (next to index 0) and the high-end (next to the highest index) of
each dimension respectively. The amount of padding can be negative, where the
absolute value of negative padding indicates the number of elements to remove
from the specified dimension.

{lit}`interior_padding` specifies the amount of padding added between any two
elements in each dimension which may not be negative. Interior padding occurs
before edge padding such that negative edge padding will remove elements from
the interior-padded operand.

More formally, {lit}`result[result_index]` is defined as:

* {lit}`operand[operand_index]` if
  {lit}`result_index = edge_padding_low + operand_index * (interior_padding + 1)`.
* {lit}`padding_value` otherwise.

## Inputs

| Label | Name                | Type                                                | Constraints      |
|-------|---------------------|-----------------------------------------------------|------------------|
| (I1)  | {lit}`operand`           | tensor or per-tensor quantized tensor               | (C1), (C2), (C4) |
| (I2)  | {lit}`padding_value`     | 0-dimensional tensor or per-tensor quantized tensor | (C1)             |
| (I3)  | {lit}`edge_padding_low`  | 1-dimensional tensor constant of type {lit}`si64`   | (C1), (C4)       |
| (I4)  | {lit}`edge_padding_high` | 1-dimensional tensor constant of type {lit}`si64`   | (C1), (C4)       |
| (I5)  | {lit}`interior_padding`  | 1-dimensional tensor constant of type {lit}`si64`   | (C2-C4)          |

## Outputs

| Name     | Type                                  | Constraints |
|----------|---------------------------------------|-------------|
| {lit}`result` | tensor or per-tensor quantized tensor | (C3-C6) |

## Constraints

* (C1) {lit}`element_type(operand) = element_type(padding_value) = element_type(result)`.
* (C2) {lit}`size(edge_padding_low) = size(edge_padding_high) = size(interior_padding) = rank(operand)`.
* (C3) {lit}`0 <= interior_padding`.
* (C4) {lit}`shape(result) = shape(operand) + edge_padding_low + max(shape(operand) - 1, 0) * interior_padding + edge_padding_high`.

## Examples

```nonLeanCode
// %operand: [
//            [1, 2, 3],
//            [4, 5, 6]
//           ]
// %padding_value: 0
%result = "stablehlo.pad"(%operand, %padding_value) {
  edge_padding_low = array<i64: 0, 1>,
  edge_padding_high = array<i64: 2, 1>,
  interior_padding = array<i64: 1, 2>
} : (tensor<2x3xi32>, tensor<i32>) -> tensor<5x9xi32>
// %result: [
//           [0, 1, 0, 0, 2, 0, 0, 3, 0],
//           [0, 0, 0, 0, 0, 0, 0, 0, 0],
//           [0, 4, 0, 0, 5, 0, 0, 6, 0],
//           [0, 0, 0, 0, 0, 0, 0, 0, 0],
//           [0, 0, 0, 0, 0, 0, 0, 0, 0]
//          ]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/pad.mlir)


-/



namespace Pad

structure Args {r} (inDims : Dims r) where
  edge_padding_low  : ArrayN ℤ r
  edge_padding_high : ArrayN ℤ r
  interior_padding : ArrayN ℕ r

def Args.outShape {r} {inDims : Dims r} (args : Args inDims) : Dims r :=
  inDims
  + args.edge_padding_low
  + ((inDims - 1) ⊔ 0) * args.interior_padding.toInt
  + args.edge_padding_high


@[ext]
structure Conditions {r} {inDims : Dims r} (args : Args inDims) (outDims : Dims r) : Prop where
  c1 : True
  c2 : args.edge_padding_low.size = r
       ∧ args.edge_padding_high.size = r
       ∧ args.interior_padding.size = r
  c3 : 0 ≤ args.interior_padding
  c4 : outDims = inDims
         + args.edge_padding_low
         + ((inDims - 1) ⊔ 0) * args.interior_padding.toInt
         + args.edge_padding_high


noncomputable instance {r} {inDims outDims : Dims r} (args : Args inDims) : Decidable (Conditions args outDims) := by
  classical
  infer_instance


end Pad

def pad
    {R} [RealScalar R]
    {r} {inDims outDims : Dims r}
    (operand : XlaTensorIndex inDims → R)
    (padding_value : R)
    (args : Pad.Args inDims)
    (_houtDims : outDims = args.outShape := by infer_var) :
    XlaTensorIndex outDims → R :=

  fun i =>
    let di := (i.1 - args.edge_padding_low) / args.interior_padding.toInt
    let ri := (i.1 - args.edge_padding_low) % args.interior_padding.toInt
    if h : (0 ≤ di ∧ di < inDims) ∧ (ri = 0) then
      operand ⟨di, by sorry_proof⟩
    else
      padding_value


def_fun_prop pad in operand padding_value
  with_transitive
  : IsContinuousLinearMap R by unfold pad; fun_prop


-- Can we express this function by a XLA function?
def pad.arg_operand.adjoint
    {R} [RealScalar R]
    {r} {inDims outDims : Dims r}
    (output : XlaTensorIndex outDims → R)
    (args : Pad.Args inDims)
    (_houtDims : outDims = args.outShape) :
    XlaTensorIndex inDims → R :=
  fun di =>
    let i := di.1 * args.interior_padding.toInt + args.edge_padding_low
    if h : (0 ≤ i ∧ i < outDims) then
      output ⟨i, by sorry_proof⟩
    else
      0


-- Can we express this function by a XLA function?
-- Probably dot_general?
def pad.arg_padding_value.adjoint
    {R} [RealScalar R]
    {r} {inDims outDims : Dims r}
    (output : XlaTensorIndex outDims → R)
    (args : Pad.Args inDims)
    (_houtDims : outDims = args.outShape) :
    R :=
  ∑ i : XlaTensorIndex outDims,
    let di := (i.1 - args.edge_padding_low) / args.interior_padding.toInt
    let ri := (i.1 - args.edge_padding_low) % args.interior_padding.toInt
    if ¬((0 ≤ di ∧ di < inDims) ∧ (ri = 0)) then
      output i
    else
      0


@[fun_trans]
theorem pad.arg_operandpadding_value.adjoint_rule
    {R} [RealScalar R]
    {r} {inDims outDims : Dims r}
    (args : Pad.Args inDims)
    (houtDims : outDims = args.outShape) :
    adjoint R (fun xy : (XlaTensorIndex inDims → R)×R => pad xy.1 xy.2 args houtDims)
    =
    fun z =>
      (pad.arg_operand.adjoint z args houtDims,
       pad.arg_padding_value.adjoint z args houtDims) := by
  sorry_proof
