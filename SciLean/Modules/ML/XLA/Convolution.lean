import SciLean.Modules.ML.XLA.XlaTensorIndex
import SciLean.VersoPrelude
import SciLean.Modules.ML.XLA.Slice
import SciLean.Modules.ML.XLA.DotGeneral
import SciLean.Modules.ML.XLA.Pad
import SciLean.Modules.ML.XLA.Slice
import SciLean.Modules.ML.XLA.Concatenate
import SciLean.Modules.ML.XLA.Split

namespace SciLean

/-!
# StableHLO function: {lit}`convolution`

Spec: {lit}`https://github.com/openxla/stablehlo/blob/main/docs/spec.md`

This file models the StableHLO {lit}`convolution` operation. See the spec for
full semantics, inputs, and constraints.
-/


namespace Convolution

structure Args {r} (lhsDims rhsDims : Dims r) where
  window_strides : ArrayN ℤ (r-2)
  padding : ArrayN (ℤ×ℤ) (r-2)
  lhs_dilation : ArrayN ℕ+ (r-2)
  rhs_dilation : ArrayN ℕ+ (r-2)
  window_reversal : ArrayN Bool r
  input_batch_dimension : Fin r
  input_feature_dimension : Fin r
  input_spatial_dimensions : ArrayN (Fin r) (r-2)
  kernel_input_feature_dimension : Fin r
  kernel_output_feature_dimension : Fin r
  kernel_spatial_dimensions : ArrayN (Fin r) (r-2)
  output_batch_dimension : Fin r
  output_feature_dimension : Fin r
  output_spatial_dimensions : ArrayN (Fin r) (r-2)
  feature_group_count : ℕ+
  batch_group_count : ℕ+
  precision_config : True

namespace Args

  def lhsSpatialShape {r} {lhsDims rhsDims : Dims r}
    (args : Args lhsDims rhsDims) : Dims (r - 2) := sorry

  def rhsSpatialShape {r} {lhsDims rhsDims : Dims r}
    (args : Args lhsDims rhsDims) : Dims (r - 2) := sorry

  def outSpatialShape {r} {lhsDims rhsDims : Dims r}
    (args : Args lhsDims rhsDims) : Dims (r - 2) := sorry

  def lowPadding {r} {lhsDims rhsDims : Dims r}
    (args : Args lhsDims rhsDims) : ArrayN ℤ (r-2) := args.padding.map (·.1)

  def highPadding {r} {lhsDims rhsDims : Dims r}
    (args : Args lhsDims rhsDims) : ArrayN ℤ (r-2) := args.padding.map (·.2)

  def lhsShapeMap {r} {lhsDims rhsDims : Dims r}
    (args : Args lhsDims rhsDims) :
    α × ArrayN α (r - 2) × α
    ≃
    ArrayN α r := sorry

  def rhsShapeMap {r} {lhsDims rhsDims : Dims r}
    (args : Args lhsDims rhsDims) :
    α × ArrayN α (r - 2) × α
    ≃
    ArrayN α r := sorry

  def outShapeMap {r} {lhsDims rhsDims : Dims r}
    (args : Args lhsDims rhsDims) :
    α × ArrayN α (r - 2) × α
    ≃
    ArrayN α r := sorry

  def outDims {r} {lhsDims rhsDims : Dims r}
      (args : Args lhsDims rhsDims) : Dims r :=

    let output_batch_dim_size := lhsDims[args.input_batch_dimension] / args.batch_group_count
    let output_feature_dim_size := rhsDims[args.kernel_output_feature_dimension]

    let dilated_input_shape := (args.lhsSpatialShape - 1) * args.lhs_dilation + 1
    let padded_input_shape := args.lowPadding + dilated_input_shape + args.highPadding
    let dilated_window_shape := (args.rhsSpatialShape - 1) * args.rhs_dilation + 1
    let is_empty_window := padded_input_shape ≤ 0 || dilated_window_shape > padded_input_shape
    let output_spatial_dims := if is_empty_window then 0 else (padded_input_shape - dilated_window_shape) / args.window_strides + 1

    args.outShapeMap (output_batch_dim_size, output_spatial_dims, output_feature_dim_size)

end Args

structure Conditions {r} {lhsDims rhsDims : Dims r}
    (args : Args lhsDims rhsDims) (outDims : Dims r) : Prop where
  c1 : True


end Convolution

variable {R} [RealScalar R]

-- case
open Convolution in
def convolutionCore {r} {lhsDims rhsDims outDims : Dims r}
    (lhs : XlaTensorIndex lhsDims → R) (rhs : XlaTensorIndex rhsDims → R)
    (args : Args lhsDims rhsDims)
    (_h : Conditions args outDims) :
    XlaTensorIndex outDims → R :=

  fun i =>
    let (_,output_spatial_index,_) := args.outShapeMap.symm i.1 -- get the correct parts of `i`

    let lhsWindowShape :=
      args.lhsShapeMap (lhsDims[args.input_batch_dimension],
                        args.rhsSpatialShape,
                        lhsDims[args.input_feature_dimension])
    let lhs_window_strides := args.lhsShapeMap (1,args.window_strides,1)
    let lhs_padding_low  := args.lhsShapeMap (0,args.lowPadding,0)
    let lhs_padding_high := args.lhsShapeMap (0,args.highPadding,0)
    let lhs_base_dilation := args.lhsShapeMap (1,args.lhs_dilation,1)
    let lhs_window_dilations := args.lhsShapeMap (1,args.rhs_dilation,1)

    let padArgs : Pad.Args lhsDims := {
      edge_padding_low := lhs_padding_low
      edge_padding_high := lhs_padding_high
      interior_padding := lhs_base_dilation.toNat
    }
    let padded_lhs := pad lhs 0 padArgs
      -- there is some issue with elaboration and we have to specify these arguments explicitly
      (outDims:= padArgs.outShape) (by infer_var)


    let lhs_window_start : ArrayN ℤ r := args.lhsShapeMap (0,output_spatial_index,0)
    let sliceArgs : Slice.Args padArgs.outShape := {
      start_indices := lhs_window_start
      limit_indices := (lhs_window_start + lhsWindowShape)
      strides := lhs_window_dilations
    }
    let lhs_window := slice padded_lhs sliceArgs sorry
      -- there is some issue with elaboration and we have to specify these arguments explicitly
      (outDims:=sliceArgs.outShape) (by infer_var)

    let dotArgs : DotGeneral.Args sliceArgs.outShape rhsDims :=
        {lhs_batching_dimensions := #[]
         rhs_batching_dimensions := #[]
         lhs_contracting_dimensions := args.input_spatial_dimensions.1 ++ #[args.input_feature_dimension]
         rhs_contracting_dimensions := args.kernel_spatial_dimensions.1 ++ #[args.kernel_input_feature_dimension]}

    let dot_product : R :=
      dot_general lhs_window rhs dotArgs
        (t := 0) (outDims := ⟨#[],by simp⟩) (by sorry) (by sorry) ⟨⟨#[],by simp⟩,by simp⟩

    dot_product


open Convolution in
def convolution {r} {lhsDims rhsDims outDims : Dims r}
    (lhs : XlaTensorIndex lhsDims → R) (rhs : XlaTensorIndex rhsDims → R)
    (args : Args lhsDims rhsDims)
    (_h : Conditions args outDims) : XlaTensorIndex outDims → R :=
  convolutionCore lhs rhs args (by trivial)
