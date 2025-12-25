import SciLean.Modules.ML.XLA.XlaTensorIndex
import SciLean.Modules.ML.XLA.ConvWithPadding
import SciLean.VersoPrelude

namespace SciLean

/-!
# StableHLO function: {lit}`dot_general`

Spec: {lit}`https://github.com/openxla/stablehlo/blob/main/docs/spec.md`

This file models the StableHLO {lit}`dot_general` operation. See the spec for
full semantics, inputs, and constraints.
-/


variable {R} [RealScalar R]

namespace DotGeneral

structure Args {r s} (lhsDims : Dims r) (rhsDims : Dims s) where
  lhs_batching_dimensions : Array (Fin r)
  rhs_batching_dimensions : Array (Fin s)
  lhs_contracting_dimensions : Array (Fin r)
  rhs_contracting_dimensions : Array (Fin s)


def Args.outShape' {r s} {lhsDims : Dims r} {rhsDims : Dims s} (args : Args lhsDims rhsDims) : Array ℤ :=
  let lhs_all_dims := List.ofFn (fun i : Fin r => i)
  let lhs_result_dimensions : Array (Fin r) :=
    lhs_all_dims.diff (args.lhs_batching_dimensions ++ args.lhs_contracting_dimensions).toList |>.toArray

  let rhs_all_dims := List.ofFn (fun i : Fin s => i)
  let rhs_result_dimensions : Array (Fin s) :=
    rhs_all_dims.diff (args.rhs_batching_dimensions ++ args.rhs_contracting_dimensions).toList |>.toArray

  let outShape :=
    args.lhs_batching_dimensions.map (lhsDims[·])
    ++
    lhs_result_dimensions.map (lhsDims[·])
    ++
    rhs_result_dimensions.map (rhsDims[·])

  outShape

def Args.outRank {r s} {lhsDims : Dims r} {rhsDims : Dims s} (args : Args lhsDims rhsDims) : ℕ :=
    args.outShape'.size

def Args.outShape {r s} {lhsDims : Dims r} {rhsDims : Dims s} (args : Args lhsDims rhsDims)
    {t} (h : t = args.outRank := by (try simp_all); (try infer_var)) : Dims t :=
  ⟨args.outShape', by simp_all[Args.outRank]⟩

structure Preconditions {r s t} {lhsDims : Dims r} {rhsDims : Dims s} (args : Args lhsDims rhsDims) (outDims : Dims t) where
  c1 : args.lhs_batching_dimensions.size = args.rhs_batching_dimensions.size
  c2 : args.lhs_contracting_dimensions.size = args.rhs_contracting_dimensions.size
  c3 : (args.lhs_batching_dimensions ++ args.lhs_contracting_dimensions).toList.Nodup
  c4 : (args.rhs_batching_dimensions ++ args.rhs_contracting_dimensions).toList.Nodup
  c5 : args.lhs_batching_dimensions.all (fun i => i.val < r)
  c6 : args.lhs_contracting_dimensions.all (fun i => i.val < r)
  c7 : args.rhs_batching_dimensions.all (fun i => i.val < s)
  c8 : args.rhs_contracting_dimensions.all (fun i => i.val < s)
  c9 : args.lhs_batching_dimensions.map (lhsDims[·]) = args.rhs_batching_dimensions.map (rhsDims[·])
  c10 : args.lhs_contracting_dimensions.map (lhsDims[·]) = args.rhs_contracting_dimensions.map (rhsDims[·])
  c11 : True
  c12 :
    let lhs_result_dimensions : Array (Fin r) :=
      (Array.ofFn (fun i : Fin r => i)).filter (fun d => ¬args.lhs_batching_dimensions.contains d ∧ ¬args.lhs_contracting_dimensions.contains d)
    let rhs_result_dimensions : Array (Fin s) :=
      (Array.ofFn (fun i : Fin s => i)).filter (fun d => ¬args.rhs_batching_dimensions.contains d ∧ ¬args.rhs_contracting_dimensions.contains d)
    let outShape :=
      args.lhs_batching_dimensions.map (lhsDims[·])
      ++
      lhs_result_dimensions.map (lhsDims[·])
      ++
      rhs_result_dimensions.map (rhsDims[·])
    outDims.1 = outShape

structure Postcondition {r s t} {lhsDims : Dims r} {rhsDims : Dims s} (args : Args lhsDims rhsDims) (outDims : Dims t) where
  c12' : t = args.outRank
  c12  : outDims = args.outShape (by simp_all)


def Args.batchDims {r s} {lhsDims : Dims r} {rhsDims : Dims s}
    (args : Args lhsDims rhsDims)
    {n} (h : n = args.lhs_batching_dimensions.size := by infer_var) : Dims n :=
  ⟨args.lhs_batching_dimensions.map (lhsDims[·]), by simp_all⟩

def Args.contraDims {r s} {lhsDims : Dims r} {rhsDims : Dims s}
    (args : Args lhsDims rhsDims)
    {n} (h : n = args.lhs_contracting_dimensions.size := by infer_var) : Dims n :=
  ⟨args.lhs_contracting_dimensions.map (lhsDims[·]), by simp_all⟩


def Args.lhsResultDims {r s} {lhsDims : Dims r} {rhsDims : Dims s}
    (args : Args lhsDims rhsDims)
    {n} (h : n = r - args.lhs_batching_dimensions.size - args.lhs_contracting_dimensions.size := by infer_var) : Dims n :=
  ⟨(Array.ofFn (fun i : Fin r => i))
    |>.filter (fun d => (d ∉ args.lhs_batching_dimensions) ∧ (d ∉ args.lhs_contracting_dimensions))
    |>.map (fun i => i.1),
   by sorry /- this proof depends on `args.c3` ! -/⟩


def Args.rhsResultDims {r s} {lhsDims : Dims r} {rhsDims : Dims s}
    (args : Args lhsDims rhsDims)
    {n} (h : n = r - args.rhs_batching_dimensions.size - args.rhs_contracting_dimensions.size := by infer_var) : Dims n :=
  ⟨(Array.ofFn (fun i : Fin s => i))
    |>.filter (fun d => (d ∉ args.rhs_batching_dimensions) ∧ (d ∉ args.rhs_contracting_dimensions))
    |>.map (fun i => i.1),
   by sorry /- this proof depends on `args.c4` ! -/⟩


def Args.lhsIndexMap {r s} {lhsDims : Dims r} {rhsDims : Dims s}
    (args : Args lhsDims rhsDims) :
    XlaTensorIndex lhsDims
    ≃
    XlaTensorIndex args.batchDims × XlaTensorIndex args.contraDims × XlaTensorIndex args.lhsResultDims where
  toFun := fun i => sorry
  invFun := fun ⟨i,l,j⟩ => sorry
  left_inv := sorry
  right_inv := sorry


def Args.rhsIndexMap {r s} {lhsDims : Dims r} {rhsDims : Dims s}
  (args : Args lhsDims rhsDims) :
  XlaTensorIndex rhsDims
  ≃
  XlaTensorIndex args.batchDims × XlaTensorIndex args.contraDims × XlaTensorIndex args.rhsResultDims := sorry


def Args.outIndexMap {r s t} {lhsDims : Dims r} {rhsDims : Dims s} {outDims : Dims t}
  (args : Args lhsDims rhsDims)
  (h : t = args.outRank := by infer_var)
  (houtDims : outDims = args.outShape := by infer_var) :
  XlaTensorIndex outDims
  ≃
  XlaTensorIndex args.batchDims × XlaTensorIndex args.lhsResultDims × XlaTensorIndex args.rhsResultDims := sorry


end DotGeneral


open DotGeneral in
def dot_general {r s t} {lhsDims : Dims r} {rhsDims : Dims s} {outDims : Dims t}
    (lhs : XlaTensorIndex lhsDims → R)
    (rhs : XlaTensorIndex rhsDims → R)
    (args : Args lhsDims rhsDims)
    (ht : t = args.outRank := by infer_var)
    (houtDims : outDims = args.outShape := by infer_var) :
    XlaTensorIndex outDims → R :=
  fun i =>
    let (i,j,k) := args.outIndexMap (by simp_all) (by simp_all) i
    ∑ l : XlaTensorIndex args.contraDims,
      lhs (args.lhsIndexMap.symm (i,l,j)) * rhs (args.rhsIndexMap.symm (i,l,k))
