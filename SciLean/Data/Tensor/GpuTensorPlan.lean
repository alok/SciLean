import SciLean.Data.Tensor.GpuTensor
import SciLean.Monad.TensorMPlan
import SciLean.VersoPrelude

namespace SciLean

namespace GpuTensor

variable {m k p : ℕ}

/-- Kernel variant selected for GEMM planning. -/
inductive GemmKernel where
  | nn
  | tn
  | nt
  | tt
  deriving Repr, Inhabited

/-- Result of planning a GEMM call under layout constraints. -/
structure GemmPlan where
  aTransposed : Bool
  bTransposed : Bool
  aCopied : Bool
  bCopied : Bool
  kernel : GemmKernel
  deriving Repr, Inhabited

/-- Plan layout decisions for GEMM without running {name}`IO`. -/
def gemmPlan
    (A : GpuTensor Float (Idx m × Idx k))
    (B : GpuTensor Float (Idx k × Idx p)) :
    TensorMPlan GemmPlan := do
  let elemBytes := (inferInstance : PlainDataType Float).btype.bytes.toNat
  let normalize {r c : ℕ} (tag : String) (t : GpuTensor Float (Idx r × Idx c)) :
      TensorMPlan (Bool × Bool) := do
    let caps ← TensorMPlan.getCaps
    let policy ← TensorMPlan.getPolicy
    let layout := t.data.layout
    let bytes := layout.numel * elemBytes
    let view? : Option (Bool × Nat × Nat) :=
      if layout.isRowContiguous then
        some (false, layout.rowStride, layout.offset)
      else
        let tl := layout.transpose
        if tl.isRowContiguous then
          some (true, tl.rowStride, tl.offset)
        else
          none
    let view? :=
      match view? with
      | some (transposed, rowStride, offset) =>
          let expectedRowStride := if transposed then r else c
          let needsStride := rowStride != expectedRowStride
          let needsOffset := offset != 0
          let needsTrans := transposed
          if (!needsTrans || caps.acceptsTransposed) &&
             (!needsStride || caps.acceptsStride) &&
             (!needsOffset || caps.acceptsOffset) then
            some (transposed, rowStride, offset)
          else
            none
      | none => none
    if policy.preferViews then
      match view? with
      | some (transposed, _, _) =>
          TensorMPlan.recordViewHit s!"gemm.{tag}"
          return (transposed, false)
      | none => pure ()
    if policy.allowCopy then
      TensorMPlan.recordCopy s!"gemm.{tag}" bytes
      return (false, true)
    throw (LayoutError.copyDisallowed "gemm")
  let (aTransposed, aCopied) ← normalize "A" A
  let (bTransposed, bCopied) ← normalize "B" B
  let kernel :=
    match aTransposed, bTransposed with
    | false, false => GemmKernel.nn
    | true, false => GemmKernel.tn
    | false, true => GemmKernel.nt
    | true, true => GemmKernel.tt
  return { aTransposed := aTransposed
          , bTransposed := bTransposed
          , aCopied := aCopied
          , bCopied := bCopied
          , kernel := kernel }

/-- Plan GEMM and return layout statistics alongside the plan. -/
def gemmPlanWithStats
    (A : GpuTensor Float (Idx m × Idx k))
    (B : GpuTensor Float (Idx k × Idx p)) :
    TensorMPlan (GemmPlan × LayoutStats) :=
  TensorMPlan.withStats (gemmPlan (A := A) (B := B))

end GpuTensor

end SciLean
