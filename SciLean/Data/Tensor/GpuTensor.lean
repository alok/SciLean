/-
Copyright (c) 2024 SciLean contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: SciLean contributors
-/
import SciLean.FFI.Metal.GpuBufferView
import SciLean.Data.DataArray
import SciLean.Data.IndexType.Shape
import SciLean.Monad.TensorM
import SciLean.VersoPrelude

-- Type-safe layout-aware GPU tensor with shape encoded in the type

set_option doc.verso false

namespace SciLean

/-- Type-safe layout-aware GPU tensor.
    Shape is tracked in the type via {name}`IndexType`, and layout metadata enables constant-time views. -/
structure GpuTensor (α : Type) [PlainDataType α] (ι : Type) {n : outParam ℕ} [IndexType ι n] : Type where
  /-- Underlying {name}`GpuBufferView`. -/
  data : GpuBufferView α

instance [PlainDataType α] [IndexType ι n] : Nonempty (GpuTensor α ι) :=
  ⟨⟨Classical.arbitrary _⟩⟩

/-! ## Type Notation for GPU Tensors -/

/-- Postfix `@metal` notation for GPU tensor types.
    `Float^[m, n]@metal` becomes `GpuTensor Float (Idx m × Idx n)` -/
syntax (name:=gpuTensorPostfix) (priority:=high+1) term "@metal" : term

open Lean Meta Elab Term in
elab_rules (kind:=gpuTensorPostfix) : term
| `($x @metal) => do
  -- Elaborate the base type (e.g., Float^[m,n] → DataArrayN Float (Idx m × Idx n))
  let baseType ← elabTerm x none
  -- Extract element type and index type from DataArrayN
  let baseType ← whnfD baseType
  match baseType.getAppFnArgs with
  | (``SciLean.DataArrayN, #[α, _, ι, _, _]) =>
    mkAppOptM ``SciLean.GpuTensor #[α, none, ι, none, none]
  | _ =>
    throwError "expected DataArrayN type for @metal, got {baseType}"

namespace GpuTensor

variable {α : Type} [PlainDataType α]
variable {ι : Type} {n : ℕ} [IndexType ι n]

/-- Number of elements. -/
def size (_ : GpuTensor α ι) : ℕ := n

/-- Size as {name}`USize`. -/
def usize (_ : GpuTensor α ι) : USize := n.toUSize

-- Layout queries (delegate to underlying buffer)

/-- Number of dimensions. -/
def rank (t : GpuTensor α ι) : Nat := t.data.rank

/-- Shape as {name}`Array`. -/
def shape (t : GpuTensor α ι) : Array Nat := t.data.shape

/-- Strides as {name}`Array`. -/
def strides (t : GpuTensor α ι) : Array Nat := t.data.strides

/-- Check if contiguous. -/
def isContiguous (t : GpuTensor α ι) : Bool := t.data.isContiguous

/-- Check if transposed. -/
def isTransposed (t : GpuTensor α ι) : Bool := t.data.isTransposed

-- O(1) view operations
-- Note: Type changes to reflect the new shape

variable {m k p : ℕ}

/-- Transpose last two dimensions (no data copy).
    Changes type from {lean}`Idx m × Idx k` to {lean}`Idx k × Idx m`. -/
def transpose (t : GpuTensor α (Idx m × Idx k)) :
    GpuTensor α (Idx k × Idx m) :=
  ⟨t.data.transpose⟩

/-- Transpose for 3D batched tensors by swapping last two dimensions.
    Changes type from {lean}`Idx p × Idx m × Idx k` to {lean}`Idx p × Idx k × Idx m`. -/
def batchTranspose (t : GpuTensor α (Idx p × Idx m × Idx k)) :
    GpuTensor α (Idx p × Idx k × Idx m) :=
  ⟨t.data.transpose⟩

/-- Slice rows of a matrix along the first dimension (O(1) view). -/
def sliceRows (t : GpuTensor α (Idx m × Idx k)) (start len : Nat) :
    GpuTensor α (Idx len × Idx k) :=
  ⟨t.data.slice 0 start len⟩

/-- Slice columns of a matrix along the second dimension (O(1) view). -/
def sliceCols (t : GpuTensor α (Idx m × Idx k)) (start len : Nat) :
    GpuTensor α (Idx m × Idx len) :=
  ⟨t.data.slice 1 start len⟩

/-! ## Generic view ops (caller picks target index type) -/

/-- Permute dimensions (O(1) view). The caller must ensure the target index type matches
    the permuted shape. -/
def permuteView {κ : Type} {m : ℕ} [IndexType κ m]
    (t : GpuTensor α ι) (perm : Array Nat) : GpuTensor α κ :=
  ⟨t.data.permute perm⟩

/-- Squeeze a size-1 dimension (O(1) view). The caller must ensure the target index type
    matches the squeezed shape. -/
def squeezeView {κ : Type} {m : ℕ} [IndexType κ m]
    (t : GpuTensor α ι) (dim : Nat) : GpuTensor α κ :=
  ⟨t.data.squeeze dim⟩

/-- Unsqueeze (add a size-1 dimension) (O(1) view). The caller must ensure the target
    index type matches the expanded shape. -/
def unsqueezeView {κ : Type} {m : ℕ} [IndexType κ m]
    (t : GpuTensor α ι) (dim : Nat) : GpuTensor α κ :=
  ⟨t.data.unsqueeze dim⟩

/-- Make a contiguous copy if needed. -/
def contiguous (t : GpuTensor α ι) : IO (GpuTensor α ι) := do
  let newData ← t.data.contiguous
  return ⟨newData⟩

/-! ## Contiguity helpers -/

/-- Ensure a contiguous layout (copy if needed). -/
def ensureContiguous (t : GpuTensor α ι) : IO (GpuTensor α ι) :=
  if t.isContiguous && t.data.layout.offset == 0 then
    pure t
  else
    t.contiguous

-- Construction

/-- Create from {name}`GpuBufferView`. -/
def ofBuffer (buf : GpuBufferView α) : GpuTensor α ι := ⟨buf⟩

/-- Create from contiguous {name}`Metal.GpuBuffer` with an explicit shape. -/
def fromContiguousBuffer (buffer : Metal.GpuBuffer) (shape : Array Nat) : GpuTensor α ι :=
  ⟨GpuBufferView.fromContiguous buffer shape⟩

/-- Create from contiguous {name}`Metal.GpuBuffer` using {name}`IndexTypeShape` for layout. -/
def fromContiguous (buffer : Metal.GpuBuffer) [IndexTypeShape ι n] : GpuTensor α ι :=
  ⟨GpuBufferView.fromContiguous buffer (IndexTypeShape.shape (ι:=ι))⟩

/-- Get underlying {name}`Metal.GpuBuffer`. -/
def toGpuBuffer (t : GpuTensor α ι) : Metal.GpuBuffer := t.data.buffer

/-! ## Element-wise Operations -/

/-- Element-wise addition on GPU (any shape). -/
def add [IndexTypeShape ι n] (a b : GpuTensor Float ι) : IO (GpuTensor Float ι) := do
  let a ← ensureContiguous a
  let b ← ensureContiguous b
  let total := IndexTypeShape.numel (ι:=ι)
  let result ← Metal.GpuBuffer.add a.data.buffer b.data.buffer total.toUSize
  return ⟨GpuBufferView.fromContiguous result (IndexTypeShape.shape (ι:=ι))⟩

/-- Element-wise subtraction on GPU (any shape). -/
def sub [IndexTypeShape ι n] (a b : GpuTensor Float ι) : IO (GpuTensor Float ι) := do
  let a ← ensureContiguous a
  let b ← ensureContiguous b
  let total := IndexTypeShape.numel (ι:=ι)
  let result ← Metal.GpuBuffer.sub a.data.buffer b.data.buffer total.toUSize
  return ⟨GpuBufferView.fromContiguous result (IndexTypeShape.shape (ι:=ι))⟩

/-- Element-wise multiplication on GPU (any shape). -/
def mul [IndexTypeShape ι n] (a b : GpuTensor Float ι) : IO (GpuTensor Float ι) := do
  let a ← ensureContiguous a
  let b ← ensureContiguous b
  let total := IndexTypeShape.numel (ι:=ι)
  let result ← Metal.GpuBuffer.mul a.data.buffer b.data.buffer total.toUSize
  return ⟨GpuBufferView.fromContiguous result (IndexTypeShape.shape (ι:=ι))⟩

/-- Element-wise negation on GPU (any shape). -/
def neg [IndexTypeShape ι n] (a : GpuTensor Float ι) : IO (GpuTensor Float ι) := do
  let a ← ensureContiguous a
  let total := IndexTypeShape.numel (ι:=ι)
  let result ← Metal.GpuBuffer.scale total.toUSize (-1.0) a.data.buffer
  return ⟨GpuBufferView.fromContiguous result (IndexTypeShape.shape (ι:=ι))⟩

/-- ReLU activation on GPU (any shape). -/
def relu [IndexTypeShape ι n] (a : GpuTensor Float ι) : IO (GpuTensor Float ι) := do
  let a ← ensureContiguous a
  let total := IndexTypeShape.numel (ι:=ι)
  let result ← Metal.GpuBuffer.relu a.data.buffer total.toUSize
  return ⟨GpuBufferView.fromContiguous result (IndexTypeShape.shape (ι:=ι))⟩

/-- Scalar multiplication on GPU. -/
def scale [IndexTypeShape ι n] (s : Float) (a : GpuTensor Float ι) : IO (GpuTensor Float ι) := do
  let a ← ensureContiguous a
  let total := IndexTypeShape.numel (ι:=ι)
  let result ← Metal.GpuBuffer.scale total.toUSize s a.data.buffer
  return ⟨GpuBufferView.fromContiguous result (IndexTypeShape.shape (ι:=ι))⟩

/-- AXPY operation: result = alpha * x + y -/
def axpy [IndexTypeShape ι n] (alpha : Float) (x y : GpuTensor Float ι) : IO (GpuTensor Float ι) := do
  let x ← ensureContiguous x
  let y ← ensureContiguous y
  let total := IndexTypeShape.numel (ι:=ι)
  let result ← Metal.GpuBuffer.axpy total.toUSize alpha x.data.buffer y.data.buffer
  return ⟨GpuBufferView.fromContiguous result (IndexTypeShape.shape (ι:=ι))⟩

end GpuTensor

/-! ## Matrix Operations -/

namespace GpuTensor

variable {α : Type} [PlainDataType α]
variable {ι : Type} {n : ℕ} [IndexType ι n]
variable {m k p : ℕ}

/-- Softmax activation on GPU (applied row-wise).
    For input shape Idx m × Idx n, softmax is applied to each row independently. -/
def softmax (a : GpuTensor Float (Idx m × Idx k)) : IO (GpuTensor Float (Idx m × Idx k)) := do
  let a ← ensureContiguous a
  let result ← Metal.GpuBuffer.softmax a.data.buffer m.toUSize k.toUSize
  return ⟨GpuBufferView.fromContiguous result #[m, k]⟩

/-- GELU activation on GPU using fused bias+gelu with zero bias (any shape). -/
def gelu [IndexTypeShape ι n] (a : GpuTensor Float ι) : IO (GpuTensor Float ι) := do
  let a ← ensureContiguous a
  let total := IndexTypeShape.numel (ι:=ι)
  -- Create zero bias buffer from empty ByteArray sized correctly
  let zeroBytes := ByteArray.mk (Array.replicate (total * 4) 0)  -- 4 bytes per Float32
  let zeroBias ← Metal.GpuBuffer.fromByteArray zeroBytes
  let result ← Metal.GpuBuffer.biasGelu a.data.buffer zeroBias total.toUSize 1
  return ⟨GpuBufferView.fromContiguous result (IndexTypeShape.shape (ι:=ι))⟩

/-- GELU backward pass on GPU (any shape). -/
def geluBackward [IndexTypeShape ι n] (x dout : GpuTensor Float ι) : IO (GpuTensor Float ι) := do
  let x ← ensureContiguous x
  let dout ← ensureContiguous dout
  let total := IndexTypeShape.numel (ι:=ι)
  let result ← Metal.GpuBuffer.geluBackward x.data.buffer dout.data.buffer total.toUSize
  return ⟨GpuBufferView.fromContiguous result (IndexTypeShape.shape (ι:=ι))⟩

/-- Fused GEMM + bias + ReLU.
    Shapes: A Idx m × Idx k, B Idx k × Idx p, bias Idx p,
    returns Idx m × Idx p. -/
def gemmBiasRelu (A : GpuTensor Float (Idx m × Idx k)) (B : GpuTensor Float (Idx k × Idx p))
    (bias : GpuTensor Float (Idx p)) : IO (GpuTensor Float (Idx m × Idx p)) := do
  let A ← ensureContiguous A
  let B ← ensureContiguous B
  let bias ← ensureContiguous bias
  let result ← Metal.GpuBuffer.gemmBiasRelu A.data.buffer B.data.buffer bias.data.buffer
    m.toUSize k.toUSize p.toUSize
  return ⟨GpuBufferView.fromContiguous result #[m, p]⟩

/-- Bias add: y = x + bias (broadcast bias across rows).
    Input shape Idx m × Idx k, bias shape Idx k, output shape Idx m × Idx k. -/
def biasAdd (x : GpuTensor Float (Idx m × Idx k)) (bias : GpuTensor Float (Idx k)) :
    IO (GpuTensor Float (Idx m × Idx k)) := do
  let x ← ensureContiguous x
  let bias ← ensureContiguous bias
  let result ← Metal.GpuBuffer.biasAdd x.data.buffer bias.data.buffer (m * k).toUSize k.toUSize
  return ⟨GpuBufferView.fromContiguous result #[m, k]⟩

/-- Sum columns: reduce across rows.
    Input shape Idx m × Idx k, output shape Idx k. -/
def colSum (x : GpuTensor Float (Idx m × Idx k)) : IO (GpuTensor Float (Idx k)) := do
  let x ← ensureContiguous x
  let result ← Metal.GpuBuffer.colSum x.data.buffer m.toUSize k.toUSize
  return ⟨GpuBufferView.fromContiguous result #[k]⟩

/-- GEMM with A transposed.
    A stored as Idx k × Idx m, B as Idx k × Idx p, result is Idx m × Idx p. -/
def gemmTransposeLeft (A : GpuTensor Float (Idx k × Idx m)) (B : GpuTensor Float (Idx k × Idx p)) :
    IO (GpuTensor Float (Idx m × Idx p)) := do
  let A ← ensureContiguous A
  let B ← ensureContiguous B
  let result ← Metal.GpuBuffer.gemmTransposeLeft_AMX A.data.buffer B.data.buffer
    m.toUSize k.toUSize p.toUSize
  return ⟨GpuBufferView.fromContiguous result #[m, p]⟩

/-- GEMM with B transposed.
    A stored as Idx m × Idx k, B as Idx p × Idx k, result is Idx m × Idx p. -/
def gemmTransposeRight (A : GpuTensor Float (Idx m × Idx k)) (B : GpuTensor Float (Idx p × Idx k)) :
    IO (GpuTensor Float (Idx m × Idx p)) := do
  let A ← ensureContiguous A
  let B ← ensureContiguous B
  let result ← Metal.GpuBuffer.gemmTransposeRight_AMX A.data.buffer B.data.buffer
    m.toUSize k.toUSize p.toUSize
  return ⟨GpuBufferView.fromContiguous result #[m, p]⟩

variable {batch : ℕ}

/-- Layout-aware matrix multiply in {name}`TensorMT`.
    Uses layout views when allowed by policy/caps and records copy statistics. -/
def gemmMT {monad : Type → Type} [Monad monad] [MonadLift IO monad]
    (A : GpuTensor Float (Idx m × Idx k))
    (B : GpuTensor Float (Idx k × Idx p)) :
    TensorMT monad (GpuTensor Float (Idx m × Idx p)) := do
  let elemBytes := (inferInstance : PlainDataType Float).btype.bytes.toNat
  let normalize {r c : ℕ} (t : GpuTensor Float (Idx r × Idx c)) :
      TensorMT monad (GpuTensor Float (Idx r × Idx c) × Bool × Nat × Nat) := do
    let caps ← TensorMT.getCaps
    let policy ← TensorMT.getPolicy
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
      | some (transposed, rowStride, offset) =>
          TensorMT.recordViewHit
          return (t, transposed, rowStride, offset)
      | none => pure ()
    if policy.allowCopy then
      let t' ← TensorMT.liftIO t.contiguous
      TensorMT.recordCopy bytes
      let layout' := t'.data.layout
      return (t', false, layout'.rowStride, layout'.offset)
    throw (LayoutError.copyDisallowed "gemm")
  let (A, aTransposed, aRowStride, aOffset) ← normalize A
  let (B, bTransposed, bRowStride, bOffset) ← normalize B
  let result ← TensorMT.liftIO <|
    Metal.GpuBuffer.gemmLayout
      A.data.buffer B.data.buffer
      m.toUSize k.toUSize p.toUSize
      aRowStride.toUSize bRowStride.toUSize
      aOffset.toUSize bOffset.toUSize
      aTransposed bTransposed
  return ⟨GpuBufferView.fromContiguous result #[m, p]⟩

/-- {name}`TensorM` specialization of {name}`gemmMT`. -/
def gemmM (A : GpuTensor Float (Idx m × Idx k))
    (B : GpuTensor Float (Idx k × Idx p)) : TensorM (GpuTensor Float (Idx m × Idx p)) :=
  by
    let _ : MonadLift IO IO := ⟨fun x => x⟩
    simpa using (gemmMT (monad:=IO) A B)

/-- {name}`TensorM` wrapper that returns the result with the final stats. -/
def gemmMWithStats (A : GpuTensor Float (Idx m × Idx k))
    (B : GpuTensor Float (Idx k × Idx p)) : TensorM (GpuTensor Float (Idx m × Idx p) × LayoutStats) :=
  TensorMT.withStats (gemmM A B)

/-- Matrix multiply with layout-aware inputs.
    Uses {name}`Metal.GpuBuffer.gemmLayout` with row strides and offsets derived from views.
    This lets transpose views work without materializing a copy. -/
def gemm (A : GpuTensor Float (Idx m × Idx k))
         (B : GpuTensor Float (Idx k × Idx p)) :
    IO (GpuTensor Float (Idx m × Idx p)) := do
  let normalize {r c : ℕ} (t : GpuTensor Float (Idx r × Idx c)) :
      IO (GpuTensor Float (Idx r × Idx c) × Bool × Nat × Nat) := do
    let layout := t.data.layout
    if layout.isRowContiguous then
      return (t, false, layout.rowStride, layout.offset)
    else
      let tl := layout.transpose
      if tl.isRowContiguous then
        return (t, true, tl.rowStride, tl.offset)
    let t' ← t.contiguous
    let layout' := t'.data.layout
    return (t', false, layout'.rowStride, layout'.offset)
  let (A, aTransposed, aRowStride, aOffset) ← normalize A
  let (B, bTransposed, bRowStride, bOffset) ← normalize B
  let result ←
    Metal.GpuBuffer.gemmLayout
      A.data.buffer B.data.buffer
      m.toUSize k.toUSize p.toUSize
      aRowStride.toUSize bRowStride.toUSize
      aOffset.toUSize bOffset.toUSize
      aTransposed bTransposed
  return ⟨GpuBufferView.fromContiguous result #[m, p]⟩

/-- Matrix multiply with auto backend selection (AMX vs MPS).
    Uses {name}`Metal.GpuBuffer.gemmAuto` for contiguous inputs and
    falls back to {name}`Metal.GpuBuffer.gemmLayout` for stride/offset views. -/
def gemmAuto (A : GpuTensor Float (Idx m × Idx k))
           (B : GpuTensor Float (Idx k × Idx p)) :
  IO (GpuTensor Float (Idx m × Idx p)) := do
  let normalize {r c : ℕ} (t : GpuTensor Float (Idx r × Idx c)) :
      IO (GpuTensor Float (Idx r × Idx c) × Bool × Nat × Nat) := do
    let layout := t.data.layout
    if layout.isRowContiguous then
      return (t, false, layout.rowStride, layout.offset)
    else
      let tl := layout.transpose
      if tl.isRowContiguous then
        return (t, true, tl.rowStride, tl.offset)
    let t' ← t.contiguous
    let layout' := t'.data.layout
    return (t', false, layout'.rowStride, layout'.offset)
  let (A, aTransposed, aRowStride, aOffset) ← normalize A
  let (B, bTransposed, bRowStride, bOffset) ← normalize B
  let aContig := (!aTransposed) && aOffset == 0 && aRowStride == k
  let bContig := (!bTransposed) && bOffset == 0 && bRowStride == p
  let result ←
    if aContig && bContig then
      Metal.GpuBuffer.gemmAuto A.data.buffer B.data.buffer m.toUSize k.toUSize p.toUSize
    else
      Metal.GpuBuffer.gemmLayout
        A.data.buffer B.data.buffer
        m.toUSize k.toUSize p.toUSize
        aRowStride.toUSize bRowStride.toUSize
        aOffset.toUSize bOffset.toUSize
        aTransposed bTransposed
  return ⟨GpuBufferView.fromContiguous result #[m, p]⟩

/-- Matrix multiply using AMX (contiguous inputs only). -/
def gemmAMX (A : GpuTensor Float (Idx m × Idx k))
            (B : GpuTensor Float (Idx k × Idx p)) :
    IO (GpuTensor Float (Idx m × Idx p)) := do
  let A ← ensureContiguous A
  let B ← ensureContiguous B
  let result ←
    Metal.GpuBuffer.gemmAMX A.data.buffer B.data.buffer m.toUSize k.toUSize p.toUSize
  return ⟨GpuBufferView.fromContiguous result #[m, p]⟩

/-- Matrix multiply using AMX with A stored transposed.
    A is stored as {lean}`Idx k × Idx m`, B as {lean}`Idx k × Idx p`,
    returns {lean}`Idx m × Idx p`. -/
def gemmTransposeLeft_AMX (A : GpuTensor Float (Idx k × Idx m))
               (B : GpuTensor Float (Idx k × Idx p)) :
    IO (GpuTensor Float (Idx m × Idx p)) := do
  let A ← ensureContiguous A
  let B ← ensureContiguous B
  let result ←
    Metal.GpuBuffer.gemmTransposeLeft_AMX A.data.buffer B.data.buffer m.toUSize k.toUSize p.toUSize
  return ⟨GpuBufferView.fromContiguous result #[m, p]⟩

/-- Matrix multiply using AMX with B stored transposed.
    A is stored as {lean}`Idx m × Idx k`, B as {lean}`Idx p × Idx k`,
    returns {lean}`Idx m × Idx p`. -/
def gemmTransposeRight_AMX (A : GpuTensor Float (Idx m × Idx k))
               (B : GpuTensor Float (Idx p × Idx k)) :
    IO (GpuTensor Float (Idx m × Idx p)) := do
  let A ← ensureContiguous A
  let B ← ensureContiguous B
  let result ←
    Metal.GpuBuffer.gemmTransposeRight_AMX A.data.buffer B.data.buffer m.toUSize k.toUSize p.toUSize
  return ⟨GpuBufferView.fromContiguous result #[m, p]⟩

-- Batched GEMM for multi-head attention

variable {batch : ℕ}

/-- Batched matrix multiply with layout-aware inputs.
    Each batch computes an independent matrix multiplication.
    Shapes: A {lean}`Idx batch × Idx m × Idx k`, B {lean}`Idx batch × Idx k × Idx p`,
    returns {lean}`Idx batch × Idx m × Idx p`.
    Automatically dispatches based on transpose state of the last two dimensions. -/
def batchedGemm (A : GpuTensor Float (Idx batch × Idx m × Idx k))
                (B : GpuTensor Float (Idx batch × Idx k × Idx p)) :
    IO (GpuTensor Float (Idx batch × Idx m × Idx p)) := do
  let normalize {r c : ℕ} (t : GpuTensor Float (Idx batch × Idx r × Idx c)) :
      IO (GpuTensor Float (Idx batch × Idx r × Idx c) × Bool × Nat × Nat × Nat) := do
    let layout := t.data.layout
    if layout.isRowContiguous then
      return (t, false, layout.strides.getD 0 0, layout.rowStride, layout.offset)
    else
      let tl := layout.transpose
      if tl.isRowContiguous then
        return (t, true, tl.strides.getD 0 0, tl.rowStride, tl.offset)
    let t' ← t.contiguous
    let layout' := t'.data.layout
    return (t', false, layout'.strides.getD 0 0, layout'.rowStride, layout'.offset)
  let (A, aTransposed, aBatchStride, aRowStride, aOffset) ← normalize A
  let (B, bTransposed, bBatchStride, bRowStride, bOffset) ← normalize B
  let aContig :=
    (!aTransposed) && aOffset == 0 && aRowStride == k && aBatchStride == m * k
  let bContig :=
    (!bTransposed) && bOffset == 0 && bRowStride == p && bBatchStride == k * p
  let result ←
    if aContig && bContig then
      Metal.GpuBuffer.gemmBatched A.data.buffer B.data.buffer
        batch.toUSize m.toUSize k.toUSize p.toUSize
    else
      Metal.GpuBuffer.gemmBatchedLayout
        A.data.buffer B.data.buffer
        batch.toUSize m.toUSize k.toUSize p.toUSize
        aBatchStride.toUSize bBatchStride.toUSize
        aRowStride.toUSize bRowStride.toUSize
        aOffset.toUSize bOffset.toUSize
        aTransposed bTransposed
  return ⟨GpuBufferView.fromContiguous result #[batch, m, p]⟩

/-- Batched GEMM backward using transpose views.
    Uses {name}`batchTranspose` and {name}`batchedGemm` to compute gradients without
    materializing transposed buffers. -/
def batchedGemmBackward
    (A : GpuTensor Float (Idx batch × Idx m × Idx k))
    (B : GpuTensor Float (Idx batch × Idx k × Idx p))
    (dC : GpuTensor Float (Idx batch × Idx m × Idx p)) :
    IO (GpuTensor Float (Idx batch × Idx m × Idx k) ×
        GpuTensor Float (Idx batch × Idx k × Idx p)) := do
  -- O(1) transpose views
  let B_T := B.batchTranspose  -- (batch, k, p) -> (batch, p, k)
  let A_T := A.batchTranspose  -- (batch, m, k) -> (batch, k, m)
  -- Batched GEMM handles transposed layouts via dispatch
  let dA ← batchedGemm dC B_T   -- (batch, m, p) @ (batch, p, k) = (batch, m, k)
  let dB ← batchedGemm A_T dC   -- (batch, k, m) @ (batch, m, p) = (batch, k, p)
  return (dA, dB)

end GpuTensor

-- Attention operations for layout-aware tensors

namespace GpuTensor

variable {seqLen headDim : ℕ}

/-- Flash Attention with layout-aware inputs.
    Uses {name}`Metal.GpuBuffer.flashAttention` on contiguous buffers.
    Inputs have shape {lean}`Idx seqLen × Idx headDim`, returns {lean}`Idx seqLen × Idx headDim`.
    Transposed inputs are made contiguous as needed. -/
def flashAttention
    (Q : GpuTensor Float (Idx seqLen × Idx headDim))
    (K : GpuTensor Float (Idx seqLen × Idx headDim))
    (V : GpuTensor Float (Idx seqLen × Idx headDim)) :
    IO (GpuTensor Float (Idx seqLen × Idx headDim)) := do
  -- The kernel expects contiguous row-major inputs
  -- Make contiguous copies if any input is transposed
  let qBuf ← if Q.isContiguous then pure Q.data.buffer else do
    let c ← Q.contiguous; pure c.data.buffer
  let kBuf ← if K.isContiguous then pure K.data.buffer else do
    let c ← K.contiguous; pure c.data.buffer
  let vBuf ← if V.isContiguous then pure V.data.buffer else do
    let c ← V.contiguous; pure c.data.buffer

  let result ← Metal.GpuBuffer.flashAttention qBuf kBuf vBuf seqLen.toUSize headDim.toUSize
  return ⟨GpuBufferView.fromContiguous result #[seqLen, headDim]⟩

/-- Flash Attention with causal mask and layout-aware inputs.
    Each position can only attend to earlier positions.
    Inputs have shape {lean}`Idx seqLen × Idx headDim`, returns {lean}`Idx seqLen × Idx headDim`. -/
def flashAttentionCausal
    (Q : GpuTensor Float (Idx seqLen × Idx headDim))
    (K : GpuTensor Float (Idx seqLen × Idx headDim))
    (V : GpuTensor Float (Idx seqLen × Idx headDim)) :
    IO (GpuTensor Float (Idx seqLen × Idx headDim)) := do
  -- Make contiguous copies if needed
  let qBuf ← if Q.isContiguous then pure Q.data.buffer else do
    let c ← Q.contiguous; pure c.data.buffer
  let kBuf ← if K.isContiguous then pure K.data.buffer else do
    let c ← K.contiguous; pure c.data.buffer
  let vBuf ← if V.isContiguous then pure V.data.buffer else do
    let c ← V.contiguous; pure c.data.buffer

  let result ← Metal.GpuBuffer.flashAttentionCausal qBuf kBuf vBuf seqLen.toUSize headDim.toUSize
  return ⟨GpuBufferView.fromContiguous result #[seqLen, headDim]⟩

/-- Scaled dot-product attention using explicit GEMM operations.
    This version uses {name}`gemm` instead of a fused kernel, which allows
    transpose views without materializing copies.
    Inputs have shape {lean}`Idx seqLen × Idx headDim`, returns {lean}`Idx seqLen × Idx headDim`. -/
def scaledDotProductAttention
    (Q : GpuTensor Float (Idx seqLen × Idx headDim))
    (K : GpuTensor Float (Idx seqLen × Idx headDim))
    (V : GpuTensor Float (Idx seqLen × Idx headDim)) :
    IO (GpuTensor Float (Idx seqLen × Idx headDim)) := do
  -- Step 1: Compute attention scores = Q @ K^T
  -- K^T is an O(1) view operation!
  let K_T := K.transpose  -- (seqLen, headDim) -> (headDim, seqLen)
  let scores ← gemm Q K_T  -- (seqLen, headDim) @ (headDim, seqLen) = (seqLen, seqLen)

  -- Step 2: Scale by 1/sqrt(headDim)
  let scaleVal := 1.0 / Float.sqrt headDim.toFloat
  let scaledScores ← Metal.GpuBuffer.scale (seqLen * seqLen).toUSize scaleVal scores.data.buffer
  let scaledTensor : GpuTensor Float (Idx seqLen × Idx seqLen) :=
    ⟨GpuBufferView.fromContiguous scaledScores #[seqLen, seqLen]⟩

  -- Step 3: Softmax (row-wise)
  let attnWeights ← Metal.GpuBuffer.softmax scaledTensor.data.buffer seqLen.toUSize seqLen.toUSize
  let attnTensor : GpuTensor Float (Idx seqLen × Idx seqLen) :=
    ⟨GpuBufferView.fromContiguous attnWeights #[seqLen, seqLen]⟩

  -- Step 4: Apply to values: output = attnWeights @ V
  gemm attnTensor V  -- (seqLen, seqLen) @ (seqLen, headDim) = (seqLen, headDim)

end GpuTensor

end SciLean
