/-
Copyright (c) 2024 SciLean contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: SciLean contributors
-/
import SciLean.FFI.Metal.StridedBuffer
import SciLean.Data.DataArray
import SciLean.Data.IndexType.Shape

-- Type-safe strided GPU tensor with shape encoded in the type

namespace SciLean

/-- Type-safe strided GPU tensor.
    Shape is tracked in the type via {name}`IndexType`, and layout metadata enables constant-time views. -/
structure StridedGpuTensor (α : Type) [PlainDataType α] (ι : Type) {n : outParam ℕ} [IndexType ι n] : Type where
  /-- Underlying {name}`StridedGpuBuffer`. -/
  data : StridedGpuBuffer α

instance [PlainDataType α] [IndexType ι n] : Nonempty (StridedGpuTensor α ι) :=
  ⟨⟨Classical.arbitrary _⟩⟩

namespace StridedGpuTensor

variable {α : Type} [PlainDataType α]
variable {ι : Type} {n : ℕ} [IndexType ι n]

-- Layout queries (delegate to underlying buffer)

/-- Number of dimensions. -/
def rank (t : StridedGpuTensor α ι) : Nat := t.data.rank

/-- Shape as {name}`Array`. -/
def shape (t : StridedGpuTensor α ι) : Array Nat := t.data.shape

/-- Strides as {name}`Array`. -/
def strides (t : StridedGpuTensor α ι) : Array Nat := t.data.strides

/-- Check if contiguous. -/
def isContiguous (t : StridedGpuTensor α ι) : Bool := t.data.isContiguous

/-- Check if transposed. -/
def isTransposed (t : StridedGpuTensor α ι) : Bool := t.data.isTransposed

-- O(1) view operations
-- Note: Type changes to reflect the new shape

variable {m k p : ℕ}

/-- Transpose last two dimensions (no data copy).
    Changes type from {lean}`Idx m × Idx k` to {lean}`Idx k × Idx m`. -/
def transpose (t : StridedGpuTensor α (Idx m × Idx k)) :
    StridedGpuTensor α (Idx k × Idx m) :=
  ⟨t.data.transpose⟩

/-- Transpose for 3D batched tensors by swapping last two dimensions.
    Changes type from {lean}`Idx p × Idx m × Idx k` to {lean}`Idx p × Idx k × Idx m`. -/
def batchTranspose (t : StridedGpuTensor α (Idx p × Idx m × Idx k)) :
    StridedGpuTensor α (Idx p × Idx k × Idx m) :=
  ⟨t.data.transpose⟩

/-- Make a contiguous copy if needed. -/
def contiguous (t : StridedGpuTensor α ι) : IO (StridedGpuTensor α ι) := do
  let newData ← t.data.contiguous
  return ⟨newData⟩

/-! ## Contiguity helpers -/

/-- Ensure a contiguous layout (copy if needed). -/
def ensureContiguous (t : StridedGpuTensor α ι) : IO (StridedGpuTensor α ι) :=
  if t.isContiguous then
    pure t
  else
    t.contiguous

-- Construction

/-- Create from {name}`StridedGpuBuffer`. -/
def ofBuffer (buf : StridedGpuBuffer α) : StridedGpuTensor α ι := ⟨buf⟩

/-- Create from contiguous {name}`Metal.GpuBuffer` with an explicit shape. -/
def fromContiguousBuffer (buffer : Metal.GpuBuffer) (shape : Array Nat) : StridedGpuTensor α ι :=
  ⟨StridedGpuBuffer.fromContiguous buffer shape⟩

/-- Create from contiguous {name}`Metal.GpuBuffer` using {name}`IndexTypeShape` for layout. -/
def fromContiguous (buffer : Metal.GpuBuffer) [IndexTypeShape ι n] : StridedGpuTensor α ι :=
  ⟨StridedGpuBuffer.fromContiguous buffer (IndexTypeShape.shape (ι:=ι))⟩

/-- Get underlying {name}`Metal.GpuBuffer`. -/
def toGpuBuffer (t : StridedGpuTensor α ι) : Metal.GpuBuffer := t.data.buffer

end StridedGpuTensor

-- Matrix operations for strided tensors

namespace StridedGpuTensor

variable {m k p : ℕ}

/-- Matrix multiply with strided inputs.
    Automatically dispatches to {name}`Metal.GpuBuffer.gemm`, {name}`Metal.GpuBuffer.gemmTN`,
    or {name}`Metal.GpuBuffer.gemmNT` based on transpose state.
    This lets transpose views work without materializing a copy. -/
def gemm (A : StridedGpuTensor Float (Idx m × Idx k))
         (B : StridedGpuTensor Float (Idx k × Idx p)) :
    IO (StridedGpuTensor Float (Idx m × Idx p)) := do
  let aTransposed := A.isTransposed
  let bTransposed := B.isTransposed
  let result ←
    match aTransposed, bTransposed with
    | false, false =>
      -- Both contiguous: C = A @ B
      Metal.GpuBuffer.gemm A.data.buffer B.data.buffer m.toUSize k.toUSize p.toUSize
    | true, false =>
      -- A transposed: C = A^T @ B
      Metal.GpuBuffer.gemmTN A.data.buffer B.data.buffer m.toUSize k.toUSize p.toUSize
    | false, true =>
      -- B transposed: C = A @ B^T
      Metal.GpuBuffer.gemmNT A.data.buffer B.data.buffer m.toUSize k.toUSize p.toUSize
    | true, true =>
      -- Both transposed: C = A^T @ B^T = (B @ A)^T
      -- Need to do B @ A then transpose, or make contiguous copies
      -- For now, make A contiguous and use gemmNT
      let aContig ← A.contiguous
      Metal.GpuBuffer.gemmNT aContig.data.buffer B.data.buffer m.toUSize k.toUSize p.toUSize
  return ⟨StridedGpuBuffer.fromContiguous result #[m, p]⟩

-- Batched GEMM for multi-head attention

variable {batch : ℕ}

/-- Batched matrix multiply with strided inputs.
    Each batch computes an independent matrix multiplication.
    Shapes: A {lean}`Idx batch × Idx m × Idx k`, B {lean}`Idx batch × Idx k × Idx p`,
    returns {lean}`Idx batch × Idx m × Idx p`.
    Automatically dispatches based on transpose state of the last two dimensions. -/
def batchedGemm (A : StridedGpuTensor Float (Idx batch × Idx m × Idx k))
                (B : StridedGpuTensor Float (Idx batch × Idx k × Idx p)) :
    IO (StridedGpuTensor Float (Idx batch × Idx m × Idx p)) := do
  let aTransposed := A.isTransposed
  let bTransposed := B.isTransposed
  let result ←
    match aTransposed, bTransposed with
    | false, false =>
      Metal.GpuBuffer.gemmBatched A.data.buffer B.data.buffer batch.toUSize m.toUSize k.toUSize p.toUSize
    | true, false =>
      Metal.GpuBuffer.gemmBatchedTN A.data.buffer B.data.buffer batch.toUSize m.toUSize k.toUSize p.toUSize
    | false, true =>
      Metal.GpuBuffer.gemmBatchedNT A.data.buffer B.data.buffer batch.toUSize m.toUSize k.toUSize p.toUSize
    | true, true =>
      -- Both transposed: make A contiguous and use batchedNT
      let aContig ← A.contiguous
      Metal.GpuBuffer.gemmBatchedNT aContig.data.buffer B.data.buffer batch.toUSize m.toUSize k.toUSize p.toUSize
  return ⟨StridedGpuBuffer.fromContiguous result #[batch, m, p]⟩

/-- Batched GEMM backward using transpose views.
    Uses {name}`batchTranspose` and {name}`batchedGemm` to compute gradients without
    materializing transposed buffers. -/
def batchedGemmBackward
    (A : StridedGpuTensor Float (Idx batch × Idx m × Idx k))
    (B : StridedGpuTensor Float (Idx batch × Idx k × Idx p))
    (dC : StridedGpuTensor Float (Idx batch × Idx m × Idx p)) :
    IO (StridedGpuTensor Float (Idx batch × Idx m × Idx k) ×
        StridedGpuTensor Float (Idx batch × Idx k × Idx p)) := do
  -- O(1) transpose views
  let B_T := B.batchTranspose  -- (batch, k, p) -> (batch, p, k)
  let A_T := A.batchTranspose  -- (batch, m, k) -> (batch, k, m)
  -- Batched GEMM handles transposed layouts via dispatch
  let dA ← batchedGemm dC B_T   -- (batch, m, p) @ (batch, p, k) = (batch, m, k)
  let dB ← batchedGemm A_T dC   -- (batch, k, m) @ (batch, m, p) = (batch, k, p)
  return (dA, dB)

end StridedGpuTensor

-- Attention operations for strided tensors

namespace StridedGpuTensor

variable {seqLen headDim : ℕ}

/-- Flash Attention with strided inputs.
    Uses {name}`Metal.GpuBuffer.flashAttention` on contiguous buffers.
    Inputs have shape {lean}`Idx seqLen × Idx headDim`, returns {lean}`Idx seqLen × Idx headDim`.
    Transposed inputs are made contiguous as needed. -/
def flashAttention
    (Q : StridedGpuTensor Float (Idx seqLen × Idx headDim))
    (K : StridedGpuTensor Float (Idx seqLen × Idx headDim))
    (V : StridedGpuTensor Float (Idx seqLen × Idx headDim)) :
    IO (StridedGpuTensor Float (Idx seqLen × Idx headDim)) := do
  -- The kernel expects contiguous row-major inputs
  -- Make contiguous copies if any input is transposed
  let qBuf ← if Q.isContiguous then pure Q.data.buffer else do
    let c ← Q.contiguous; pure c.data.buffer
  let kBuf ← if K.isContiguous then pure K.data.buffer else do
    let c ← K.contiguous; pure c.data.buffer
  let vBuf ← if V.isContiguous then pure V.data.buffer else do
    let c ← V.contiguous; pure c.data.buffer

  let result ← Metal.GpuBuffer.flashAttention qBuf kBuf vBuf seqLen.toUSize headDim.toUSize
  return ⟨StridedGpuBuffer.fromContiguous result #[seqLen, headDim]⟩

/-- Flash Attention with causal mask and strided inputs.
    Each position can only attend to earlier positions.
    Inputs have shape {lean}`Idx seqLen × Idx headDim`, returns {lean}`Idx seqLen × Idx headDim`. -/
def flashAttentionCausal
    (Q : StridedGpuTensor Float (Idx seqLen × Idx headDim))
    (K : StridedGpuTensor Float (Idx seqLen × Idx headDim))
    (V : StridedGpuTensor Float (Idx seqLen × Idx headDim)) :
    IO (StridedGpuTensor Float (Idx seqLen × Idx headDim)) := do
  -- Make contiguous copies if needed
  let qBuf ← if Q.isContiguous then pure Q.data.buffer else do
    let c ← Q.contiguous; pure c.data.buffer
  let kBuf ← if K.isContiguous then pure K.data.buffer else do
    let c ← K.contiguous; pure c.data.buffer
  let vBuf ← if V.isContiguous then pure V.data.buffer else do
    let c ← V.contiguous; pure c.data.buffer

  let result ← Metal.GpuBuffer.flashAttentionCausal qBuf kBuf vBuf seqLen.toUSize headDim.toUSize
  return ⟨StridedGpuBuffer.fromContiguous result #[seqLen, headDim]⟩

/-- Scaled dot-product attention using explicit GEMM operations.
    This version uses {name}`gemm` instead of a fused kernel, which allows
    transpose views without materializing copies.
    Inputs have shape {lean}`Idx seqLen × Idx headDim`, returns {lean}`Idx seqLen × Idx headDim`. -/
def scaledDotProductAttention
    (Q : StridedGpuTensor Float (Idx seqLen × Idx headDim))
    (K : StridedGpuTensor Float (Idx seqLen × Idx headDim))
    (V : StridedGpuTensor Float (Idx seqLen × Idx headDim)) :
    IO (StridedGpuTensor Float (Idx seqLen × Idx headDim)) := do
  -- Step 1: Compute attention scores = Q @ K^T
  -- K^T is an O(1) view operation!
  let K_T := K.transpose  -- (seqLen, headDim) -> (headDim, seqLen)
  let scores ← gemm Q K_T  -- (seqLen, headDim) @ (headDim, seqLen) = (seqLen, seqLen)

  -- Step 2: Scale by 1/sqrt(headDim)
  let scaleVal := 1.0 / Float.sqrt headDim.toFloat
  let scaledScores ← Metal.GpuBuffer.scale (seqLen * seqLen).toUSize scaleVal scores.data.buffer
  let scaledTensor : StridedGpuTensor Float (Idx seqLen × Idx seqLen) :=
    ⟨StridedGpuBuffer.fromContiguous scaledScores #[seqLen, seqLen]⟩

  -- Step 3: Softmax (row-wise)
  let attnWeights ← Metal.GpuBuffer.softmax scaledTensor.data.buffer seqLen.toUSize seqLen.toUSize
  let attnTensor : StridedGpuTensor Float (Idx seqLen × Idx seqLen) :=
    ⟨StridedGpuBuffer.fromContiguous attnWeights #[seqLen, seqLen]⟩

  -- Step 4: Apply to values: output = attnWeights @ V
  gemm attnTensor V  -- (seqLen, seqLen) @ (seqLen, headDim) = (seqLen, headDim)

end StridedGpuTensor

end SciLean
