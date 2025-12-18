/-
Copyright (c) 2024 SciLean contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: SciLean contributors
-/
import SciLean.FFI.Metal.StridedBuffer
import SciLean.Data.DataArray

-- Type-safe strided GPU tensor with shape encoded in the type

namespace SciLean

/-- Type-safe strided GPU tensor.
    Shape is tracked in the type via IndexType, layout metadata enables O(1) views. -/
structure StridedGpuTensor (α : Type) [PlainDataType α] (ι : Type) {n : outParam ℕ} [IndexType ι n] : Type where
  /-- Underlying strided GPU buffer -/
  data : StridedGpuBuffer α

instance [PlainDataType α] [IndexType ι n] : Nonempty (StridedGpuTensor α ι) :=
  ⟨⟨Classical.arbitrary _⟩⟩

namespace StridedGpuTensor

variable {α : Type} [PlainDataType α]
variable {ι : Type} {n : ℕ} [IndexType ι n]

-- Layout queries (delegate to underlying buffer)

/-- Number of dimensions -/
def rank (t : StridedGpuTensor α ι) : Nat := t.data.rank

/-- Shape as array -/
def shape (t : StridedGpuTensor α ι) : Array Nat := t.data.shape

/-- Strides as array -/
def strides (t : StridedGpuTensor α ι) : Array Nat := t.data.strides

/-- Check if contiguous -/
def isContiguous (t : StridedGpuTensor α ι) : Bool := t.data.isContiguous

/-- Check if transposed -/
def isTransposed (t : StridedGpuTensor α ι) : Bool := t.data.isTransposed

-- O(1) view operations
-- Note: Type changes to reflect the new shape

variable {m k p : ℕ}

/-- Transpose last two dimensions - O(1), no data copy.
    Changes type from (Idx m × Idx k) to (Idx k × Idx m). -/
def transpose (t : StridedGpuTensor α (Idx m × Idx k)) :
    StridedGpuTensor α (Idx k × Idx m) :=
  ⟨t.data.transpose⟩

/-- Transpose for 3D batched tensors - swaps last two dims.
    (batch × m × k) → (batch × k × m) -/
def batchTranspose (t : StridedGpuTensor α (Idx p × Idx m × Idx k)) :
    StridedGpuTensor α (Idx p × Idx k × Idx m) :=
  ⟨t.data.transpose⟩

/-- Make contiguous copy if needed -/
def contiguous (t : StridedGpuTensor α ι) : IO (StridedGpuTensor α ι) := do
  let newData ← t.data.contiguous
  return ⟨newData⟩

-- Construction

/-- Create from StridedGpuBuffer -/
def ofBuffer (buf : StridedGpuBuffer α) : StridedGpuTensor α ι := ⟨buf⟩

/-- Create from contiguous GPU buffer with shape derived from IndexType -/
def fromContiguousBuffer (buffer : Metal.GpuBuffer) (shape : Array Nat) : StridedGpuTensor α ι :=
  ⟨StridedGpuBuffer.fromContiguous buffer shape⟩

-- Conversion to/from legacy GpuTensor (for migration)

/-- Get underlying GPU buffer -/
def toGpuBuffer (t : StridedGpuTensor α ι) : Metal.GpuBuffer := t.data.buffer

end StridedGpuTensor

-- Matrix operations for strided tensors

namespace StridedGpuTensor

variable {m k p : ℕ}

/-- Matrix multiply with strided inputs.
    Automatically dispatches to gemm, gemmTN, or gemmNT based on transpose state.
    This enables O(1) transpose views to work efficiently with GPU GEMM. -/
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
    A is (batch, m, k), B is (batch, k, p), returns C (batch, m, p).
    Automatically dispatches based on transpose state of last two dims. -/
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

/-- Batched GEMM backward using O(1) transpose views.
    For each batch b: C\_b = A\_b @ B\_b, so:
    - dA\_b = dC\_b @ B\_b^T
    - dB\_b = A\_b^T @ dC\_b
    Transposes are O(1) metadata swaps! -/
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
    Computes: output = softmax(Q @ K^T / sqrt(headDim)) @ V

    Q, K, V have shape (seqLen, headDim).
    Returns output of shape (seqLen, headDim).

    Handles transposed inputs by making contiguous copies if needed.
    The internal kernel already handles K^T, so we only need to ensure
    Q and V are in the expected layout. -/
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
    Position i can only attend to positions ≤ i (for autoregressive models).

    Q, K, V have shape (seqLen, headDim).
    Returns output of shape (seqLen, headDim). -/
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

/-- Scaled dot-product attention using strided GEMM.
    Computes: output = softmax(Q @ K^T / sqrt(headDim)) @ V

    This version uses explicit GEMM operations instead of a fused kernel,
    which allows K to be a transposed view (O(1) transpose).

    Note: Less memory-efficient than flash attention for long sequences,
    but more flexible with strided inputs. -/
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
