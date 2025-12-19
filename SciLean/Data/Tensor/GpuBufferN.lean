/-
Copyright (c) 2024 SciLean contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: SciLean contributors
-/
import SciLean.FFI.Metal
import SciLean.Data.IndexType.Basic
import SciLean.Data.DataArray.PlainDataType
import SciLean.Data.DataArray.DataArray

namespace SciLean

set_option linter.deprecated false

/-!
# Shape-tracked GPU Buffer (Legacy)

Legacy contiguous GPU buffer. {lit}`GpuBufferN` {lit}`α ι` is a GPU-resident buffer with element type
{lit}`α` and shape tracked by index type {lit}`ι`. This mirrors {name}`DataArrayN` but lives on GPU
(Metal) rather than CPU.

Prefer {lit}`StridedGpuTensor` or {lit}`GpuTensor` for new code. Use {lit}`LegacyInterop` only as a
temporary bridge during migration.
-/

/-- Legacy GPU buffer with shape tracked in the type.
    Mirrors {name}`DataArrayN` but data lives on GPU (Metal).

    Unlike {name}`DataArrayN`, we don't carry a size proof because {name}`Metal.GpuBuffer` is opaque.
    Instead, the size invariant is maintained by construction. -/
@[deprecated "Use StridedGpuTensor / GpuTensor. LegacyInterop provides migration helpers." (since := "v4.26")]
structure GpuBufferN (α : Type*) [PlainDataType α] (ι : Type*) {n : outParam ℕ} [IndexType ι n] : Type where
  /-- The underlying {name}`Metal.GpuBuffer` handle. -/
  buffer : Metal.GpuBuffer

namespace GpuBufferN

variable {α : Type*} [pd : PlainDataType α]
variable {ι : Type*} {n : ℕ} [IndexType ι n]

/-- Create a GPU buffer from a {name}`DataArrayN` (CPU → GPU transfer). -/
@[deprecated "Use StridedGpuTensor.fromContiguous or LegacyInterop.ofGpuBufferN." (since := "v4.26")]
def fromDataArrayN (arr : DataArrayN α ι) : IO (GpuBufferN α ι) := do
  let gpuBuf ← Metal.GpuBuffer.fromByteArray arr.data.byteData
  return ⟨gpuBuf⟩

/-- Copy GPU buffer back to CPU as {name}`DataArrayN` (GPU → CPU transfer). -/
@[deprecated "Use GpuTensor.toCpu or LegacyInterop.toGpuBufferN then toDataArrayN." (since := "v4.26")]
def toDataArrayN (buf : GpuBufferN α ι) : IO (DataArrayN α ι) := do
  let bytes ← Metal.GpuBuffer.toByteArray buf.buffer
  let data : DataArray α := ⟨bytes, sorry_proof⟩
  return ⟨data, sorry_proof⟩

/-- Allocate an uninitialized GPU buffer of the given shape. -/
@[deprecated "Use StridedGpuTensor.fromContiguousBuffer or Metal.GpuBuffer directly." (since := "v4.26")]
def alloc : IO (GpuBufferN α ι) := do
  let elemBytes := pd.btype.bytes
  let gpuBuf ← Metal.GpuBuffer.alloc (n.toUSize * elemBytes)
  return ⟨gpuBuf⟩

/-- Number of elements in the buffer (known at compile time from the type). -/
@[deprecated "Use GpuTensor.size / StridedGpuTensor.rank or shape." (since := "v4.26")]
def size (_ : GpuBufferN α ι) : ℕ := n

/-- Size as {name}`USize` for FFI calls. -/
@[deprecated "Use GpuTensor.usize / StridedGpuTensor.rank or shape." (since := "v4.26")]
def usize (_ : GpuBufferN α ι) : USize := n.toUSize

/-- Get underlying buffer size in bytes (runtime check). -/
@[deprecated "Use StridedGpuTensor.shape/strides and Metal.GpuBuffer sizing." (since := "v4.26")]
def sizeBytes (buf : GpuBufferN α ι) : USize :=
  Metal.GpuBuffer.sizeBytes buf.buffer

end GpuBufferN

end SciLean
