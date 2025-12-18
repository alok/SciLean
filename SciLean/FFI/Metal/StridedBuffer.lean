/-
Copyright (c) 2024 SciLean contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: SciLean contributors
-/
import SciLean.FFI.Metal
import SciLean.Data.Tensor.Layout

-- GPU buffer with N-D strided layout for O(1) view operations

namespace SciLean

/-- GPU buffer paired with N-D strided layout metadata.
    Enables O(1) transpose, slice, permute without data copies.
    The type parameter α tracks the element type (usually Float). -/
structure StridedGpuBuffer (α : Type) where
  /-- Underlying GPU memory -/
  buffer : Metal.GpuBuffer
  /-- N-D layout with shape, strides, offset -/
  layout : TensorLayout
  deriving Nonempty

namespace StridedGpuBuffer

variable {α : Type}

/-- Number of dimensions -/
def rank (buf : StridedGpuBuffer α) : Nat := buf.layout.rank

/-- Shape as array -/
def shape (buf : StridedGpuBuffer α) : Array Nat := buf.layout.shape

/-- Strides as array -/
def strides (buf : StridedGpuBuffer α) : Array Nat := buf.layout.strides

/-- Total number of elements -/
def numel (buf : StridedGpuBuffer α) : Nat := buf.layout.numel

/-- Check if layout is contiguous (no gaps, standard row-major order) -/
def isContiguous (buf : StridedGpuBuffer α) : Bool := buf.layout.isContiguous

/-- Check if last two dims are transposed -/
def isTransposed (buf : StridedGpuBuffer α) : Bool := buf.layout.isTransposed

/-- Row stride for matrix operations -/
def rowStride (buf : StridedGpuBuffer α) : Nat := buf.layout.rowStride

-- O(1) view operations - no data copy, just metadata changes

/-- Transpose last two dimensions - O(1), no data movement -/
def transpose (buf : StridedGpuBuffer α) : StridedGpuBuffer α :=
  ⟨buf.buffer, buf.layout.transpose⟩

/-- Permute dimensions according to perm array - O(1) -/
def permute (buf : StridedGpuBuffer α) (perm : Array Nat) : StridedGpuBuffer α :=
  ⟨buf.buffer, buf.layout.permute perm⟩

/-- Slice along a dimension - O(1) -/
def slice (buf : StridedGpuBuffer α) (dim : Nat) (start len : Nat) : StridedGpuBuffer α :=
  ⟨buf.buffer, buf.layout.slice dim start len⟩

/-- Squeeze dimension of size 1 - O(1) -/
def squeeze (buf : StridedGpuBuffer α) (dim : Nat) : StridedGpuBuffer α :=
  ⟨buf.buffer, buf.layout.squeeze dim⟩

/-- Unsqueeze - add dimension of size 1 - O(1) -/
def unsqueeze (buf : StridedGpuBuffer α) (dim : Nat) : StridedGpuBuffer α :=
  ⟨buf.buffer, buf.layout.unsqueeze dim⟩

/-- Create from contiguous GPU buffer with given shape -/
def fromContiguous (buffer : Metal.GpuBuffer) (shape : Array Nat) : StridedGpuBuffer α :=
  ⟨buffer, TensorLayout.contiguous shape⟩

/-- Create contiguous copy if not already contiguous.
    Returns same buffer if already contiguous, otherwise copies data. -/
def contiguous (buf : StridedGpuBuffer α) : IO (StridedGpuBuffer α) := do
  if buf.isContiguous then
    return buf
  else
    -- TODO: Implement strided copy kernel
    -- For now, this is a placeholder that just returns the buffer
    -- Real implementation needs Metal kernel to copy with strides
    return buf

/-- Get dimension at index -/
def dim (buf : StridedGpuBuffer α) (i : Nat) : Nat :=
  buf.layout.shape.getD i 0

/-- Get stride at index -/
def strideAt (buf : StridedGpuBuffer α) (i : Nat) : Nat :=
  buf.layout.strides.getD i 0

end StridedGpuBuffer

end SciLean
