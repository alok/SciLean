/-
Copyright (c) 2024 SciLean contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: SciLean contributors
-/
import SciLean.FFI.Metal
import SciLean.Data.Tensor.Layout

-- GPU buffer with N-D layout for O(1) view operations

namespace SciLean

/-- GPU buffer paired with N-D layout metadata.
    Enables O(1) transpose, slice, permute without data copies.
    The type parameter α tracks the element type (usually Float). -/
structure GpuBufferView (α : Type) where
  /-- Underlying GPU memory -/
  buffer : Metal.GpuBuffer
  /-- N-D layout with shape, strides, offset -/
  layout : TensorLayout
  deriving Nonempty

namespace GpuBufferView

variable {α : Type}

/-- Number of dimensions -/
def rank (buf : GpuBufferView α) : Nat := buf.layout.rank

/-- Shape as array -/
def shape (buf : GpuBufferView α) : Array Nat := buf.layout.shape

/-- Strides as array -/
def strides (buf : GpuBufferView α) : Array Nat := buf.layout.strides

/-- Total number of elements -/
def numel (buf : GpuBufferView α) : Nat := buf.layout.numel

/-- Check if layout is contiguous (no gaps, standard row-major order) -/
def isContiguous (buf : GpuBufferView α) : Bool := buf.layout.isContiguous

/-- Check if last two dims are transposed -/
def isTransposed (buf : GpuBufferView α) : Bool := buf.layout.isTransposed

/-- Row stride for matrix operations -/
def rowStride (buf : GpuBufferView α) : Nat := buf.layout.rowStride

-- O(1) view operations - no data copy, just metadata changes

/-- Transpose last two dimensions - O(1), no data movement -/
def transpose (buf : GpuBufferView α) : GpuBufferView α :=
  ⟨buf.buffer, buf.layout.transpose⟩

/-- Permute dimensions according to perm array - O(1) -/
def permute (buf : GpuBufferView α) (perm : Array Nat) : GpuBufferView α :=
  ⟨buf.buffer, buf.layout.permute perm⟩

/-- Slice along a dimension - O(1) -/
def slice (buf : GpuBufferView α) (dim : Nat) (start len : Nat) : GpuBufferView α :=
  ⟨buf.buffer, buf.layout.slice dim start len⟩

/-- Squeeze dimension of size 1 - O(1) -/
def squeeze (buf : GpuBufferView α) (dim : Nat) : GpuBufferView α :=
  ⟨buf.buffer, buf.layout.squeeze dim⟩

/-- Unsqueeze - add dimension of size 1 - O(1) -/
def unsqueeze (buf : GpuBufferView α) (dim : Nat) : GpuBufferView α :=
  ⟨buf.buffer, buf.layout.unsqueeze dim⟩

/-- Create from contiguous GPU buffer with given shape -/
def fromContiguous (buffer : Metal.GpuBuffer) (shape : Array Nat) : GpuBufferView α :=
  ⟨buffer, TensorLayout.contiguous shape⟩

/-- Create contiguous copy if not already contiguous.
    Returns same buffer if already contiguous, otherwise copies data.
    Uses GPU layout copy kernel for efficient data movement. -/
def contiguous (buf : GpuBufferView α) : IO (GpuBufferView α) := do
  if buf.isContiguous then
    return buf
  else
    -- Convert Nat arrays to USize arrays for FFI
    let shapeU : Array USize := buf.layout.shape.map (·.toUSize)
    let stridesU : Array USize := buf.layout.strides.map (·.toUSize)
    let offsetU : USize := buf.layout.offset.toUSize
    -- Call Metal layout copy kernel
    let newBuffer ← Metal.GpuBuffer.copyLayout buf.buffer shapeU stridesU offsetU
    -- Return with contiguous layout
    let newLayout := TensorLayout.contiguous buf.layout.shape
    return ⟨newBuffer, newLayout⟩

/-- Get dimension at index -/
def dim (buf : GpuBufferView α) (i : Nat) : Nat :=
  buf.layout.shape.getD i 0

/-- Get stride at index -/
def strideAt (buf : GpuBufferView α) (i : Nat) : Nat :=
  buf.layout.strides.getD i 0

end GpuBufferView

end SciLean
