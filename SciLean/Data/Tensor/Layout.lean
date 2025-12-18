/-
Copyright (c) 2024 SciLean contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: SciLean contributors
-/

namespace SciLean

/-- N-dimensional tensor layout with strides for O(1) view operations.

Enables PyTorch-style strided tensors where transpose, slice, permute are O(1)
operations that just modify metadata without copying data.

Row-major (C-style) layout: last dimension is contiguous in memory. -/
structure TensorLayout where
  /-- Shape of each dimension -/
  shape : Array Nat
  /-- Elements between consecutive indices per dimension -/
  strides : Array Nat
  /-- Element offset into underlying buffer -/
  offset : Nat := 0
  deriving Repr, BEq, Inhabited

namespace TensorLayout

/-- Number of dimensions -/
def rank (l : TensorLayout) : Nat := l.shape.size

/-- Total number of elements -/
def numel (l : TensorLayout) : Nat := l.shape.foldl (· * ·) 1

/-- Check if layout is row-major contiguous (no gaps, no transposition) -/
def isContiguous (l : TensorLayout) : Bool :=
  let n := l.rank
  if n == 0 then true
  else if l.strides.size != n then false
  else Id.run do
    let mut expected := 1
    for i in [0:n] do
      let idx := n - 1 - i
      if l.strides[idx]! != expected then return false
      expected := expected * l.shape[idx]!
    return true

/-- Minimum storage size needed in elements -/
def storageSize (l : TensorLayout) : Nat :=
  if l.rank == 0 then 1
  else l.offset + Id.run do
    let mut maxIdx := 0
    for i in [0:l.rank] do
      maxIdx := maxIdx + (l.shape[i]! - 1) * l.strides[i]!
    return maxIdx + 1

/-- Create contiguous row-major layout from shape -/
def contiguous (shape : Array Nat) : TensorLayout :=
  let n := shape.size
  if n == 0 then ⟨#[], #[], 0⟩
  else
    let strides := Id.run do
      let mut s := Array.mk (List.replicate n 1)
      for i in [1:n] do
        let idx := n - 1 - i
        s := s.set! idx (s[idx + 1]! * shape[idx + 1]!)
      return s
    ⟨shape, strides, 0⟩

/-- Permute dimensions - O(1), just reorders shape/strides arrays -/
def permute (l : TensorLayout) (perm : Array Nat) : TensorLayout :=
  if perm.size != l.rank then l  -- invalid permutation
  else
    let newShape := perm.map fun i => l.shape.getD i 0
    let newStrides := perm.map fun i => l.strides.getD i 0
    ⟨newShape, newStrides, l.offset⟩

/-- Transpose last two dimensions (matrix transpose) - O(1) -/
def transpose (l : TensorLayout) : TensorLayout :=
  let n := l.rank
  if n < 2 then l
  else
    let newShape := l.shape.set! (n-2) (l.shape.getD (n-1) 0)
                  |>.set! (n-1) (l.shape.getD (n-2) 0)
    let newStrides := l.strides.set! (n-2) (l.strides.getD (n-1) 0)
                    |>.set! (n-1) (l.strides.getD (n-2) 0)
    ⟨newShape, newStrides, l.offset⟩

/-- Slice along a dimension - O(1) -/
def slice (l : TensorLayout) (dim : Nat) (start len : Nat) : TensorLayout :=
  if dim >= l.rank then l
  else if start + len > l.shape.getD dim 0 then l
  else
    let newShape := l.shape.set! dim len
    let newOffset := l.offset + start * l.strides.getD dim 0
    ⟨newShape, l.strides, newOffset⟩

/-- Remove element at index from array -/
private def eraseAt (arr : Array α) (idx : Nat) : Array α :=
  let left := arr.toList.take idx
  let right := arr.toList.drop (idx + 1)
  Array.mk (left ++ right)

/-- Squeeze a dimension of size 1 - O(1) -/
def squeeze (l : TensorLayout) (dim : Nat) : TensorLayout :=
  if dim >= l.rank then l
  else if l.shape.getD dim 0 != 1 then l
  else
    ⟨eraseAt l.shape dim, eraseAt l.strides dim, l.offset⟩

/-- Insert element at position in array -/
private def insertAt (arr : Array α) (idx : Nat) (val : α) : Array α :=
  let left := arr.toList.take idx
  let right := arr.toList.drop idx
  Array.mk (left ++ [val] ++ right)

/-- Unsqueeze - add dimension of size 1 at position - O(1) -/
def unsqueeze (l : TensorLayout) (dim : Nat) : TensorLayout :=
  if dim > l.rank then l
  else
    ⟨insertAt l.shape dim 1, insertAt l.strides dim 0, l.offset⟩

/-- Compute flat index from multi-dimensional index -/
def flatIndex (l : TensorLayout) (idx : Array Nat) : Nat :=
  if idx.size != l.rank then 0
  else l.offset + Id.run do
    let mut flat := 0
    for i in [0:l.rank] do
      flat := flat + idx[i]! * l.strides[i]!
    return flat

/-- Check if tensor is transposed (last two dims swapped vs contiguous) -/
def isTransposed (l : TensorLayout) : Bool :=
  let n := l.rank
  if n < 2 then false
  else l.strides.getD (n-2) 0 < l.strides.getD (n-1) 0

/-- Get the row stride for 2D matrix operations (larger of last two strides) -/
def rowStride (l : TensorLayout) : Nat :=
  let n := l.rank
  if n < 2 then 1
  else max (l.strides.getD (n-2) 0) (l.strides.getD (n-1) 0)

/-- Get shape as a string for debugging -/
def shapeStr (l : TensorLayout) : String :=
  s!"({l.shape.toList})"

end TensorLayout

end SciLean
