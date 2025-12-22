/-
Copyright (c) 2024 SciLean contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: SciLean contributors
-/
import SciLean.Data.DataArray
import SciLean.Data.Tensor.StridedGpuTensor
import SciLean.Data.Tensor.Layout

namespace SciLean

/-!
# Device-Tracked Tensor Type

{lit}`Tensor d α ι` is a tensor with element type {lit}`α` and shape tracked by index type
{lit}`ι`, where {lit}`d : Device` specifies whether data resides on CPU or GPU (Metal).

The device is encoded in the type, providing compile-time guarantees about data location
and eliminating runtime dispatch overhead. This enables:
- Type-safe transfers between devices via explicit transfer functions
- Device-specific optimizations in tensor operation typeclasses
- Zero-cost abstraction over {name}`DataArrayN` for CPU tensors
-/

/-- Compute device for tensor operations. -/
inductive Device where
  | cpu    -- CPU with DataArrayN storage
  | metal  -- GPU via Metal with strided layout metadata
  deriving DecidableEq, Repr, Inhabited

namespace Device

/-- Check if Metal GPU is available at runtime. -/
def metalAvailable : IO Bool := do
  -- TODO: Add actual Metal availability check via FFI
  return true

/-- Get best available device (prefers GPU). -/
def best : IO Device := do
  if ← metalAvailable then
    return .metal
  else
    return .cpu

instance : ToString Device where
  toString
    | .cpu => "cpu"
    | .metal => "metal"

end Device

/-- CPU tensor wrapping {name}`DataArrayN` with device tracking. -/
structure CpuTensor (α : Type) [PlainDataType α] (ι : Type) {n : outParam ℕ} [IndexType ι n] : Type where
  /-- The underlying CPU array. -/
  data : DataArrayN α ι

/-- GPU tensor with layout metadata and device tracking. -/
abbrev GpuTensor (α : Type) [PlainDataType α] (ι : Type) {n : outParam ℕ} [IndexType ι n] : Type :=
  StridedGpuTensor α ι

/-- Device-indexed tensor type. Maps {name}`Device` to appropriate storage type. -/
abbrev Tensor (d : Device) (α : Type) [PlainDataType α] (ι : Type) {n : outParam ℕ} [IndexType ι n] : Type :=
  match d with
  | .cpu => CpuTensor α ι
  | .metal => GpuTensor α ι

namespace CpuTensor

variable {α : Type} [PlainDataType α]
variable {ι : Type} {n : ℕ} [IndexType ι n]

/-! ## Construction -/

/-- Create from {name}`DataArrayN`. -/
@[inline]
def ofDataArrayN (arr : DataArrayN α ι) : CpuTensor α ι := ⟨arr⟩

/-- Extract {name}`DataArrayN`. -/
@[inline]
def toDataArrayN (t : CpuTensor α ι) : DataArrayN α ι := t.data

/-! ## Zero-Cost Coercions -/

instance : Coe (DataArrayN α ι) (CpuTensor α ι) where
  coe := ofDataArrayN

instance : Coe (CpuTensor α ι) (DataArrayN α ι) where
  coe := toDataArrayN

/-! ## Basic Properties -/

/-- Number of elements. -/
def size (_ : CpuTensor α ι) : ℕ := n

/-- Size as {name}`USize`. -/
def usize (_ : CpuTensor α ι) : USize := n.toUSize

/-! ## Element Access -/

/-- Get element at index. -/
@[inline]
def get (t : CpuTensor α ι) (i : ι) : α := t.data.get i

/-- Set element at index. -/
@[inline]
def set (t : CpuTensor α ι) (i : ι) (v : α) : CpuTensor α ι :=
  ⟨t.data.set i v⟩

instance : GetElem (CpuTensor α ι) ι α (fun _ _ => True) where
  getElem t i _ := t.get i

instance : SetElem (CpuTensor α ι) ι α (fun _ _ => True) where
  setElem t i v _ := t.set i v
  setElem_valid := sorry_proof

end CpuTensor

namespace GpuTensor

variable {α : Type} [PlainDataType α]
variable {ι : Type} {n : ℕ} [IndexType ι n]

/-! ## Basic Properties -/

/-- Number of elements. -/
def size (_ : GpuTensor α ι) : ℕ := n

/-- Size as {name}`USize`. -/
def usize (_ : GpuTensor α ι) : USize := n.toUSize

/-! ## Layout and Views -/

/-- Rank of the tensor. -/
@[inline]
def rank (t : GpuTensor α ι) : Nat := StridedGpuTensor.rank t

/-- Shape array. -/
@[inline]
def shape (t : GpuTensor α ι) : Array Nat := StridedGpuTensor.shape t

/-- Strides array. -/
@[inline]
def strides (t : GpuTensor α ι) : Array Nat := StridedGpuTensor.strides t

/-- Whether tensor is contiguous. -/
@[inline]
def isContiguous (t : GpuTensor α ι) : Bool := StridedGpuTensor.isContiguous t

/-- Whether tensor is a simple transpose view. -/
@[inline]
def isTransposed (t : GpuTensor α ι) : Bool := StridedGpuTensor.isTransposed t

/-- O(1) transpose view. -/
@[inline]
def transpose (t : GpuTensor α (Idx m × Idx k)) : GpuTensor α (Idx k × Idx m) :=
  StridedGpuTensor.transpose t

/-- O(1) batch transpose view. -/
@[inline]
def batchTranspose (t : GpuTensor α (Idx p × Idx m × Idx k)) : GpuTensor α (Idx p × Idx k × Idx m) :=
  StridedGpuTensor.batchTranspose t

/-- Make a contiguous copy if needed. -/
@[inline]
def contiguous (t : GpuTensor α ι) : IO (GpuTensor α ι) :=
  StridedGpuTensor.contiguous t

/-- Ensure the tensor is contiguous (no-op if already contiguous). -/
@[inline]
def ensureContiguous (t : GpuTensor α ι) : IO (GpuTensor α ι) :=
  StridedGpuTensor.ensureContiguous t

/-! ## Buffer Access -/

/-- Get underlying {name}`Metal.GpuBuffer`. -/
@[inline]
def toGpuBuffer (t : GpuTensor α ι) : Metal.GpuBuffer :=
  StridedGpuTensor.toGpuBuffer t

/-! ## Construction -/

/-- Wrap an existing strided buffer. -/
@[inline]
def ofBuffer (buf : StridedGpuBuffer α) : GpuTensor α ι :=
  StridedGpuTensor.ofBuffer buf

/-! ## Contiguous Constructors -/

/-- Create from a contiguous {name}`Metal.GpuBuffer` and shape. -/
@[inline]
def fromContiguousBuffer (buffer : Metal.GpuBuffer) (shape : Array Nat) : GpuTensor α ι :=
  StridedGpuTensor.fromContiguousBuffer (ι:=ι) buffer shape

/-- Create from a contiguous {name}`Metal.GpuBuffer` using {name}`IndexTypeShape`. -/
@[inline]
def fromContiguous (buffer : Metal.GpuBuffer) [IndexTypeShape ι n] : GpuTensor α ι :=
  StridedGpuTensor.fromContiguous (ι:=ι) buffer

end GpuTensor

/-! ## Notation -/

/-- Notation for tensor types: {lit}`α^[ι]@d` for {lit}`Tensor d α ι`. -/
scoped notation:max α "^[" ι "]@cpu" => CpuTensor α ι
scoped notation:max α "^[" ι "]@metal" => GpuTensor α ι

/-- Notation for strided GPU tensors. -/
scoped notation:max α "^[" ι "]ᵍ" => GpuTensor α ι

end SciLean
