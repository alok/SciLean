/-
Copyright (c) 2024 SciLean contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: SciLean contributors
-/
import SciLean.Data.DataArray
import SciLean.Data.Tensor.GpuTensor
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
  | metal  -- GPU via Metal with layout metadata
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


/-! ## Notation -/

/-- Notation for tensor types: {lit}`α^[ι]@d` for {lit}`Tensor d α ι`. -/
scoped notation:max α "^[" ι "]@cpu" => CpuTensor α ι
scoped notation:max α "^[" ι "]@metal" => GpuTensor α ι

/-- Notation for GPU tensors. -/
scoped notation:max α "^[" ι "]ᵍ" => GpuTensor α ι

end SciLean
