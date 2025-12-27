/-
Copyright (c) 2024 SciLean contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: SciLean contributors
-/
import SciLean.Data.Tensor.Basic
import SciLean.Data.IndexType.Shape
import SciLean.VersoPrelude

namespace SciLean

/-!
# Device Transfer Operations

Type-safe transfers between {name}`CpuTensor` and {name}`GpuTensor`. The device is tracked in the type,
so transfers are explicit and visible in function signatures.

Usage: call {lit}`CpuTensor.toGpu` to upload, do GPU computation, then {lit}`GpuTensor.toCpu` to download.
-/

variable {α : Type} [PlainDataType α]
variable {ι : Type} {n : ℕ} [IndexType ι n]

/-! ## CPU → GPU Transfer -/

/-- Transfer CPU tensor to GPU (contiguous layout). -/
def CpuTensor.toGpu (t : CpuTensor α ι) [IndexTypeShape ι n] : IO (GpuTensor α ι) := do
  let gpuBuf ← Metal.GpuBuffer.fromByteArray t.data.data.byteData
  return GpuTensor.fromContiguous (ι:=ι) gpuBuf

/-! ## GPU → CPU Transfer -/

/-- Transfer GPU tensor to CPU (makes contiguous copy if needed). -/
def GpuTensor.toCpu (t : GpuTensor α ι) : IO (CpuTensor α ι) := do
  let tContig ← GpuTensor.ensureContiguous t
  let bytes ← Metal.GpuBuffer.toByteArray tContig.data.buffer
  let data : DataArray α := ⟨bytes, sorry_proof⟩
  let arr : DataArrayN α ι := ⟨data, sorry_proof⟩
  return ⟨arr⟩

/-! ## Polymorphic Transfer API -/

/-- Transfer a tensor to GPU. Identity for GPU tensors, copy for CPU tensors. -/
class ToGpu (T : Type) (G : outParam Type) where
  toGpu : T → IO G

instance [IndexTypeShape ι n] : ToGpu (CpuTensor α ι) (GpuTensor α ι) where
  toGpu t := CpuTensor.toGpu (ι:=ι) t

instance : ToGpu (GpuTensor α ι) (GpuTensor α ι) where
  toGpu t := pure t

/-- Transfer a tensor to CPU. Identity for CPU tensors, copy for GPU tensors. -/
class ToCpu (T : Type) (C : outParam Type) where
  toCpu : T → IO C

instance : ToCpu (GpuTensor α ι) (CpuTensor α ι) where
  toCpu := GpuTensor.toCpu

instance : ToCpu (CpuTensor α ι) (CpuTensor α ι) where
  toCpu t := pure t

/-! ## Convenience Functions -/

/-- Run a GPU computation on CPU data, handling transfers automatically. -/
def withGpu (input : CpuTensor α ι) [IndexTypeShape ι n]
    (f : GpuTensor α ι → IO (GpuTensor α ι))
    : IO (CpuTensor α ι) := do
  let gpuIn ← input.toGpu
  let gpuOut ← f gpuIn
  gpuOut.toCpu

/-- Run a GPU computation on CPU data with a different output shape. -/
def withGpu' {β : Type} [PlainDataType β] {κ : Type} {m : ℕ} [IndexType κ m]
    (input : CpuTensor α ι) [IndexTypeShape ι n]
    (f : GpuTensor α ι → IO (GpuTensor β κ))
    : IO (CpuTensor β κ) := do
  let gpuIn ← input.toGpu
  let gpuOut ← f gpuIn
  gpuOut.toCpu

/-- Transfer {name}`DataArrayN` directly to GPU. -/
def DataArrayN.toGpu (arr : DataArrayN α ι) [IndexTypeShape ι n] : IO (GpuTensor α ι) :=
  CpuTensor.toGpu ⟨arr⟩

/-- Alias for compatibility with {name}`Tensor` notation. -/
abbrev Tensor.toGpu [ToGpu T G] (t : T) : IO G := ToGpu.toGpu t
abbrev Tensor.toCpu [ToCpu T C] (t : T) : IO C := ToCpu.toCpu t

end SciLean
