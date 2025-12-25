/-
Copyright (c) 2024 SciLean Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Alok Singh
-/
import SciLean.Kernel.DType
import SciLean.Kernel.Ops
import SciLean.Kernel.Spec
import SciLean.Kernel.Axioms
import SciLean.Kernel.AD
import SciLean.Data.DataArray.DataArray
import SciLean.Data.DataArray.PlainDataType
import SciLean.AD.HasRevFDeriv
import SciLean.Tactic.DataSynth.Attr
import SciLean.VersoPrelude

namespace SciLean.Kernel

open SciLean

/-!
# Kernel ↔ DataArrayN Integration

This module bridges the byte-level kernel operations with SciLean's typed
{name}`DataArrayN` interface. It provides:

1. **DType ↔ PlainDataType mapping** - Connect runtime {name}`DType` to compile-time
   {name}`PlainDataType`
2. **Typed kernel operations** - GEMM, GEMV, etc. on {lit}`Float^[m,n]`
3. **AD rule registration** - {lit}`@[data_synth]` instances for {name}`HasRevFDeriv`

## Design

The kernel operates on {name}`ByteArray` with {name}`DType` tags.
{name}`DataArrayN` uses {name}`PlainDataType` for byte representation.
This module connects them:

```
Float^[m,n] ───extract───→ ByteArray ───kernel op───→ ByteArray ───wrap───→ Float^[p,q]
```

For efficiency, operations reuse the underlying ByteArray where possible.
-/

-- ============================================================================
-- DType ↔ PlainDataType Connection
-- ============================================================================

/-- Get DType for a PlainDataType-carrying type. -/
class HasDType (α : Type*) [PlainDataType α] where
  dtype : DType

instance : HasDType Float where
  dtype := .f64

instance : HasDType Float32 where
  dtype := .f32

/-- Get the DType for a given PlainDataType. -/
def getDType (α : Type*) [PlainDataType α] [HasDType α] : DType :=
  HasDType.dtype (α := α)

-- ============================================================================
-- Kernel Operations on DataArrayN
-- ============================================================================

/-! Typed kernel operations on {name}`DataArrayN`.
    Use {lit}`ArrayOps.add`, {lit}`ArrayOps.gemm`, etc. -/
namespace ArrayOps

variable {α : Type*} [PlainDataType α] [HasDType α]

-- Helper for wrapping results
private def wrapResult {ι : Type*} {n : ℕ} [Size' ι n] (bytes : ByteArray) : α^[ι] :=
  ⟨⟨bytes, sorry_proof⟩, sorry_proof⟩

-- Helper for extracting bytes
private def getBytes {ι : Type*} {n : ℕ} [Size' ι n] (arr : α^[ι]) : ByteArray :=
  arr.data.byteData

/-- Elementwise addition on DataArrayN. -/
@[inline]
def add {ι : Type*} {n : ℕ} [IndexType ι n] (a b : α^[ι]) : α^[ι] :=
  let dt := getDType α
  wrapResult (Typed.add dt (getBytes a) (getBytes b))

/-- Elementwise subtraction on DataArrayN. -/
@[inline]
def sub {ι : Type*} {n : ℕ} [IndexType ι n] (a b : α^[ι]) : α^[ι] :=
  let dt := getDType α
  wrapResult (Typed.sub dt (getBytes a) (getBytes b))

/-- Elementwise multiplication on DataArrayN. -/
@[inline]
def mul {ι : Type*} {n : ℕ} [IndexType ι n] (a b : α^[ι]) : α^[ι] :=
  let dt := getDType α
  wrapResult (Typed.mul dt (getBytes a) (getBytes b))

/-- Elementwise division on DataArrayN. -/
@[inline]
def div {ι : Type*} {n : ℕ} [IndexType ι n] (a b : α^[ι]) : α^[ι] :=
  let dt := getDType α
  wrapResult (Typed.div dt (getBytes a) (getBytes b))

/-- Elementwise negation on DataArrayN. -/
@[inline]
def neg {ι : Type*} {n : ℕ} [IndexType ι n] (a : α^[ι]) : α^[ι] :=
  let dt := getDType α
  wrapResult (Typed.neg dt (getBytes a))

/-- Elementwise exp on DataArrayN. -/
@[inline]
def exp {ι : Type*} {n : ℕ} [IndexType ι n] (a : α^[ι]) : α^[ι] :=
  let dt := getDType α
  wrapResult (Typed.exp dt (getBytes a))

/-- Elementwise log on DataArrayN. -/
@[inline]
def log {ι : Type*} {n : ℕ} [IndexType ι n] (a : α^[ι]) : α^[ι] :=
  let dt := getDType α
  wrapResult (Typed.log dt (getBytes a))

/-- Elementwise sqrt on DataArrayN. -/
@[inline]
def sqrt {ι : Type*} {n : ℕ} [IndexType ι n] (a : α^[ι]) : α^[ι] :=
  let dt := getDType α
  wrapResult (Typed.sqrt dt (getBytes a))

/-- Elementwise tanh on DataArrayN. -/
@[inline]
def tanh {ι : Type*} {n : ℕ} [IndexType ι n] (a : α^[ι]) : α^[ι] :=
  let dt := getDType α
  wrapResult (Typed.tanh dt (getBytes a))

/-- Sum reduction. -/
@[inline]
def sum {ι : Type*} {n : ℕ} [IndexType ι n] (a : α^[ι]) : Float :=
  let dt := getDType α
  Typed.sum dt (getBytes a)

/-- Softmax on a vector. -/
@[inline]
def softmax {n : ℕ} (a : α^[Idx n]) : α^[Idx n] :=
  let dt := getDType α
  wrapResult (Typed.softmax dt (getBytes a))

/-- Matrix-vector multiply: {lit}`y = A @ x`.
    {lit}`A : α^[Idx m, Idx k]`, {lit}`x : α^[Idx k]` → {lit}`y : α^[Idx m]`. -/
@[inline]
def gemv {m k : ℕ} (A : α^[Idx m, Idx k]) (x : α^[Idx k]) : α^[Idx m] :=
  let dt := getDType α
  wrapResult (Typed.gemv dt (getBytes A) (getBytes x) m k)

/-- Matrix multiply: {lit}`C = A @ B`.
    {lit}`A : α^[Idx m, Idx k]`, {lit}`B : α^[Idx k, Idx n]` → {lit}`C : α^[Idx m, Idx n]`. -/
@[inline]
def gemm {m k n : ℕ} (A : α^[Idx m, Idx k]) (B : α^[Idx k, Idx n]) : α^[Idx m, Idx n] :=
  let dt := getDType α
  wrapResult (Typed.gemm dt (getBytes A) (getBytes B) m k n)

/-- Matrix transpose: {lit}`Bᵀ[j,i] = B[i,j]`. -/
@[inline]
def transpose {m n : ℕ} (A : α^[Idx m, Idx n]) : α^[Idx n, Idx m] :=
  let dt := getDType α
  wrapResult (Typed.transpose dt (getBytes A) m n)

/-- Scaled vector addition: {lit}`y = α*x + β*y`. -/
@[inline]
def axpby {n : ℕ} (alpha : Float) (x : α^[Idx n]) (beta : Float) (y : α^[Idx n]) : α^[Idx n] :=
  let dt := getDType α
  wrapResult (Typed.axpby dt alpha (getBytes x) beta (getBytes y))

/-- Seed the RNG. -/
def seedRng (s : UInt64) : Unit := rngSeed s

/-- Create array filled with uniform random values in {lit}`[0, 1)`. -/
@[inline]
def randUniform {ι : Type*} {n : ℕ} [Size' ι n] : α^[ι] :=
  let dt := getDType α
  wrapResult (Typed.randUniform dt n)

/-- Create array filled with standard normal random values. -/
@[inline]
def randNormal {ι : Type*} {n : ℕ} [Size' ι n] : α^[ι] :=
  let dt := getDType α
  wrapResult (Typed.randNormal dt n)

end ArrayOps

-- ============================================================================
-- AD Rules for Typed Kernel Operations
-- ============================================================================

/-!
# AD Integration Notes

The full AD rules require integration with SciLean's {name}`HasRevFDeriv`,
which uses {name}`RCLike` (typically ℝ). The typed kernel operations work on
{lit}`Float^[n]`, which doesn't directly satisfy {name}`RCLike`.

For now, AD rules are specified in {lit}`Kernel/AD.lean` at the byte level.
Full integration with {lit}`@[data_synth]` requires:

1. Proving {lit}`Float^[n]` forms an {name}`AdjointSpace` over ℝ
2. Connecting kernel ops to the mathematical specs
3. Deriving {name}`HasRevFDeriv` from the byte-level adjoints

This is left for future work. The key insight is that the AD rules in
{lit}`AD.lean` describe the correct mathematical relationships; they just need
to be lifted to the typed interface.
-/

section ADHelpers

variable {α : Type*} [PlainDataType α] [HasDType α]

/-- Backward pass for {lit}`y = A @ x` (matrix-vector multiply).
    Given {lit}`dy`, returns {lit}`dx = Aᵀ @ dy`. -/
@[inline]
def gemv_backward_x {m k : ℕ} (A : α^[Idx m, Idx k]) (dy : α^[Idx m]) : α^[Idx k] :=
  let At := ArrayOps.transpose A
  ArrayOps.gemv At dy

/-- Backward pass for {lit}`C = A @ B` (matrix multiply) w.r.t. {lit}`A`.
    Given {lit}`dC`, returns {lit}`dA = dC @ Bᵀ`. -/
@[inline]
def gemm_backward_A {m k n : ℕ} (B : α^[Idx k, Idx n]) (dC : α^[Idx m, Idx n]) : α^[Idx m, Idx k] :=
  let Bt := ArrayOps.transpose B
  ArrayOps.gemm dC Bt

/-- Backward pass for {lit}`C = A @ B` (matrix multiply) w.r.t. {lit}`B`.
    Given {lit}`dC`, returns {lit}`dB = Aᵀ @ dC`. -/
@[inline]
def gemm_backward_B {m k n : ℕ} (A : α^[Idx m, Idx k]) (dC : α^[Idx m, Idx n]) : α^[Idx k, Idx n] :=
  let At := ArrayOps.transpose A
  ArrayOps.gemm At dC

/-- Backward pass for elementwise {lit}`exp`.
    Given {lit}`y = exp(x)` and {lit}`dy`, returns {lit}`dx = y * dy`. -/
@[inline]
def exp_backward {ι : Type*} {n : ℕ} [IndexType ι n] (y dy : α^[ι]) : α^[ι] :=
  ArrayOps.mul y dy

/-- Backward pass for elementwise {lit}`log`.
    Given {lit}`x` and {lit}`dy`, returns {lit}`dx = dy / x`. -/
@[inline]
def log_backward {ι : Type*} {n : ℕ} [IndexType ι n] (x dy : α^[ι]) : α^[ι] :=
  ArrayOps.div dy x

/-- Backward pass for elementwise {lit}`tanh`.
    Given {lit}`y = tanh(x)` and {lit}`dy`, returns {lit}`dx = (1 - y²) * dy`. -/
@[inline]
def tanh_backward {ι : Type*} {n : ℕ} [IndexType ι n] (y dy : α^[ι]) : α^[ι] :=
  let y_sq := ArrayOps.mul y y
  let y_sq_dy := ArrayOps.mul y_sq dy
  ArrayOps.sub dy y_sq_dy

end ADHelpers

-- ============================================================================
-- Convenience Notations
-- ============================================================================

/-- Matrix multiply notation. -/
scoped infixl:70 " ⬝ " => gemm

/-- Matrix-vector multiply notation. -/
scoped infixl:70 " ⬝ᵥ " => gemv

end SciLean.Kernel
