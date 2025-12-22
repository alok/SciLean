/-
Copyright (c) 2024 SciLean contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: SciLean contributors
-/
import SciLean.Data.Tensor.Basic
import SciLean.Data.Tensor.Transfer

namespace SciLean

/-!
# Device-Polymorphic Tensor Operations

{lit}`TensorOps` is a typeclass providing device-polymorphic tensor operations.
Different instances handle CPU (BLAS) and GPU (Metal) implementations transparently.

The operations are defined for {name}`Float` tensors as this is the common case for ML.
-/

variable {ι : Type} {n : ℕ} [IndexType ι n]
variable {κ : Type} {m : ℕ} [IndexType κ m]

/-! ## CPU Operations -/

namespace CpuTensor

/-- Element-wise addition -/
def add (a b : CpuTensor Float ι) : CpuTensor Float ι :=
  ⟨⊞ i => a.data[i] + b.data[i]⟩

/-- Element-wise multiplication -/
def mul (a b : CpuTensor Float ι) : CpuTensor Float ι :=
  ⟨⊞ i => a.data[i] * b.data[i]⟩

/-- Element-wise negation -/
def neg (a : CpuTensor Float ι) : CpuTensor Float ι :=
  ⟨⊞ i => -a.data[i]⟩

/-- Scalar multiplication -/
def smul (s : Float) (a : CpuTensor Float ι) : CpuTensor Float ι :=
  ⟨⊞ i => s * a.data[i]⟩

/-- ReLU activation (clips negative values to 0). -/
def relu (a : CpuTensor Float ι) : CpuTensor Float ι :=
  ⟨⊞ i => if a.data[i] > 0 then a.data[i] else 0⟩

end CpuTensor

/-! ## GPU Operations (1D tensors) -/

namespace GpuTensor

/-- Element-wise addition on GPU -/
def add (a b : GpuTensor Float (Idx n)) : IO (GpuTensor Float (Idx n)) := do
  let a ← ensureContiguous a
  let b ← ensureContiguous b
  let result ← Metal.GpuBuffer.add a.data.buffer b.data.buffer n.toUSize
  return GpuTensor.fromContiguousBuffer (ι:=Idx n) result #[n]

/-- Element-wise multiplication on GPU -/
def mul (a b : GpuTensor Float (Idx n)) : IO (GpuTensor Float (Idx n)) := do
  let a ← ensureContiguous a
  let b ← ensureContiguous b
  let result ← Metal.GpuBuffer.mul a.data.buffer b.data.buffer n.toUSize
  return GpuTensor.fromContiguousBuffer (ι:=Idx n) result #[n]

/-- ReLU activation on GPU -/
def relu (a : GpuTensor Float (Idx n)) : IO (GpuTensor Float (Idx n)) := do
  let a ← ensureContiguous a
  let result ← Metal.GpuBuffer.relu a.data.buffer n.toUSize
  return GpuTensor.fromContiguousBuffer (ι:=Idx n) result #[n]

/-- Fused GEMM + bias + ReLU.
    Shapes: A {lean}`Idx m × Idx k`, B {lean}`Idx k × Idx n`, bias {lean}`Idx n`,
    returns {lean}`Idx m × Idx n`. -/
def gemmBiasRelu (A : GpuTensor Float (Idx m × Idx k)) (B : GpuTensor Float (Idx k × Idx n))
    (bias : GpuTensor Float (Idx n)) : IO (GpuTensor Float (Idx m × Idx n)) := do
  let A ← ensureContiguous A
  let B ← ensureContiguous B
  let bias ← ensureContiguous bias
  let result ← Metal.GpuBuffer.gemmBiasRelu A.data.buffer B.data.buffer bias.data.buffer
    m.toUSize k.toUSize n.toUSize
  return GpuTensor.fromContiguousBuffer (ι:=Idx m × Idx n) result #[m, n]

/-- Softmax activation on GPU (applied row-wise).
    For input shape {lean}`Idx m × Idx n`, softmax is applied to each row independently. -/
def softmax (a : GpuTensor Float (Idx m × Idx n)) : IO (GpuTensor Float (Idx m × Idx n)) := do
  let a ← ensureContiguous a
  let result ← Metal.GpuBuffer.softmax a.data.buffer m.toUSize n.toUSize
  return GpuTensor.fromContiguousBuffer (ι:=Idx m × Idx n) result #[m, n]

/-- GELU activation on GPU using fused bias+gelu with zero bias.
    For performance, prefer {lit}`biasGelu` when adding bias anyway. -/
def gelu (a : GpuTensor Float (Idx n)) : IO (GpuTensor Float (Idx n)) := do
  -- Use biasGelu with zero bias - the kernel handles this efficiently
  let a ← ensureContiguous a
  let zeroBias ← Metal.CpuBuffer.zeros n |>.upload
  let result ← Metal.GpuBuffer.biasGelu a.data.buffer zeroBias n.toUSize n.toUSize
  return GpuTensor.fromContiguousBuffer (ι:=Idx n) result #[n]

/-- GEMM with A transposed.
    A is stored as {lean}`Idx k × Idx m`, B as {lean}`Idx k × Idx n`,
    returns {lean}`Idx m × Idx n`. -/
def gemmTN (A : GpuTensor Float (Idx k × Idx m)) (B : GpuTensor Float (Idx k × Idx n)) :
    IO (GpuTensor Float (Idx m × Idx n)) := do
  let A_T := GpuTensor.transpose (m:=k) (k:=m) A
  GpuTensor.gemm A_T B

/-- GEMM with B transposed.
    A is stored as {lean}`Idx m × Idx k`, B as {lean}`Idx n × Idx k`,
    returns {lean}`Idx m × Idx n`. -/
def gemmNT (A : GpuTensor Float (Idx m × Idx k)) (B : GpuTensor Float (Idx n × Idx k)) :
    IO (GpuTensor Float (Idx m × Idx n)) := do
  let B_T := GpuTensor.transpose (m:=n) (k:=k) B
  GpuTensor.gemm A B_T

/-! ### Element-wise Operations -/

/-- Element-wise subtraction on GPU (any shape). -/
def sub {ι : Type} {n : ℕ} [IndexType ι n] [IndexTypeShape ι n]
    (a b : GpuTensor Float ι) : IO (GpuTensor Float ι) := do
  let a ← ensureContiguous a
  let b ← ensureContiguous b
  let total := IndexTypeShape.numel (ι:=ι)
  let result ← Metal.GpuBuffer.sub a.data.buffer b.data.buffer total.toUSize
  return GpuTensor.fromContiguousBuffer (ι:=ι) result (IndexTypeShape.shape (ι:=ι))

/-- Scalar multiplication on GPU. -/
def scale (alpha : Float) (a : GpuTensor Float (Idx n)) : IO (GpuTensor Float (Idx n)) := do
  let a ← ensureContiguous a
  let result ← Metal.GpuBuffer.scale n.toUSize alpha a.data.buffer
  return GpuTensor.fromContiguousBuffer (ι:=Idx n) result #[n]

/-- Negation on GPU (implemented as scale by -1). -/
def neg (a : GpuTensor Float (Idx n)) : IO (GpuTensor Float (Idx n)) :=
  scale (-1.0) a

/-- AXPY on GPU (scaled sum, any shape). -/
def axpy {ι : Type} {n : ℕ} [IndexType ι n] [IndexTypeShape ι n]
    (alpha : Float) (x y : GpuTensor Float ι) : IO (GpuTensor Float ι) := do
  let x ← ensureContiguous x
  let y ← ensureContiguous y
  let total := IndexTypeShape.numel (ι:=ι)
  let result ← Metal.GpuBuffer.axpy total.toUSize alpha x.data.buffer y.data.buffer
  return GpuTensor.fromContiguousBuffer (ι:=ι) result (IndexTypeShape.shape (ι:=ι))

/-! ### Reduction Operations -/

/-- Sum all elements on GPU -/
def sum (a : GpuTensor Float (Idx n)) : IO Float :=
  do
    let a ← ensureContiguous a
    Metal.GpuBuffer.sum a.data.buffer n.toUSize

/-- Column-wise sum on GPU.
    For input shape {lean}`Idx m × Idx n`, sums over rows and returns {lean}`Idx n`.
    Used for gradient accumulation over a batch dimension. -/
def colSum (a : GpuTensor Float (Idx m × Idx n)) : IO (GpuTensor Float (Idx n)) := do
  let a ← ensureContiguous a
  let result ← Metal.GpuBuffer.colSum a.data.buffer m.toUSize n.toUSize
  return GpuTensor.fromContiguousBuffer (ι:=Idx n) result #[n]

/-! ### Normalization Operations -/

/-- Layer normalization on GPU.
    Input {lean}`x` has shape {lean}`Idx m × Idx n`, and {lean}`gamma`/ {lean}`beta` have shape {lean}`Idx n`. -/
def layerNorm (x : GpuTensor Float (Idx m × Idx n))
    (gamma beta : GpuTensor Float (Idx n)) : IO (GpuTensor Float (Idx m × Idx n)) := do
  let x ← ensureContiguous x
  let gamma ← ensureContiguous gamma
  let beta ← ensureContiguous beta
  let totalElements := m * n
  let result ← Metal.GpuBuffer.layerNorm x.data.buffer gamma.data.buffer beta.data.buffer
      totalElements.toUSize n.toUSize
  return GpuTensor.fromContiguousBuffer (ι:=Idx m × Idx n) result #[m, n]

/-- Batch normalization 2D (inference mode) on GPU.
    Input shape {lean}`Idx batch × Idx channels × Idx height × Idx width` in NCHW format.
    {lean}`gamma`, {lean}`beta`, {lean}`mean`, {lean}`var` have shape {lean}`Idx channels`. -/
def batchNorm2d {batch channels height width : ℕ}
    (x : GpuTensor Float (Idx batch × Idx channels × Idx height × Idx width))
    (gamma beta mean var : GpuTensor Float (Idx channels))
    (eps : Float) (applyRelu : Bool := false) :
    IO (GpuTensor Float (Idx batch × Idx channels × Idx height × Idx width)) := do
  let x ← ensureContiguous x
  let gamma ← ensureContiguous gamma
  let beta ← ensureContiguous beta
  let mean ← ensureContiguous mean
  let var ← ensureContiguous var
  let result ← Metal.GpuBuffer.batchNorm2d x.data.buffer gamma.data.buffer beta.data.buffer
      mean.data.buffer var.data.buffer
      batch.toUSize channels.toUSize height.toUSize width.toUSize
      eps (if applyRelu then 1 else 0)
  return GpuTensor.fromContiguousBuffer (ι:=Idx batch × Idx channels × Idx height × Idx width)
    result #[batch, channels, height, width]

/-! ### Activation Operations -/

/-- Fused bias + ReLU on GPU.
    Input {lean}`x` has shape {lean}`Idx m × Idx n`, bias has shape {lean}`Idx n`. -/
def biasRelu (x : GpuTensor Float (Idx m × Idx n)) (bias : GpuTensor Float (Idx n)) :
    IO (GpuTensor Float (Idx m × Idx n)) := do
  let x ← ensureContiguous x
  let bias ← ensureContiguous bias
  let totalElements := m * n
  let result ← Metal.GpuBuffer.biasRelu x.data.buffer bias.data.buffer
      totalElements.toUSize n.toUSize
  return GpuTensor.fromContiguousBuffer (ι:=Idx m × Idx n) result #[m, n]

/-- Fused bias + GELU on GPU.
    Input {lean}`x` has shape {lean}`Idx m × Idx n`, bias has shape {lean}`Idx n`. -/
def biasGelu (x : GpuTensor Float (Idx m × Idx n)) (bias : GpuTensor Float (Idx n)) :
    IO (GpuTensor Float (Idx m × Idx n)) := do
  let x ← ensureContiguous x
  let bias ← ensureContiguous bias
  let totalElements := m * n
  let result ← Metal.GpuBuffer.biasGelu x.data.buffer bias.data.buffer
      totalElements.toUSize n.toUSize
  return GpuTensor.fromContiguousBuffer (ι:=Idx m × Idx n) result #[m, n]

/-- GELU backward on GPU (element-wise, any shape). -/
def geluBackward {ι : Type} {n : ℕ} [IndexType ι n] [IndexTypeShape ι n]
    (input gradOutput : GpuTensor Float ι) : IO (GpuTensor Float ι) := do
  let input ← ensureContiguous input
  let gradOutput ← ensureContiguous gradOutput
  let total := IndexTypeShape.numel (ι:=ι)
  let result ← Metal.GpuBuffer.geluBackward input.data.buffer gradOutput.data.buffer total.toUSize
  return GpuTensor.fromContiguousBuffer (ι:=ι) result (IndexTypeShape.shape (ι:=ι))

/-- Bias add (no activation) on GPU.
    Input {lean}`x` has shape {lean}`Idx m × Idx n`, bias has shape {lean}`Idx n`. -/
def biasAdd (x : GpuTensor Float (Idx m × Idx n)) (bias : GpuTensor Float (Idx n)) :
    IO (GpuTensor Float (Idx m × Idx n)) := do
  let x ← ensureContiguous x
  let bias ← ensureContiguous bias
  let totalElements := m * n
  let result ← Metal.GpuBuffer.biasAdd x.data.buffer bias.data.buffer
      totalElements.toUSize n.toUSize
  return GpuTensor.fromContiguousBuffer (ι:=Idx m × Idx n) result #[m, n]

/-! ### Pooling Operations -/

/-- Max pooling 2D on GPU.
    Input shape {lean}`Idx batch × Idx channels × Idx inH × Idx inW` in NCHW format.
    Returns {lean}`Idx batch × Idx channels × Idx ((inH - poolH) / strideH + 1) × Idx ((inW - poolW) / strideW + 1)`. -/
def maxPool2d {batch channels inH inW poolH poolW strideH strideW : ℕ}
    (x : GpuTensor Float (Idx batch × Idx channels × Idx inH × Idx inW)) :
    IO (GpuTensor Float (Idx batch × Idx channels × Idx ((inH - poolH) / strideH + 1) × Idx ((inW - poolW) / strideW + 1))) := do
  let x ← ensureContiguous x
  let result ← Metal.GpuBuffer.maxPool2d x.data.buffer
      batch.toUSize channels.toUSize inH.toUSize inW.toUSize
      poolH.toUSize poolW.toUSize strideH.toUSize strideW.toUSize
  return GpuTensor.fromContiguousBuffer
    (ι:=Idx batch × Idx channels × Idx ((inH - poolH) / strideH + 1) × Idx ((inW - poolW) / strideW + 1))
    result #[batch, channels, (inH - poolH) / strideH + 1, (inW - poolW) / strideW + 1]

/-- Average pooling 2D on GPU.
    Input shape {lean}`Idx batch × Idx channels × Idx inH × Idx inW` in NCHW format.
    Returns {lean}`Idx batch × Idx channels × Idx ((inH - poolH) / strideH + 1) × Idx ((inW - poolW) / strideW + 1)`. -/
def avgPool2d {batch channels inH inW poolH poolW strideH strideW : ℕ}
    (x : GpuTensor Float (Idx batch × Idx channels × Idx inH × Idx inW)) :
    IO (GpuTensor Float (Idx batch × Idx channels × Idx ((inH - poolH) / strideH + 1) × Idx ((inW - poolW) / strideW + 1))) := do
  let x ← ensureContiguous x
  let result ← Metal.GpuBuffer.avgpool2d x.data.buffer
      batch.toUSize channels.toUSize inH.toUSize inW.toUSize
      poolH.toUSize poolW.toUSize strideH.toUSize strideW.toUSize
  return GpuTensor.fromContiguousBuffer
    (ι:=Idx batch × Idx channels × Idx ((inH - poolH) / strideH + 1) × Idx ((inW - poolW) / strideW + 1))
    result #[batch, channels, (inH - poolH) / strideH + 1, (inW - poolW) / strideW + 1]

/-! ### Convolution Operations -/

/-- Conv2D on GPU with optional fused ReLU.
    Input shape {lean}`Idx batch × Idx inCh × Idx inH × Idx inW` in NCHW format.
    Kernel shape {lean}`Idx outCh × Idx inCh × Idx kH × Idx kW`, bias shape {lean}`Idx outCh`.
    Returns {lean}`Idx batch × Idx outCh × Idx ((inH + 2*padH - kH) / strideH + 1) × Idx ((inW + 2*padW - kW) / strideW + 1)`. -/
def conv2d {batch inCh outCh inH inW kH kW strideH strideW padH padW : ℕ}
    (x : GpuTensor Float (Idx batch × Idx inCh × Idx inH × Idx inW))
    (kernel : GpuTensor Float (Idx outCh × Idx inCh × Idx kH × Idx kW))
    (bias : GpuTensor Float (Idx outCh))
    (useRelu : Bool := false) :
    IO (GpuTensor Float (Idx batch × Idx outCh × Idx ((inH + 2*padH - kH) / strideH + 1) × Idx ((inW + 2*padW - kW) / strideW + 1))) := do
  let x ← ensureContiguous x
  let kernel ← ensureContiguous kernel
  let bias ← ensureContiguous bias
  let result ← Metal.GpuBuffer.conv2d x.data.buffer kernel.data.buffer bias.data.buffer
      batch.toUSize inCh.toUSize outCh.toUSize
      inH.toUSize inW.toUSize
      kH.toUSize kW.toUSize
      strideH.toUSize strideW.toUSize
      padH.toUSize padW.toUSize
      (if useRelu then 1 else 0)
  return GpuTensor.fromContiguousBuffer
    (ι:=Idx batch × Idx outCh × Idx ((inH + 2*padH - kH) / strideH + 1) × Idx ((inW + 2*padW - kW) / strideW + 1))
    result #[batch, outCh, (inH + 2*padH - kH) / strideH + 1, (inW + 2*padW - kW) / strideW + 1]

/-! ### Attention Operations -/

/-! ### Buffer Operations -/

/-- Slice a GPU tensor (GPU-to-GPU copy, not a view). -/
def slice (a : GpuTensor Float (Idx n)) (offset count : ℕ) : IO (GpuTensor Float (Idx count)) := do
  let a ← ensureContiguous a
  let result ← Metal.GpuBuffer.slice a.data.buffer offset.toUSize count.toUSize
  return GpuTensor.fromContiguousBuffer (ι:=Idx count) result #[count]

end GpuTensor

/-! ## TensorOps Typeclass -/

/-- Type class for tensor operations that work across devices.
    For GPU operations, results are wrapped in {name}`IO`. -/
class TensorOps (T : Type) where
  /-- Element-wise addition -/
  add : T → T → T
  /-- Element-wise multiplication (Hadamard product) -/
  mul : T → T → T
  /-- Scalar multiplication -/
  smul : Float → T → T

/-- CPU tensor operations (pure, no {name}`IO`). -/
instance : TensorOps (CpuTensor Float ι) where
  add := CpuTensor.add
  mul := CpuTensor.mul
  smul := CpuTensor.smul

/-! ## TensorOpsIO for GPU -/

/-- Type class for tensor operations that may require {name}`IO` (GPU). -/
class TensorOpsIO (T : Type) where
  /-- Element-wise addition -/
  add : T → T → IO T
  /-- Element-wise multiplication -/
  mul : T → T → IO T
  /-- ReLU activation -/
  relu : T → IO T

/-- GPU tensor operations (require {name}`IO`). -/
instance : TensorOpsIO (GpuTensor Float (Idx n)) where
  add := GpuTensor.add
  mul := GpuTensor.mul
  relu := GpuTensor.relu

/-- Lift pure CPU ops to {name}`IO` for a uniform interface. -/
instance [TensorOps T] : TensorOpsIO T where
  add a b := pure (TensorOps.add a b)
  mul a b := pure (TensorOps.mul a b)
  relu _ := panic! "relu not implemented for this tensor type"

/-! ## Algebra Instances for CPU Tensors -/

instance : Add (CpuTensor Float ι) where
  add := CpuTensor.add

instance : Neg (CpuTensor Float ι) where
  neg := CpuTensor.neg

instance : Sub (CpuTensor Float ι) where
  sub a b := CpuTensor.add a (CpuTensor.neg b)

instance : HMul Float (CpuTensor Float ι) (CpuTensor Float ι) where
  hMul := CpuTensor.smul

end SciLean
