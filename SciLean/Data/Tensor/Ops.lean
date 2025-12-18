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

`TensorOps` is a typeclass providing device-polymorphic tensor operations.
Different instances handle CPU (BLAS) and GPU (Metal) implementations transparently.

The operations are defined for Float tensors as this is the common case for ML.
-/

variable {ι : Type*} {n : ℕ} [IndexType ι n]
variable {κ : Type*} {m : ℕ} [IndexType κ m]

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

/-- ReLU activation: max(0, x) -/
def relu (a : CpuTensor Float ι) : CpuTensor Float ι :=
  ⟨⊞ i => if a.data[i] > 0 then a.data[i] else 0⟩

end CpuTensor

/-! ## GPU Operations (1D tensors) -/

namespace GpuTensor

/-- Element-wise addition on GPU -/
def add (a b : GpuTensor Float (Idx n)) : IO (GpuTensor Float (Idx n)) := do
  let result ← Metal.GpuBuffer.add a.data.buffer b.data.buffer n.toUSize
  return ⟨⟨result⟩⟩

/-- Element-wise multiplication on GPU -/
def mul (a b : GpuTensor Float (Idx n)) : IO (GpuTensor Float (Idx n)) := do
  let result ← Metal.GpuBuffer.mul a.data.buffer b.data.buffer n.toUSize
  return ⟨⟨result⟩⟩

/-- ReLU activation on GPU -/
def relu (a : GpuTensor Float (Idx n)) : IO (GpuTensor Float (Idx n)) := do
  let result ← Metal.GpuBuffer.relu a.data.buffer n.toUSize
  return ⟨⟨result⟩⟩

/-- Fused GEMM + Bias + ReLU: `C = max(0, A @ B + bias)`.
    A: (m, k), B: (k, n), bias: (n), returns C: (m, n). -/
def gemmBiasRelu (A : GpuTensor Float (Idx m × Idx k)) (B : GpuTensor Float (Idx k × Idx n))
    (bias : GpuTensor Float (Idx n)) : IO (GpuTensor Float (Idx m × Idx n)) := do
  let result ← Metal.GpuBuffer.gemmBiasRelu A.data.buffer B.data.buffer bias.data.buffer
    m.toUSize k.toUSize n.toUSize
  return ⟨⟨result⟩⟩

/-- Softmax activation on GPU (applied row-wise).
    For tensor of shape (m, n), softmax is applied to each row independently.
    Returns normalized probabilities where each row sums to 1. -/
def softmax (a : GpuTensor Float (Idx m × Idx n)) : IO (GpuTensor Float (Idx m × Idx n)) := do
  let result ← Metal.GpuBuffer.softmax a.data.buffer m.toUSize n.toUSize
  return ⟨⟨result⟩⟩

/-- GELU activation on GPU using fused bias+gelu with zero bias.
    GELU(x) ≈ 0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x³)))
    Note: For performance, prefer `biasGelu` when adding bias anyway. -/
def gelu (a : GpuTensor Float (Idx n)) : IO (GpuTensor Float (Idx n)) := do
  -- Use biasGelu with zero bias - the kernel handles this efficiently
  let zeroBias ← Metal.CpuBuffer.zeros n |>.upload
  let result ← Metal.GpuBuffer.biasGelu a.data.buffer zeroBias n.toUSize n.toUSize
  return ⟨⟨result⟩⟩

/-- Matrix multiply on GPU: `C = A @ B`.
    A: (m, k), B: (k, n), returns C: (m, n). -/
def gemm (A : GpuTensor Float (Idx m × Idx k)) (B : GpuTensor Float (Idx k × Idx n)) :
    IO (GpuTensor Float (Idx m × Idx n)) := do
  let result ← Metal.GpuBuffer.gemm A.data.buffer B.data.buffer m.toUSize k.toUSize n.toUSize
  return ⟨⟨result⟩⟩

/-- GEMM with A transposed: `C = A^T @ B`.
    A is stored as (k, m), computes A^T(m, k) @ B(k, n) = C(m, n).
    Used in backward pass: `grad_B = A^T @ grad_C`. -/
def gemmTN (A : GpuTensor Float (Idx k × Idx m)) (B : GpuTensor Float (Idx k × Idx n)) :
    IO (GpuTensor Float (Idx m × Idx n)) := do
  let result ← Metal.GpuBuffer.gemmTN A.data.buffer B.data.buffer m.toUSize k.toUSize n.toUSize
  return ⟨⟨result⟩⟩

/-- GEMM with B transposed: `C = A @ B^T`.
    A is (m, k), B is stored as (n, k), computes A @ B^T(k, n) = C(m, n).
    Used in backward pass: `grad_A = grad_C @ B^T`. -/
def gemmNT (A : GpuTensor Float (Idx m × Idx k)) (B : GpuTensor Float (Idx n × Idx k)) :
    IO (GpuTensor Float (Idx m × Idx n)) := do
  let result ← Metal.GpuBuffer.gemmNT A.data.buffer B.data.buffer m.toUSize k.toUSize n.toUSize
  return ⟨⟨result⟩⟩

/-! ### Element-wise Operations -/

/-- Element-wise subtraction on GPU -/
def sub (a b : GpuTensor Float (Idx n)) : IO (GpuTensor Float (Idx n)) := do
  let result ← Metal.GpuBuffer.sub a.data.buffer b.data.buffer n.toUSize
  return ⟨⟨result⟩⟩

/-- Scalar multiplication on GPU: `y = alpha * x` -/
def scale (alpha : Float) (a : GpuTensor Float (Idx n)) : IO (GpuTensor Float (Idx n)) := do
  let result ← Metal.GpuBuffer.scale n.toUSize alpha a.data.buffer
  return ⟨⟨result⟩⟩

/-- Negation on GPU: `y = -x` (implemented as scale by -1) -/
def neg (a : GpuTensor Float (Idx n)) : IO (GpuTensor Float (Idx n)) :=
  scale (-1.0) a

/-- AXPY on GPU: `y = alpha * x + y` -/
def axpy (alpha : Float) (x y : GpuTensor Float (Idx n)) : IO (GpuTensor Float (Idx n)) := do
  let result ← Metal.GpuBuffer.axpy n.toUSize alpha x.data.buffer y.data.buffer
  return ⟨⟨result⟩⟩

/-! ### Reduction Operations -/

/-- Sum all elements on GPU -/
def sum (a : GpuTensor Float (Idx n)) : IO Float :=
  Metal.GpuBuffer.sum a.data.buffer n.toUSize

/-- Column-wise sum on GPU: for matrix `(rows, cols)`, sum over rows for each column.
    Returns `(cols)` sums. Used for gradient accumulation over batch dimension. -/
def colSum (a : GpuTensor Float (Idx m × Idx n)) : IO (GpuTensor Float (Idx n)) := do
  let result ← Metal.GpuBuffer.colSum a.data.buffer m.toUSize n.toUSize
  return ⟨⟨result⟩⟩

/-! ### Normalization Operations -/

/-- Layer normalization on GPU: `y = gamma * (x - mean) / sqrt(var + eps) + beta`.
    Input `x` has shape `(batch, hiddenSize)`, `gamma` and `beta` have shape `(hiddenSize)`. -/
def layerNorm (x : GpuTensor Float (Idx m × Idx n))
    (gamma beta : GpuTensor Float (Idx n)) : IO (GpuTensor Float (Idx m × Idx n)) := do
  let totalElements := m * n
  let result ← Metal.GpuBuffer.layerNorm x.data.buffer gamma.data.buffer beta.data.buffer
      totalElements.toUSize n.toUSize
  return ⟨⟨result⟩⟩

/-- Batch normalization 2D (inference mode) on GPU.
    Input: `(batch, channels, height, width)` in NCHW format.
    `gamma`, `beta`, `mean`, `var` have shape `(channels)`. -/
def batchNorm2d {batch channels height width : ℕ}
    (x : GpuTensor Float (Idx batch × Idx channels × Idx height × Idx width))
    (gamma beta mean var : GpuTensor Float (Idx channels))
    (eps : Float) (applyRelu : Bool := false) :
    IO (GpuTensor Float (Idx batch × Idx channels × Idx height × Idx width)) := do
  let result ← Metal.GpuBuffer.batchNorm2d x.data.buffer gamma.data.buffer beta.data.buffer
      mean.data.buffer var.data.buffer
      batch.toUSize channels.toUSize height.toUSize width.toUSize
      eps (if applyRelu then 1 else 0)
  return ⟨⟨result⟩⟩

/-! ### Activation Operations -/

/-- Fused bias + ReLU on GPU: `y = max(0, x + bias)`.
    Input `x` has shape `(batch, features)`, `bias` has shape `(features)`. -/
def biasRelu (x : GpuTensor Float (Idx m × Idx n)) (bias : GpuTensor Float (Idx n)) :
    IO (GpuTensor Float (Idx m × Idx n)) := do
  let totalElements := m * n
  let result ← Metal.GpuBuffer.biasRelu x.data.buffer bias.data.buffer
      totalElements.toUSize n.toUSize
  return ⟨⟨result⟩⟩

/-- Fused bias + GELU on GPU: `y = GELU(x + bias)`.
    Input `x` has shape `(batch, features)`, `bias` has shape `(features)`. -/
def biasGelu (x : GpuTensor Float (Idx m × Idx n)) (bias : GpuTensor Float (Idx n)) :
    IO (GpuTensor Float (Idx m × Idx n)) := do
  let totalElements := m * n
  let result ← Metal.GpuBuffer.biasGelu x.data.buffer bias.data.buffer
      totalElements.toUSize n.toUSize
  return ⟨⟨result⟩⟩

/-- Bias add (no activation) on GPU: `y = x + bias`.
    Input `x` has shape `(batch, features)`, `bias` has shape `(features)`. -/
def biasAdd (x : GpuTensor Float (Idx m × Idx n)) (bias : GpuTensor Float (Idx n)) :
    IO (GpuTensor Float (Idx m × Idx n)) := do
  let totalElements := m * n
  let result ← Metal.GpuBuffer.biasAdd x.data.buffer bias.data.buffer
      totalElements.toUSize n.toUSize
  return ⟨⟨result⟩⟩

/-! ### Pooling Operations -/

/-- Max pooling 2D on GPU.
    Input: `(batch, channels, height, width)` in NCHW format.
    Returns: `(batch, channels, outHeight, outWidth)`. -/
def maxPool2d {batch channels inH inW poolH poolW strideH strideW : ℕ}
    (x : GpuTensor Float (Idx batch × Idx channels × Idx inH × Idx inW)) :
    IO (GpuTensor Float (Idx batch × Idx channels × Idx ((inH - poolH) / strideH + 1) × Idx ((inW - poolW) / strideW + 1))) := do
  let result ← Metal.GpuBuffer.maxPool2d x.data.buffer
      batch.toUSize channels.toUSize inH.toUSize inW.toUSize
      poolH.toUSize poolW.toUSize strideH.toUSize strideW.toUSize
  return ⟨⟨result⟩⟩

/-- Average pooling 2D on GPU.
    Input: `(batch, channels, height, width)` in NCHW format.
    Returns: `(batch, channels, outHeight, outWidth)`. -/
def avgPool2d {batch channels inH inW poolH poolW strideH strideW : ℕ}
    (x : GpuTensor Float (Idx batch × Idx channels × Idx inH × Idx inW)) :
    IO (GpuTensor Float (Idx batch × Idx channels × Idx ((inH - poolH) / strideH + 1) × Idx ((inW - poolW) / strideW + 1))) := do
  let result ← Metal.GpuBuffer.avgpool2d x.data.buffer
      batch.toUSize channels.toUSize inH.toUSize inW.toUSize
      poolH.toUSize poolW.toUSize strideH.toUSize strideW.toUSize
  return ⟨⟨result⟩⟩

/-! ### Convolution Operations -/

/-- Conv2D on GPU with optional fused ReLU.
    Input: `(batch, inChannels, height, width)` in NCHW format.
    Kernel: `(outChannels, inChannels, kernelH, kernelW)`.
    Bias: `(outChannels)`.
    Returns: `(batch, outChannels, outHeight, outWidth)`. -/
def conv2d {batch inCh outCh inH inW kH kW strideH strideW padH padW : ℕ}
    (x : GpuTensor Float (Idx batch × Idx inCh × Idx inH × Idx inW))
    (kernel : GpuTensor Float (Idx outCh × Idx inCh × Idx kH × Idx kW))
    (bias : GpuTensor Float (Idx outCh))
    (useRelu : Bool := false) :
    IO (GpuTensor Float (Idx batch × Idx outCh × Idx ((inH + 2*padH - kH) / strideH + 1) × Idx ((inW + 2*padW - kW) / strideW + 1))) := do
  let result ← Metal.GpuBuffer.conv2d x.data.buffer kernel.data.buffer bias.data.buffer
      batch.toUSize inCh.toUSize outCh.toUSize
      inH.toUSize inW.toUSize
      kH.toUSize kW.toUSize
      strideH.toUSize strideW.toUSize
      padH.toUSize padW.toUSize
      (if useRelu then 1 else 0)
  return ⟨⟨result⟩⟩

/-! ### Attention Operations -/

/-- Flash Attention on GPU: `output = softmax(Q @ K^T / sqrt(headDim)) @ V`.
    Q, K, V have shape `(seqLen, headDim)`.
    Returns `(seqLen, headDim)`. -/
def flashAttention (Q K V : GpuTensor Float (Idx seqLen × Idx headDim)) :
    IO (GpuTensor Float (Idx seqLen × Idx headDim)) := do
  let result ← Metal.GpuBuffer.flashAttention Q.data.buffer K.data.buffer V.data.buffer
      seqLen.toUSize headDim.toUSize
  return ⟨⟨result⟩⟩

/-- Flash Attention with causal mask on GPU.
    Position i can only attend to positions at most i (for autoregressive models).
    Q, K, V have shape `(seqLen, headDim)`.
    Returns `(seqLen, headDim)`. -/
def flashAttentionCausal (Q K V : GpuTensor Float (Idx seqLen × Idx headDim)) :
    IO (GpuTensor Float (Idx seqLen × Idx headDim)) := do
  let result ← Metal.GpuBuffer.flashAttentionCausal Q.data.buffer K.data.buffer V.data.buffer
      seqLen.toUSize headDim.toUSize
  return ⟨⟨result⟩⟩

/-! ### Buffer Operations -/

/-- Slice a GPU tensor: returns new tensor with elements from offset.
    This is a GPU-to-GPU copy (not a view). -/
def slice (a : GpuTensor Float (Idx n)) (offset count : ℕ) : IO (GpuTensor Float (Idx count)) := do
  let result ← Metal.GpuBuffer.slice a.data.buffer offset.toUSize count.toUSize
  return ⟨⟨result⟩⟩

end GpuTensor

/-! ## TensorOps Typeclass -/

/-- Type class for tensor operations that work across devices.
    For GPU operations, results are wrapped in IO. -/
class TensorOps (T : Type) where
  /-- Element-wise addition -/
  add : T → T → T
  /-- Element-wise multiplication (Hadamard product) -/
  mul : T → T → T
  /-- Scalar multiplication -/
  smul : Float → T → T

/-- CPU tensor operations (pure, no IO) -/
instance : TensorOps (CpuTensor Float ι) where
  add := CpuTensor.add
  mul := CpuTensor.mul
  smul := CpuTensor.smul

/-! ## TensorOpsIO for GPU -/

/-- Type class for tensor operations that may require IO (GPU) -/
class TensorOpsIO (T : Type) where
  /-- Element-wise addition -/
  add : T → T → IO T
  /-- Element-wise multiplication -/
  mul : T → T → IO T
  /-- ReLU activation -/
  relu : T → IO T

/-- GPU tensor operations (require IO) -/
instance : TensorOpsIO (GpuTensor Float (Idx n)) where
  add := GpuTensor.add
  mul := GpuTensor.mul
  relu := GpuTensor.relu

/-- Lift pure CPU ops to IO for uniform interface -/
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
