import SciLean.Monad.TensorM
import SciLean.Monad.GPU
import SciLean.Data.Tensor.GpuTensor
import SciLean.Data.IndexType.Shape

set_option doc.verso false

namespace SciLean

/-!
# TensorMGPU

GPU-backed specialization of {name}`TensorMT` with helpers that run inside
the {name}`GPU` monad and optionally escape to {name}`IO` via {name}`GPU.exec`.

This monad combines:
- Layout capability tracking (what the kernel accepts)
- Layout policy (prefer views vs copies)
- Layout statistics (how many copies were made)
- GPU batching via the {name}`GPU` monad
-/

abbrev TensorMGPU := TensorMT GPU

namespace TensorMGPU

def defaultCaps : LayoutCaps := TensorM.defaultCaps
def defaultPolicy : LayoutPolicy := TensorM.defaultPolicy
def defaultStats : LayoutStats := TensorM.defaultStats

@[inline]
def run (caps : LayoutCaps) (policy : LayoutPolicy) (stats : LayoutStats) (x : TensorMGPU α) :
    GPU (Except LayoutError (α × LayoutStats)) :=
  ExceptT.run <| (StateT.run (ReaderT.run (ReaderT.run x caps) policy) stats)

@[inline]
def eval (caps : LayoutCaps) (policy : LayoutPolicy) (stats : LayoutStats) (x : TensorMGPU α) :
    GPU (Except LayoutError α) := do
  let r ← run caps policy stats x
  return r.map (·.1)

@[inline]
def exec (caps : LayoutCaps) (policy : LayoutPolicy) (stats : LayoutStats) (x : TensorMGPU α) :
    GPU (Except LayoutError LayoutStats) := do
  let r ← run caps policy stats x
  return r.map (·.2)

@[inline]
def runDefault (x : TensorMGPU α) : GPU (Except LayoutError α) :=
  eval defaultCaps defaultPolicy defaultStats x

@[inline]
def runIO (caps : LayoutCaps) (policy : LayoutPolicy) (stats : LayoutStats) (x : TensorMGPU α) :
    IO (Except LayoutError (α × LayoutStats)) :=
  GPU.exec (run caps policy stats x)

@[inline]
def runDefaultIO (x : TensorMGPU α) : IO (Except LayoutError α) :=
  GPU.exec (runDefault x)

/-! ## Layout-Aware Tensor Operations -/

variable {ι : Type} {n : ℕ} [IndexType ι n] [IndexTypeShape ι n]
variable {κ : Type} {m : ℕ} [IndexType κ m] [IndexTypeShape κ m]

/-- Ensure a tensor is contiguous, copying if needed and policy allows. -/
def ensureContiguous (t : GpuTensor Float ι) : TensorMGPU (GpuTensor Float ι) := do
  if t.isContiguous then
    TensorMT.recordViewHit
    pure t
  else
    let caps ← TensorMT.getCaps
    let policy ← TensorMT.getPolicy
    if caps.acceptsStride then
      -- Kernel accepts non-contiguous, no copy needed
      TensorMT.recordViewHit
      pure t
    else if policy.allowCopy then
      -- Need to materialize a contiguous copy
      let elemBytes := 4 -- Float32
      TensorMT.recordCopy (n * elemBytes)
      liftM (GpuTensor.ensureContiguous t : IO (GpuTensor Float ι))
    else
      throw (.copyDisallowed "ensureContiguous")

/-- Element-wise addition with layout tracking. -/
def add (a b : GpuTensor Float ι) : TensorMGPU (GpuTensor Float ι) := do
  let a' ← ensureContiguous a
  let b' ← ensureContiguous b
  liftM (GpuTensor.add a' b' : IO (GpuTensor Float ι))

/-- Element-wise multiplication with layout tracking. -/
def mul (a b : GpuTensor Float ι) : TensorMGPU (GpuTensor Float ι) := do
  let a' ← ensureContiguous a
  let b' ← ensureContiguous b
  liftM (GpuTensor.mul a' b' : IO (GpuTensor Float ι))

/-- Element-wise subtraction with layout tracking. -/
def sub (a b : GpuTensor Float ι) : TensorMGPU (GpuTensor Float ι) := do
  let a' ← ensureContiguous a
  let b' ← ensureContiguous b
  liftM (GpuTensor.sub a' b' : IO (GpuTensor Float ι))

/-- Scalar multiplication with layout tracking. -/
def scale (s : Float) (a : GpuTensor Float ι) : TensorMGPU (GpuTensor Float ι) := do
  let a' ← ensureContiguous a
  liftM (GpuTensor.scale s a' : IO (GpuTensor Float ι))

/-- Negation with layout tracking. -/
def neg (a : GpuTensor Float ι) : TensorMGPU (GpuTensor Float ι) := do
  let a' ← ensureContiguous a
  liftM (GpuTensor.neg a' : IO (GpuTensor Float ι))

/-- ReLU with layout tracking. -/
def relu (a : GpuTensor Float ι) : TensorMGPU (GpuTensor Float ι) := do
  let a' ← ensureContiguous a
  liftM (GpuTensor.relu a' : IO (GpuTensor Float ι))

/-- GELU with layout tracking. -/
def gelu (a : GpuTensor Float ι) : TensorMGPU (GpuTensor Float ι) := do
  let a' ← ensureContiguous a
  liftM (GpuTensor.gelu a' : IO (GpuTensor Float ι))

/-- AXPY: result = alpha * x + y -/
def axpy (alpha : Float) (x y : GpuTensor Float ι) : TensorMGPU (GpuTensor Float ι) := do
  let x' ← ensureContiguous x
  let y' ← ensureContiguous y
  liftM (GpuTensor.axpy alpha x' y' : IO (GpuTensor Float ι))

/-- Softmax with layout tracking (matrix operation). -/
def softmax {m k : ℕ} (a : GpuTensor Float (Idx m × Idx k)) :
    TensorMGPU (GpuTensor Float (Idx m × Idx k)) := do
  let a' ← ensureContiguous a
  liftM (GpuTensor.softmax a' : IO (GpuTensor Float (Idx m × Idx k)))

/-- Column sum reduction. -/
def colSum {m k : ℕ} (a : GpuTensor Float (Idx m × Idx k)) :
    TensorMGPU (GpuTensor Float (Idx k)) := do
  let a' ← ensureContiguous a
  liftM (GpuTensor.colSum a' : IO (GpuTensor Float (Idx k)))

/-! ## Matrix Operations with Layout Awareness -/

/-- GEMM with transpose view support.
    Uses O(1) transpose views when caps.acceptsTransposed is true. -/
def gemm (A : GpuTensor Float (Idx m × Idx k)) (B : GpuTensor Float (Idx k × Idx n)) :
    TensorMGPU (GpuTensor Float (Idx m × Idx n)) := do
  let caps ← TensorMT.getCaps
  -- GEMM kernels typically accept transposed inputs natively
  if caps.acceptsTransposed then
    TensorMT.recordViewHit
    TensorMT.recordViewHit
    liftM (GpuTensor.gemm A B : IO (GpuTensor Float (Idx m × Idx n)))
  else
    -- Need contiguous inputs
    let A' ← ensureContiguous A
    let B' ← ensureContiguous B
    liftM (GpuTensor.gemm A' B' : IO (GpuTensor Float (Idx m × Idx n)))

/-- Fused GEMM + bias + ReLU. -/
def gemmBiasRelu (A : GpuTensor Float (Idx m × Idx k))
    (B : GpuTensor Float (Idx k × Idx n))
    (bias : GpuTensor Float (Idx n)) : TensorMGPU (GpuTensor Float (Idx m × Idx n)) := do
  let caps ← TensorMT.getCaps
  if caps.acceptsTransposed then
    TensorMT.recordViewHit
    TensorMT.recordViewHit
    TensorMT.recordViewHit
    liftM (GpuTensor.gemmBiasRelu A B bias : IO (GpuTensor Float (Idx m × Idx n)))
  else
    let A' ← ensureContiguous A
    let B' ← ensureContiguous B
    let bias' ← ensureContiguous bias
    liftM (GpuTensor.gemmBiasRelu A' B' bias' : IO (GpuTensor Float (Idx m × Idx n)))

/-! ## Transfer Operations -/

/-- Upload CPU tensor to GPU. -/
def upload (cpu : CpuTensor Float ι) : TensorMGPU (GpuTensor Float ι) :=
  liftM (do
    let bytes := cpu.data.data.byteData
    let gpuBuf ← Metal.GpuBuffer.fromByteArray bytes
    pure (GpuTensor.fromContiguous (ι:=ι) gpuBuf) : IO (GpuTensor Float ι))

/-- Download GPU tensor to CPU. -/
def download (gpu : GpuTensor Float ι) : TensorMGPU (CpuTensor Float ι) := do
  let gpu' ← ensureContiguous gpu
  liftM (do
    let bytes ← Metal.GpuBuffer.toByteArray gpu'.data.buffer
    pure ⟨⟨⟨bytes, sorry_proof⟩, sorry_proof⟩⟩ : IO (CpuTensor Float ι))

/-! ## High-Level Patterns -/

/-- Run a GPU computation on CPU data with automatic transfers and layout tracking. -/
def compute (input : CpuTensor Float ι)
    (f : GpuTensor Float ι → TensorMGPU (GpuTensor Float ι)) : TensorMGPU (CpuTensor Float ι) := do
  let gpuIn ← upload input
  let gpuOut ← f gpuIn
  download gpuOut

end TensorMGPU

end SciLean
