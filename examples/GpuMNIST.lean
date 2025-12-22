/-
GPU-Accelerated MNIST Training

A 2-layer MLP trained entirely on GPU using Metal:
  Input: 784 (28×28 flattened)
  Hidden: 128 with GELU
  Output: 10 with softmax

Uses SciLean's Metal backend for all operations:
- Forward: gemm → biasGelu → gemm → softmax
- Backward: softmaxBackward → gemmTN/gemmNT → geluBackward → gemmTN/gemmNT
- Update: axpy (SGD)

Type Safety: Uses {name}`CpuBuffer` and {name}`GpuTensor` to enforce explicit data transfers.
All GPU<->CPU transfers must use {lit}`.upload` or {lit}`.download` - no implicit coercions!
-/

import SciLean.FFI.Metal
import SciLean.FFI.Float32Array
import SciLean.Data.Tensor.Ops
import Batteries.Lean.Float

open SciLean
open SciLean.Metal

/-! ## CpuBuffer Helpers -/

namespace SciLean.Metal.CpuBuffer

/-- Read a Float32 from CpuBuffer at element index -/
def readFloat32 (buf : CpuBuffer) (elemIdx : Nat) : Float :=
  (buf.data.ugetFloat32 (elemIdx * 4).toUSize).toFloat

/-- Write a Float32 to CpuBuffer at element index -/
def writeFloat32 (buf : CpuBuffer) (elemIdx : Nat) (val : Float) : CpuBuffer :=
  ⟨buf.data.usetFloat32 (elemIdx * 4).toUSize val.toFloat32⟩

end SciLean.Metal.CpuBuffer

/-! ## GpuTensor Helpers -/

/-- Upload {name}`CpuBuffer` to GPU and wrap as a typed {name}`GpuTensor`. -/
def uploadTensor {ι : Type} {n : Nat} [IndexType ι n] [IndexTypeShape ι n]
    (buf : CpuBuffer) : IO (Float^[ι]@metal) := do
  let gpuBuf ← buf.upload
  return GpuTensor.fromContiguous (ι:=ι) gpuBuf

/-- Wrap a contiguous {name}`GpuBuffer` as a typed {name}`GpuTensor`. -/
def wrapTensor {ι : Type} {n : Nat} [IndexType ι n] [IndexTypeShape ι n]
    (buf : GpuBuffer) : Float^[ι]@metal :=
  GpuTensor.fromContiguous (ι:=ι) buf

/-- Slice a batch-major tensor along the first dimension (O(1) view, no copy). -/
def sliceBatch {total k : Nat}
    (buf : Float^[Idx total × Idx k]@metal) (start batchSize : Nat) :
    Float^[Idx batchSize × Idx k]@metal :=
  GpuTensor.sliceRows buf start batchSize

/-! ## Data Loading (returns CpuBuffer for type safety) -/

def checkFileExists (path : System.FilePath) : IO Unit := do
  if ¬(← path.pathExists) then
     throw (IO.userError s!"MNIST data file '{path}' not found. Please download from https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/")

/-- Load MNIST images to CPU buffer (explicit CPU-resident data) -/
def loadImagesCpu (path : System.FilePath) (maxImages : Nat) : IO CpuBuffer := do
  checkFileExists path
  if maxImages = 0 then return CpuBuffer.zeros 0

  let start ← IO.monoMsNow
  IO.print s!"Loading images from {path}... "

  let content ← IO.FS.readBinFile path
  -- Skip 16-byte header
  let data := content.extract 16 (16 + maxImages * 784)

  if data.size < maxImages * 784 then
    throw <| IO.userError s!"File {path} contains insufficient data"

  -- Convert to Float32 CpuBuffer (normalized to [0,1])
  let mut result := ByteArray.replicateFloat32 (maxImages * 784) 0.0
  for i in [0:maxImages * 784] do
    let byteVal := data.get! i
    let floatVal : Float32 := (byteVal.toNat.toFloat / 255.0).toFloat32
    result := result.usetFloat32 (i * 4).toUSize floatVal

  IO.println s!"done ({(← IO.monoMsNow) - start}ms)"
  return CpuBuffer.mk result

/-- Load MNIST labels to CPU buffer (explicit CPU-resident data) -/
def loadLabelsCpu (path : System.FilePath) (maxLabels : Nat) : IO CpuBuffer := do
  checkFileExists path
  if maxLabels = 0 then return CpuBuffer.zeros 0

  let start ← IO.monoMsNow
  IO.print s!"Loading labels from {path}... "

  let content ← IO.FS.readBinFile path
  -- Skip 8-byte header
  let data := content.extract 8 (8 + maxLabels)

  if data.size < maxLabels then
    throw <| IO.userError s!"File {path} contains insufficient labels"

  -- Convert to one-hot Float32 CpuBuffer
  let mut result := ByteArray.replicateFloat32 (maxLabels * 10) 0.0
  for i in [0:maxLabels] do
    let labelIdx := data.get! i
    for j in [0:10] do
      let val : Float32 := if j = labelIdx.toNat then 1.0 else 0.0
      result := result.usetFloat32 ((i * 10 + j) * 4).toUSize val

  IO.println s!"done ({(← IO.monoMsNow) - start}ms)"
  return CpuBuffer.mk result

/-! ## Weight Initialization (returns CpuBuffer) -/

/-- Initialize weights on CPU with random values -/
def initWeightsCpu (rows cols : Nat) (scale : Float) : IO CpuBuffer := do
  let size := rows * cols
  let mut result := ByteArray.replicateFloat32 size 0.0
  for i in [0:size] do
    let r := (← IO.rand 0 10000).toFloat / 10000.0
    let val : Float32 := (r * scale - scale / 2.0).toFloat32
    result := result.usetFloat32 (i * 4).toUSize val
  return CpuBuffer.mk result

/-- Initialize zero buffer on CPU -/
def initZerosCpu (size : Nat) : CpuBuffer :=
  CpuBuffer.zeros size

/-! ## GPU Weight Structure -/

structure GpuWeights where
  w1 : Float^[Idx 128 × Idx 784]@metal
  b1 : Float^[Idx 128]@metal
  w2 : Float^[Idx 10 × Idx 128]@metal
  b2 : Float^[Idx 10]@metal

structure GpuGradients where
  dw1 : Float^[Idx 128 × Idx 784]@metal
  db1 : Float^[Idx 128]@metal
  dw2 : Float^[Idx 10 × Idx 128]@metal
  db2 : Float^[Idx 10]@metal

/-! ## Forward Pass -/

/-- Forward pass (internal, no batching): {lit}`x → h = gelu(W1 @ x + b1) → o = W2 @ h + b2 → softmax(o)`.
    Returns softmax output plus hidden pre/post-activation for backward. -/
def forwardBatchInternal {batch : Nat} (weights : GpuWeights)
    (x : Float^[Idx batch × Idx 784]@metal) :
    IO (Float^[Idx batch × Idx 10]@metal ×
        Float^[Idx batch × Idx 128]@metal ×
        Float^[Idx batch × Idx 128]@metal ×
        Float^[Idx batch × Idx 10]@metal) := do
  -- First layer: h_pre = x @ W1^T, result is [batch, 128]
  let h_pre ← GpuTensor.gemmAuto x (GpuTensor.transpose weights.w1)

  -- Fused bias + gelu: h = gelu(h_pre + b1)
  let h ← GpuTensor.biasGelu h_pre weights.b1

  -- Second layer: o_pre = h @ W2^T, result is [batch, 10]
  let o_pre ← GpuTensor.gemmAuto h (GpuTensor.transpose weights.w2)

  -- Add bias to output
  let o ← GpuTensor.biasAdd o_pre weights.b2

  -- Softmax
  let y ← GpuTensor.softmax o

  return (y, h_pre, h, o)

/-- Forward pass with command buffer batching -/
def forwardBatch {batch : Nat} (weights : GpuWeights)
    (x : Float^[Idx batch × Idx 784]@metal) :
    IO (Float^[Idx batch × Idx 10]@metal ×
        Float^[Idx batch × Idx 128]@metal ×
        Float^[Idx batch × Idx 128]@metal ×
        Float^[Idx batch × Idx 10]@metal) :=
  withBatch (forwardBatchInternal weights x)

/-! ## Backward Pass -/

/-- Backward pass (internal, no batching): computes gradients for all weights. -/
def backwardBatchInternal {batch : Nat} (weights : GpuWeights)
    (x : Float^[Idx batch × Idx 784]@metal)
    (y target : Float^[Idx batch × Idx 10]@metal)
    (h_pre h : Float^[Idx batch × Idx 128]@metal) : IO GpuGradients := do
  -- dL/do = y - target (softmax + cross-entropy combined gradient)
  let d_o ← GpuTensor.sub y target

  -- dL/dW2 = d_o^T @ h
  -- Note: Using MPS for backward pass (AMX causes numerical instability - TODO: investigate)
  let dw2 ← GpuTensor.gemmTN d_o h

  -- dL/db2 = sum(d_o, axis=0)
  let db2 ← GpuTensor.colSum d_o

  -- dL/dh = d_o @ W2
  let d_h ← GpuTensor.gemm d_o weights.w2

  -- dL/dh_pre = d_h * gelu'(h_pre)
  let d_h_pre ← GpuTensor.geluBackward h_pre d_h

  -- dL/dW1 = d_h_pre^T @ x
  let dw1 ← GpuTensor.gemmTN d_h_pre x

  -- dL/db1 = sum(d_h_pre, axis=0)
  let db1 ← GpuTensor.colSum d_h_pre

  return { dw1 := dw1, db1 := db1, dw2 := dw2, db2 := db2 }

/-- Backward pass with command buffer batching -/
def backwardBatch {batch : Nat} (weights : GpuWeights)
    (x : Float^[Idx batch × Idx 784]@metal)
    (y target : Float^[Idx batch × Idx 10]@metal)
    (h_pre h : Float^[Idx batch × Idx 128]@metal) : IO GpuGradients :=
  withBatch (backwardBatchInternal weights x y target h_pre h)

/-! ## SGD Update -/

/-- SGD update (internal, no batching): {lit}`w := w - lr·grad`. -/
def sgdUpdateInternal (weights : GpuWeights) (grads : GpuGradients)
    (lr : Float) : IO GpuWeights := do
  -- w1 = w1 - lr * dw1
  let w1' ← GpuTensor.axpy (-lr) grads.dw1 weights.w1
  let b1' ← GpuTensor.axpy (-lr) grads.db1 weights.b1
  let w2' ← GpuTensor.axpy (-lr) grads.dw2 weights.w2
  let b2' ← GpuTensor.axpy (-lr) grads.db2 weights.b2
  return { w1 := w1', b1 := b1', w2 := w2', b2 := b2' }

/-- SGD update with command buffer batching -/
def sgdUpdate (weights : GpuWeights) (grads : GpuGradients)
    (lr : Float) : IO GpuWeights :=
  withBatch (sgdUpdateInternal weights grads lr)

/-! ## Debug Helpers -/

/-- Print buffer statistics for debugging (explicitly downloads from GPU) -/
def debugBuffer {ι : Type} {n : outParam ℕ} [IndexType ι n]
    (name : String) (gpuBuf : Float^[ι]@metal) (count : Nat) : IO Unit := do
  -- EXPLICIT download from GPU to CPU
  let cpuBuf ← gpuBuf.data.buffer.download
  let mut sum : Float := 0
  let mut minVal : Float := Float.inf
  let mut maxVal : Float := -Float.inf
  let mut hasNaN := false
  let mut hasInf := false
  let mut numZeros : Nat := 0
  for i in [0:count] do
    let val := cpuBuf.readFloat32 i
    if val.isNaN then hasNaN := true
    if val.isInf then hasInf := true
    if val == 0.0 then numZeros := numZeros + 1
    sum := sum + val
    if val < minVal then minVal := val
    if val > maxVal then maxVal := val
  let mean := sum / count.toFloat
  IO.println s!"  {name}: mean={mean}, min={minVal}, max={maxVal}, zeros={numZeros}/{count}, hasNaN={hasNaN}, hasInf={hasInf}"

/-- Diagnostic forward pass that prints intermediate values. -/
def forwardBatchDiag {batch : Nat} (weights : GpuWeights)
    (x : Float^[Idx batch × Idx 784]@metal) :
    IO (Float^[Idx batch × Idx 10]@metal ×
        Float^[Idx batch × Idx 128]@metal ×
        Float^[Idx batch × Idx 128]@metal ×
        Float^[Idx batch × Idx 10]@metal) := do
  IO.println s!"  --- Forward pass diagnostics (batch={batch}) ---"

  debugBuffer "input x" x (batch * 784)

  -- First layer: h_pre = x @ W1^T
  let h_pre ← GpuTensor.gemmAuto x (GpuTensor.transpose weights.w1)
  debugBuffer "h_pre (after gemmAuto)" h_pre (batch * 128)

  -- Fused bias + gelu
  let h ← GpuTensor.biasGelu h_pre weights.b1
  debugBuffer "h (after biasGelu)" h (batch * 128)

  -- Second layer: o_pre = h @ W2^T
  let o_pre ← GpuTensor.gemmAuto h (GpuTensor.transpose weights.w2)
  debugBuffer "o_pre (after gemmAuto)" o_pre (batch * 10)

  -- Add bias
  let o ← GpuTensor.biasAdd o_pre weights.b2
  debugBuffer "o (before softmax)" o (batch * 10)

  -- Softmax
  let y ← GpuTensor.softmax o
  debugBuffer "y (softmax out)" y (batch * 10)

  return (y, h_pre, h, o)

/-- Diagnostic backward pass that prints intermediate values. -/
def backwardBatchDiag {batch : Nat} (weights : GpuWeights)
    (x : Float^[Idx batch × Idx 784]@metal)
    (y target : Float^[Idx batch × Idx 10]@metal)
    (h_pre h : Float^[Idx batch × Idx 128]@metal) : IO GpuGradients := do
  IO.println s!"  --- Backward pass diagnostics (batch={batch}) ---"

  -- dL/do = y - target (softmax + cross-entropy combined gradient)
  let d_o ← GpuTensor.sub y target
  debugBuffer "d_o (y-target)" d_o (batch * 10)

  -- dL/dW2 = d_o^T @ h
  let dw2 ← GpuTensor.gemmTN d_o h
  debugBuffer "dw2" dw2 (10 * 128)

  -- dL/db2 = sum(d_o, axis=0)
  let db2 ← GpuTensor.colSum d_o
  debugBuffer "db2" db2 10

  -- dL/dh = d_o @ W2
  let d_h ← GpuTensor.gemm d_o weights.w2
  debugBuffer "d_h" d_h (batch * 128)

  -- dL/dh_pre = d_h * gelu'(h_pre)
  let d_h_pre ← GpuTensor.geluBackward h_pre d_h
  debugBuffer "d_h_pre" d_h_pre (batch * 128)

  -- dL/dW1 = d_h_pre^T @ x
  let dw1 ← GpuTensor.gemmTN d_h_pre x
  debugBuffer "dw1" dw1 (128 * 784)

  -- dL/db1 = sum(d_h_pre, axis=0)
  let db1 ← GpuTensor.colSum d_h_pre
  debugBuffer "db1" db1 128

  return { dw1 := dw1, db1 := db1, dw2 := dw2, db2 := db2 }

/-! ## Loss and Accuracy -/

/-- Compute cross-entropy loss (EXPLICIT download to CPU). -/
def computeLoss {batch : Nat}
    (gpuPred gpuTarget : Float^[Idx batch × Idx 10]@metal) : IO Float := do
  -- EXPLICIT download from GPU to CPU
  let pred ← gpuPred.data.buffer.download
  let target ← gpuTarget.data.buffer.download

  let eps : Float := 1e-7
  let n := batch

  let mut loss : Float := 0
  for i in [0:n] do
    for j in [0:10] do
      let idx := i * 10 + j
      let p := pred.readFloat32 idx
      let t := target.readFloat32 idx
      loss := loss - t * Float.log (p + eps)

  return loss / n.toFloat

/-- Compute accuracy (EXPLICIT download to CPU). -/
def computeAccuracy {batch : Nat}
    (gpuPred gpuTarget : Float^[Idx batch × Idx 10]@metal) : IO Float := do
  -- EXPLICIT download from GPU to CPU
  let pred ← gpuPred.data.buffer.download
  let target ← gpuTarget.data.buffer.download

  let n := batch

  let mut correct : Nat := 0
  for i in [0:n] do
    let mut predMax : Float := -1e10
    let mut targMax : Float := -1e10
    let mut predIdx : Nat := 0
    let mut targIdx : Nat := 0

    for j in [0:10] do
      let idx := i * 10 + j
      let p := pred.readFloat32 idx
      let t := target.readFloat32 idx

      if p > predMax then
        predMax := p
        predIdx := j
      if t > targMax then
        targMax := t
        targIdx := j

    if predIdx = targIdx then
      correct := correct + 1

  return correct.toFloat / n.toFloat

/-! ## Diagnostic Functions -/

/-- Run diagnostic test comparing different batch sizes -/
def diagBatchSizes (cpuImages cpuLabels : CpuBuffer)
    (cpuW1 cpuB1 cpuW2 cpuB2 : CpuBuffer) : IO Unit := do
  IO.println "\n=== Batch Size Diagnostic Test ==="

  let numSamples := cpuImages.sizeBytes / (784 * 4)
  let imagesFull : Float^[Idx numSamples × Idx 784]@metal ← uploadTensor cpuImages
  let labelsFull : Float^[Idx numSamples × Idx 10]@metal ← uploadTensor cpuLabels

  for testBatch in [1000, 2000, 3000, 4000] do
    IO.println s!"\n--- Testing batch size {testBatch} ---"

    -- Slice out the test batch
    let images := sliceBatch imagesFull 0 testBatch
    let labels := sliceBatch labelsFull 0 testBatch

    -- EXPLICIT upload from CPU to GPU
    let w1 : Float^[Idx 128 × Idx 784]@metal ← uploadTensor cpuW1
    let b1 : Float^[Idx 128]@metal ← uploadTensor cpuB1
    let w2 : Float^[Idx 10 × Idx 128]@metal ← uploadTensor cpuW2
    let b2 : Float^[Idx 10]@metal ← uploadTensor cpuB2
    let weights : GpuWeights := { w1, b1, w2, b2 }

    -- Forward pass with diagnostics
    let (pred, h_pre, h, _) ← forwardBatchDiag weights images

    -- Diagnostic backward pass
    let grads ← backwardBatchDiag weights images pred labels h_pre h

    -- Check weight update
    let lr := 0.0005
    let weights' ← sgdUpdate weights grads lr
    debugBuffer "w1 after update" weights'.w1 (128 * 784)

    -- Check accuracy before and after
    let (pred2, _, _, _) ← forwardBatch weights' images
    let acc ← computeAccuracy pred2 labels
    IO.println s!"  Accuracy after 1 step: {acc * 100}%"

/-! ## Training Loop -/

/-- Check if any value in a buffer is NaN (debugging helper) -/
def hasNaNInBuffer {ι : Type} {n : outParam ℕ} [IndexType ι n]
    (buf : Float^[ι]@metal) (count : Nat) : IO Bool := do
  let cpu ← buf.data.buffer.download
  for i in [0:count] do
    if (cpu.readFloat32 i).isNaN then return true
  return false

/-- Combined training step: forward + backward + update in a single command buffer.
    This eliminates per-operation dispatch overhead for maximum throughput.
    The learning rate should already be scaled for batch size.
    Note: No downloads inside withBatch - downloads commit the batch early! -/
def trainStep {batch : Nat} (weights : GpuWeights)
    (images : Float^[Idx batch × Idx 784]@metal)
    (labels : Float^[Idx batch × Idx 10]@metal)
    (lr : Float) : IO GpuWeights :=
  withBatch do
    -- Forward pass
    let (pred, h_pre, h, _) ← forwardBatchInternal weights images

    -- Backward pass
    let grads ← backwardBatchInternal weights images pred labels h_pre h

    -- SGD update
    sgdUpdateInternal weights grads lr

/-- Train one epoch with mini-batching: iterates through all samples in mini-batches.
    Uses GPU tensor slicing to extract each mini-batch. -/
def trainEpochMiniBatch {numSamples : Nat} (weights : GpuWeights)
    (images : Float^[Idx numSamples × Idx 784]@metal)
    (labels : Float^[Idx numSamples × Idx 10]@metal)
    (miniBatchSize : Nat) (lr : Float) : IO GpuWeights := do
  let mut w := weights
  let numBatches := numSamples / miniBatchSize

  for batchIdx in [0:numBatches] do
    -- Slice out this mini-batch (GPU-to-GPU copy)
    let start := batchIdx * miniBatchSize
    let batchImages := sliceBatch images start miniBatchSize
    let batchLabels := sliceBatch labels start miniBatchSize

    -- Training step on this mini-batch
    w ← trainStep w batchImages batchLabels lr

  return w

/-! ## Main -/

def main : IO Unit := do
  IO.println "GPU-Accelerated MNIST Training (Metal)"
  IO.println "========================================"
  IO.println "Using type-safe CpuBuffer/GpuTensor API (no implicit coercions!)"

  -- Check Metal availability
  if !isAvailable () then
    throw (IO.userError "Metal GPU not available")
  IO.println "Metal GPU: available"

  -- Configuration
  let numTrain := 60000  -- Full MNIST training set
  let miniBatchSize := 256  -- Mini-batch size for training
  let evalBatchSize := 1000  -- Batch size for evaluation
  let epochs := 10
  -- Learning rate: gradients are summed over mini-batch, so we scale lr inversely
  -- lr_effective = baseLr / miniBatchSize gives averaged gradients
  let baseLr := 0.5
  let lr := baseLr / miniBatchSize.toFloat

  -- Load training data to CPU (explicit CPU-resident data)
  let cpuImages ← loadImagesCpu "data/train-images-idx3-ubyte" numTrain
  let cpuLabels ← loadLabelsCpu "data/train-labels-idx1-ubyte" numTrain

  IO.println s!"Loaded {numTrain} training samples to CPU"

  -- Initialize weights on CPU (He initialization)
  let scale1 := Float.sqrt (2.0 / 784.0)
  let scale2 := Float.sqrt (2.0 / 128.0)

  IO.print "Initializing weights on CPU... "
  let cpuW1 ← initWeightsCpu 128 784 scale1
  let cpuB1 := initZerosCpu 128
  let cpuW2 ← initWeightsCpu 10 128 scale2
  let cpuB2 := initZerosCpu 10
  IO.println "done"

  -- EXPLICIT upload to GPU (type system enforces this!)
  IO.print "Uploading to GPU (explicit transfer)... "
  let images : Float^[Idx numTrain × Idx 784]@metal ← uploadTensor cpuImages
  let labels : Float^[Idx numTrain × Idx 10]@metal ← uploadTensor cpuLabels
  let w1 : Float^[Idx 128 × Idx 784]@metal ← uploadTensor cpuW1
  let b1 : Float^[Idx 128]@metal ← uploadTensor cpuB1
  let w2 : Float^[Idx 10 × Idx 128]@metal ← uploadTensor cpuW2
  let b2 : Float^[Idx 10]@metal ← uploadTensor cpuB2
  IO.println "done"

  -- DEBUG: Verify sizes immediately after upload
  IO.println s!"DEBUG: After upload - b1.sizeBytes = {b1.data.buffer.sizeBytes} (expected 512)"
  IO.println s!"DEBUG: After upload - w1.sizeBytes = {w1.data.buffer.sizeBytes} (expected {128 * 784 * 4})"
  IO.println s!"DEBUG: cpuB1.sizeBytes = {cpuB1.sizeBytes} (expected 512)"

  let initialWeights : GpuWeights := { w1 := w1, b1 := b1, w2 := w2, b2 := b2 }

  -- For evaluation, use first evalBatchSize samples
  let evalImages := sliceBatch images 0 evalBatchSize
  let evalLabels := sliceBatch labels 0 evalBatchSize

  -- Initial accuracy (downloads from GPU internally)
  let (initPred, _, _, _) ← forwardBatch initialWeights evalImages
  let initAcc ← computeAccuracy initPred evalLabels
  IO.println s!"Initial accuracy: {initAcc * 100}%"

  -- Training loop with mini-batching
  IO.println s!"\nTraining ({numTrain} samples, mini-batch={miniBatchSize})..."
  let mut weights := initialWeights

  for epoch in [0:epochs] do
    let start ← IO.monoMsNow

    -- Train one epoch: iterate through all mini-batches
    weights ← trainEpochMiniBatch weights images labels miniBatchSize lr

    let elapsed := (← IO.monoMsNow) - start

    -- NaN check: detect when corruption happens
    let w1Data ← weights.w1.data.buffer.download
    let mut hasNaN := false
    for i in [0:128*784] do
      if (w1Data.readFloat32 i).isNaN then hasNaN := true
    if hasNaN then
      IO.println s!"WARNING: NaN detected in w1 after epoch {epoch + 1}!"
      debugBuffer "w1" weights.w1 (128 * 784)
      debugBuffer "b1" weights.b1 128
      debugBuffer "w2" weights.w2 (10 * 128)
      debugBuffer "b2" weights.b2 10

    -- Compute accuracy on evaluation set
    let (pred, _, _, _) ← forwardBatch weights evalImages
    let acc ← computeAccuracy pred evalLabels

    let numBatches := numTrain / miniBatchSize
    IO.println s!"Epoch {epoch + 1}: accuracy = {acc * 100}%, time = {elapsed}ms ({numBatches} batches)"

  -- Final results
  let (finalPred, _, _, _) ← forwardBatch weights evalImages
  let finalAcc ← computeAccuracy finalPred evalLabels
  IO.println s!"\nFinal accuracy: {finalAcc * 100}%"

  IO.println "\nDone!"
