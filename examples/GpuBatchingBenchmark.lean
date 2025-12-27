/-
GPU Batching Benchmark

Compares performance of batched vs unbatched GPU operations.
Batching reduces CPU-GPU synchronization overhead by combining
multiple operations into a single command buffer submission.
-/
import SciLean.FFI.Metal
import SciLean.FFI.Float32Array
import SciLean.Util.Benchmark
import SciLean.Data.Tensor.Ops

open SciLean

def toGpu1d (data : ByteArray) (size : Nat) : IO (Float^[Idx size]@metal) := do
  let gpuBuf ← (Metal.CpuBuffer.mk data).upload
  return GpuTensor.fromContiguousBuffer (ι:=Idx size) gpuBuf #[size]

def logTimeIters (group mode : String) (size : Nat) (iters : Nat) (ms : Float) : IO Unit := do
  Benchmark.logMetric
    group
    "time_ms"
    ms
    (unit? := some "ms")
    (params := [
      Benchmark.paramStr "mode" mode,
      Benchmark.paramNat "size" size,
      Benchmark.paramNat "iters" iters
    ])

def logSpeedupIters (group : String) (size : Nat) (iters : Nat) (speedup : Float) : IO Unit := do
  Benchmark.logMetric
    group
    "speedup"
    speedup
    (unit? := some "x")
    (params := [
      Benchmark.paramNat "size" size,
      Benchmark.paramNat "iters" iters
    ])

def logTimeChain (group mode : String) (size : Nat) (chainLen : Nat) (ms : Float) : IO Unit := do
  Benchmark.logMetric
    group
    "time_ms"
    ms
    (unit? := some "ms")
    (params := [
      Benchmark.paramStr "mode" mode,
      Benchmark.paramNat "size" size,
      Benchmark.paramNat "chain_len" chainLen
    ])

def logSpeedupChain (group : String) (size : Nat) (chainLen : Nat) (speedup : Float) : IO Unit := do
  Benchmark.logMetric
    group
    "speedup"
    speedup
    (unit? := some "x")
    (params := [
      Benchmark.paramNat "size" size,
      Benchmark.paramNat "chain_len" chainLen
    ])

/-- Create a ByteArray filled with random floats -/
def randomFloats (n : Nat) (seed : Nat) : ByteArray := Id.run do
  let mut arr := ByteArray.replicateFloat32 n 0.0
  let mut s := seed
  for i in List.range n do
    s := (s * 1103515245 + 12345) % (2^31)
    let val := (Float.ofNat (s % 1000)) / 1000.0
    arr := arr.usetFloat32 (i * 4).toUSize val.toFloat32
  return arr

/-- Measure time for an IO action -/
def timeIO (action : IO Unit) : IO Float := do
  let start ← IO.monoMsNow
  action
  let stop ← IO.monoMsNow
  return (Float.ofNat (stop - start))

/-- Run N operations WITHOUT batching (each op syncs separately) -/
def runUnbatched (n : Nat) (size : Nat) : IO Float := do
  let data := randomFloats size 42
  let gpu ← toGpu1d data size

  timeIO do
    for _ in List.range n do
      let r1 ← GpuTensor.relu gpu
      let r2 ← GpuTensor.add r1 r1
      let _ ← GpuTensor.mul r2 gpu
      pure ()

/-- Run N operations WITH batching (single command buffer) -/
def runBatched (n : Nat) (size : Nat) : IO Float := do
  let data := randomFloats size 42
  let gpu ← toGpu1d data size

  timeIO do
    for _ in List.range n do
      let _ ← Metal.withBatch do
        let r1 ← GpuTensor.relu gpu
        let r2 ← GpuTensor.add r1 r1
        GpuTensor.mul r2 gpu
      pure ()

/-- Run a chain of operations without batching -/
def runChainUnbatched (chainLen : Nat) (size : Nat) : IO Float := do
  let data := randomFloats size 42
  let gpu ← toGpu1d data size

  timeIO do
    let mut current := gpu
    for _ in List.range chainLen do
      current ← GpuTensor.relu current
    pure ()

/-- Run a chain of operations with batching -/
def runChainBatched (chainLen : Nat) (size : Nat) : IO Float := do
  let data := randomFloats size 42
  let gpu ← toGpu1d data size

  timeIO do
    let _ ← Metal.withBatch do
      let mut current := gpu
      for _ in List.range chainLen do
        current ← GpuTensor.relu current
      return current
    pure ()

def main : IO Unit := do
  if !Metal.isAvailable () then
    IO.println "Metal GPU not available"
    return

  let quick := (← IO.getEnv "SCILEAN_BENCH_QUICK").isSome
  if quick then
    IO.println "Quick mode enabled (SCILEAN_BENCH_QUICK)"

  IO.println "=== GPU Batching Benchmark ==="
  IO.println ""

  -- Warmup
  IO.println "Warming up GPU..."
  let warmupIters := if quick then 3 else 10
  let warmupSize := if quick then 512 else 1024
  let _ ← runUnbatched warmupIters warmupSize
  let _ ← runBatched warmupIters warmupSize

  IO.println ""
  IO.println "--- Test 1: Multiple iterations (3 ops each) ---"
  IO.println "Each iteration does: relu → add → mul"
  IO.println ""

  let iterCases := if quick then [10, 20] else [10, 50, 100]
  let iterSize := if quick then 5000 else 10000
  for numIters in iterCases do
    let size := iterSize
    let unbatchedTime ← runUnbatched numIters size
    let batchedTime ← runBatched numIters size
    let speedup := unbatchedTime / (if batchedTime < 0.001 then 0.001 else batchedTime)

    IO.println s!"  {numIters} iterations, {size} elements:"
    IO.println s!"    Unbatched: {unbatchedTime} ms"
    IO.println s!"    Batched:   {batchedTime} ms"
    IO.println s!"    Speedup:   {speedup}x"
    IO.println ""
    logTimeIters "batching/iters" "unbatched" size numIters unbatchedTime
    logTimeIters "batching/iters" "batched" size numIters batchedTime
    logSpeedupIters "batching/iters" size numIters speedup

  IO.println "--- Test 2: Long chains (single iteration) ---"
  IO.println "Chain of N relu operations"
  IO.println ""

  let chainCases := if quick then [10, 50] else [10, 50, 100, 200]
  let chainSize := if quick then 20000 else 100000
  for chainLen in chainCases do
    let size := chainSize
    let unbatchedTime ← runChainUnbatched chainLen size
    let batchedTime ← runChainBatched chainLen size

    let speedup := unbatchedTime / (if batchedTime < 0.001 then 0.001 else batchedTime)

    IO.println s!"  Chain length {chainLen}, {size} elements:"
    IO.println s!"    Unbatched: {unbatchedTime} ms"
    IO.println s!"    Batched:   {batchedTime} ms"
    IO.println s!"    Speedup:   {speedup}x"
    IO.println ""
    logTimeChain "batching/chain" "unbatched" size chainLen unbatchedTime
    logTimeChain "batching/chain" "batched" size chainLen batchedTime
    logSpeedupChain "batching/chain" size chainLen speedup

  IO.println "=== Benchmark Complete ==="
  IO.println ""
  IO.println "Note: Batching reduces CPU-GPU synchronization overhead."
  IO.println "For small/fast operations, the overhead reduction is significant."
  IO.println "For large/slow operations, the GPU compute time dominates."
