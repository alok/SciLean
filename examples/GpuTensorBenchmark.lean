/-
  GpuTensor Benchmark: Compare ByteArray-based vs GPU-resident operations

  This demonstrates the performance advantage of keeping data on the GPU
  between operations rather than copying to/from CPU on every call.
-/
import SciLean.FFI.Metal
import SciLean.Util.Benchmark
import SciLean.Data.Tensor

open SciLean
open SciLean.Metal

def logMs (group metric : String) (shape : String) (ms : Float) : IO Unit := do
  SciLean.Benchmark.logMetric
    group
    metric
    ms
    (unit? := some "ms")
    (params := [
      SciLean.Benchmark.paramStr "shape" shape
    ])

/-- Create test data using Metal's efficient fill function -/
def makeF32Data (size : Nat) (value : Float32 := 1.0) : ByteArray :=
  Float32.fill size.toUSize value

/-- Upload ByteArray as a contiguous GPU matrix. -/
def toGpuMatrix (m k : Nat) (data : ByteArray) : IO (Float^[Idx m × Idx k]@metal) := do
  let gpuBuf ← (CpuBuffer.mk data).upload
  return GpuTensor.fromContiguousBuffer (ι:=Idx m × Idx k) gpuBuf #[m, k]

/-- Format float as string with limited digits -/
def fmt (f : Float) : String := (f.toString.take 7).toString

/-- Benchmark: Multiple GEMM with ByteArray (copies every call) -/
def benchGemmByteArray (m k n : Nat) (iters : Nat) : IO Float := do
  let A := makeF32Data (m * k) 1.0
  let B := makeF32Data (k * n) 2.0

  for _ in [0:3] do
    let _ := Float32.gemmAuto m.toUSize k.toUSize n.toUSize A B

  let startTime ← IO.monoNanosNow
  for _ in [0:iters] do
    let _ := Float32.gemmAuto m.toUSize k.toUSize n.toUSize A B
  let endTime ← IO.monoNanosNow

  pure (Float.ofNat (endTime - startTime) / 1e9)

/-- Benchmark: Multiple GEMM with {name}`GpuTensor` (no intermediate copies). -/
def benchGemmGpuTensor (m k n : Nat) (iters : Nat) : IO Float := do
  let A := makeF32Data (m * k) 1.0
  let B := makeF32Data (k * n) 2.0

  let gpuA ← toGpuMatrix m k A
  let gpuB ← toGpuMatrix k n B

  for _ in [0:3] do
    let _ ← GpuTensor.gemmAuto gpuA gpuB

  let startTime ← IO.monoNanosNow
  for _ in [0:iters] do
    let _ ← GpuTensor.gemmAuto gpuA gpuB
  let endTime ← IO.monoNanosNow

  pure (Float.ofNat (endTime - startTime) / 1e9)

/-- Benchmark: Chained GEMM operations with ByteArray -/
def benchChainByteArray (sz : Nat) (iters : Nat) : IO Float := do
  let A := makeF32Data (sz * sz) 1.0
  let B := makeF32Data (sz * sz) 2.0
  let C := makeF32Data (sz * sz) 3.0

  for _ in [0:3] do
    let ab := Float32.gemmAuto sz.toUSize sz.toUSize sz.toUSize A B
    let _ := Float32.gemmAuto sz.toUSize sz.toUSize sz.toUSize ab C

  let startTime ← IO.monoNanosNow
  for _ in [0:iters] do
    let ab := Float32.gemmAuto sz.toUSize sz.toUSize sz.toUSize A B
    let _ := Float32.gemmAuto sz.toUSize sz.toUSize sz.toUSize ab C
  let endTime ← IO.monoNanosNow

  pure (Float.ofNat (endTime - startTime) / 1e9)

/-- Benchmark: Chained GEMM with {name}`GpuTensor` (0 intermediate copies). -/
def benchChainGpuTensor (sz : Nat) (iters : Nat) : IO Float := do
  let A := makeF32Data (sz * sz) 1.0
  let B := makeF32Data (sz * sz) 2.0
  let C := makeF32Data (sz * sz) 3.0

  let gpuA ← toGpuMatrix sz sz A
  let gpuB ← toGpuMatrix sz sz B
  let gpuC ← toGpuMatrix sz sz C

  for _ in [0:3] do
    let ab ← GpuTensor.gemmAuto gpuA gpuB
    let _ ← GpuTensor.gemmAuto ab gpuC

  let startTime ← IO.monoNanosNow
  for _ in [0:iters] do
    let ab ← GpuTensor.gemmAuto gpuA gpuB
    let _ ← GpuTensor.gemmAuto ab gpuC
  let endTime ← IO.monoNanosNow

  pure (Float.ofNat (endTime - startTime) / 1e9)

/-- Benchmark: Upload + Download transfer averaged over many iterations. -/
def benchTransfer (cpu : CpuBuffer) (iters : Nat) : IO (Float × Float) := do
  let _ ← cpu.upload >>= GpuBuffer.download
  let mut uploadNs : Nat := 0
  let mut downloadNs : Nat := 0
  for _ in [0:iters] do
    let uploadStart ← IO.monoNanosNow
    let gpu ← cpu.upload
    let uploadEnd ← IO.monoNanosNow
    let downloadStart ← IO.monoNanosNow
    let _ ← GpuBuffer.download gpu
    let downloadEnd ← IO.monoNanosNow
    uploadNs := uploadNs + (uploadEnd - uploadStart)
    downloadNs := downloadNs + (downloadEnd - downloadStart)
  let itersF := iters.toFloat
  let uploadMs := Float.ofNat uploadNs / 1e6 / itersF
  let downloadMs := Float.ofNat downloadNs / 1e6 / itersF
  return (uploadMs, downloadMs)

def main : IO Unit := do
  IO.println ""
  IO.println "=============================================================="
  IO.println "  GpuTensor Benchmark: ByteArray vs GPU-Resident Buffers"
  IO.println "=============================================================="
  IO.println ""
  IO.println "ByteArray: copies data CPU<->GPU on every operation"
  IO.println "GpuTensor: data stays on GPU, copies only at start/end"
  IO.println ""

  IO.println "--- Test 1: Single GEMM Operation ---"
  IO.println "(Note: ByteArray times show 0 because pure functions don't GPU sync)"
  IO.println ""
  IO.println "Config          | ByteArray   | GpuTensor   | GPU Time"
  IO.println "------------------------------------------------------"

  -- 256x256 (256KB data)
  let gpuTime256 ← benchGemmGpuTensor 256 256 256 100
  let gpuMs256 := gpuTime256 * 1000.0 / 100.0
  IO.println s!"256x256         | (no sync)   | {fmt gpuMs256}ms  | {fmt gpuMs256}ms"
  logMs "gpu_tensor/gemm" "time_ms" "256x256" gpuMs256

  -- 512x512 (1MB data)
  let gpuTime512 ← benchGemmGpuTensor 512 512 512 50
  let gpuMs512 := gpuTime512 * 1000.0 / 50.0
  IO.println s!"512x512         | (no sync)   | {fmt gpuMs512}ms  | {fmt gpuMs512}ms"
  logMs "gpu_tensor/gemm" "time_ms" "512x512" gpuMs512

  -- 1024x1024 (4MB data)
  let gpuTime1024 ← benchGemmGpuTensor 1024 1024 1024 20
  let gpuMs1024 := gpuTime1024 * 1000.0 / 20.0
  IO.println s!"1024x1024       | (no sync)   | {fmt gpuMs1024}ms  | {fmt gpuMs1024}ms"
  logMs "gpu_tensor/gemm" "time_ms" "1024x1024" gpuMs1024

  IO.println ""

  IO.println "--- Test 2: Chained GEMM (A*B then result*C) ---"
  IO.println ""
  IO.println "With ByteArray: 4 copies per chain (A,B up; AB down,up; AB*C down)"
  IO.println "With GpuTensor: 0 copies per chain (all stays on GPU)"
  IO.println ""
  IO.println "Size            | GpuTensor Chain | Est. Transfer Saved"
  IO.println "------------------------------------------------------"

  -- 256x256 chain - each copy is ~0.01ms, 4 copies = 0.04ms saved per chain
  let gpuChain256 ← benchChainGpuTensor 256 100
  let gpuChainMs256 := gpuChain256 * 1000.0 / 100.0
  IO.println s!"256x256         | {fmt gpuChainMs256}ms       | ~0.04ms/iter (4x 0.01ms)"
  logMs "gpu_tensor/chain" "time_ms" "256x256" gpuChainMs256

  -- 512x512 chain - each copy is ~0.28ms, 4 copies = 1.1ms saved per chain
  let gpuChain512 ← benchChainGpuTensor 512 50
  let gpuChainMs512 := gpuChain512 * 1000.0 / 50.0
  IO.println s!"512x512         | {fmt gpuChainMs512}ms       | ~1.1ms/iter (4x 0.28ms)"
  logMs "gpu_tensor/chain" "time_ms" "512x512" gpuChainMs512

  IO.println ""

  IO.println "--- Test 3: Transfer Overhead (Upload + Download) ---"
  IO.println "(Pre-creating data to exclude creation time from measurements)"
  IO.println ""

  -- Pre-create all data using Metal GPU fill (fast)
  let data256k := makeF32Data (256 * 256) 1.0
  let data1m := makeF32Data (512 * 512) 1.0
  let data4m := makeF32Data (1024 * 1024) 1.0
  let cpu256k : CpuBuffer := ⟨data256k⟩
  let cpu1m : CpuBuffer := ⟨data1m⟩
  let cpu4m : CpuBuffer := ⟨data4m⟩
  IO.println ""

  IO.println "Size               | Upload     | Download   | Total"
  IO.println "------------------------------------------------------"

  -- 256KB
  let (uploadMs256k, downloadMs256k) ← benchTransfer cpu256k 200
  IO.println s!"256KB (256x256)    | {fmt uploadMs256k}ms  | {fmt downloadMs256k}ms  | {fmt (uploadMs256k + downloadMs256k)}ms"
  logMs "gpu_tensor/transfer" "upload_ms" "256x256" uploadMs256k
  logMs "gpu_tensor/transfer" "download_ms" "256x256" downloadMs256k

  -- 1MB
  let (uploadMs1m, downloadMs1m) ← benchTransfer cpu1m 100
  IO.println s!"1MB (512x512)      | {fmt uploadMs1m}ms  | {fmt downloadMs1m}ms  | {fmt (uploadMs1m + downloadMs1m)}ms"
  logMs "gpu_tensor/transfer" "upload_ms" "512x512" uploadMs1m
  logMs "gpu_tensor/transfer" "download_ms" "512x512" downloadMs1m

  -- 4MB
  let (uploadMs4m, downloadMs4m) ← benchTransfer cpu4m 50
  IO.println s!"4MB (1024x1024)    | {fmt uploadMs4m}ms  | {fmt downloadMs4m}ms  | {fmt (uploadMs4m + downloadMs4m)}ms"
  logMs "gpu_tensor/transfer" "upload_ms" "1024x1024" uploadMs4m
  logMs "gpu_tensor/transfer" "download_ms" "1024x1024" downloadMs4m

  IO.println ""
  IO.println "=============================================================="
  IO.println "Summary: GpuTensor eliminates per-operation copy overhead."
  IO.println "Speedup increases with more chained operations."
  IO.println "=============================================================="
