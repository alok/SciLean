/-
AMX vs MPS GEMM Benchmark

Compares Accelerate/AMX (CPU) vs MPS (GPU) GEMM performance.
-/
import SciLean.FFI.Metal
import SciLean.Util.Benchmark
import SciLean.Data.Tensor

open SciLean
open SciLean.Metal

/-- Upload ByteArray as a contiguous GPU matrix. -/
def toGpuMatrix (m k : Nat) (data : ByteArray) : IO (Float^[Idx m × Idx k]@metal) := do
  let gpuBuf ← (CpuBuffer.mk data).upload
  return GpuTensor.fromContiguousBuffer (ι:=Idx m × Idx k) gpuBuf #[m, k]

def benchmarkGemm {m k n : Nat} (caseName backendName : String)
    (gemm : Float^[Idx m × Idx k]@metal → Float^[Idx k × Idx n]@metal → IO (Float^[Idx m × Idx n]@metal))
    (A : Float^[Idx m × Idx k]@metal) (B : Float^[Idx k × Idx n]@metal)
    (iters : Nat) : IO Unit := do
  -- Warmup
  for _ in [0:5] do let _ ← gemm A B
  -- Benchmark
  let start ← IO.monoMsNow
  for _ in [0:iters] do let _ ← gemm A B
  let elapsed := (← IO.monoMsNow) - start
  let elapsedMs := elapsed.toFloat
  let flops := 2.0 * m.toFloat * k.toFloat * n.toFloat * iters.toFloat
  let gflops := flops / elapsedMs / 1e6
  IO.println s!"  {backendName}: {gflops} GFLOP/s ({elapsed}ms / {iters} iters)"
  let params := [
    SciLean.Benchmark.paramStr "case" caseName,
    SciLean.Benchmark.paramStr "backend" backendName,
    SciLean.Benchmark.paramNat "m" m,
    SciLean.Benchmark.paramNat "k" k,
    SciLean.Benchmark.paramNat "n" n,
    SciLean.Benchmark.paramNat "iters" iters
  ]
  SciLean.Benchmark.logMetric "amx_gemm" "gflops" gflops (unit? := some "GFLOP/s") (params := params)
  SciLean.Benchmark.logMetric "amx_gemm" "elapsed_ms" elapsedMs (unit? := some "ms") (params := params)

def main : IO Unit := do
  if !Metal.isAvailable () then
    IO.println "Metal GPU not available"
    return

  let quick := (← IO.getEnv "SCILEAN_BENCH_QUICK").isSome
  if quick then
    IO.println "Quick mode enabled (SCILEAN_BENCH_QUICK)"

  IO.println "═══════════════════════════════════════════════════════════════"
  IO.println "           AMX (Accelerate/CPU) vs MPS (GPU) GEMM"
  IO.println "═══════════════════════════════════════════════════════════════"

  -- MNIST-sized matrices (non-square to avoid MPS buffer size issue)
  let tests :=
    if quick then
      [
        ("Layer 1 fwd [256,784]@[784,128]", 256, 784, 128, 100),
        ("Layer 2 fwd [256,128]@[128,10]", 256, 128, 10, 300)
      ]
    else
      [
        ("Layer 1 fwd [256,784]@[784,128]", 256, 784, 128, 500),
        ("Layer 2 fwd [256,128]@[128,10]", 256, 128, 10, 5000),
        ("Backward dW1 [128,256]@[256,784]", 128, 256, 784, 500),
        ("Backward dW2 [10,256]@[256,128]", 10, 256, 128, 5000)
      ]

  for (name, m, k, n, iters) in tests do
    IO.println s!"\n{name}"
    let aData := ByteArray.replicateFloat32 (m * k) 0.5
    let bData := ByteArray.replicateFloat32 (k * n) 0.5
    let A ← toGpuMatrix m k aData
    let B ← toGpuMatrix k n bData
    benchmarkGemm name "AMX" GpuTensor.gemmAMX A B iters
    benchmarkGemm name "MPS" GpuTensor.gemm A B iters
    benchmarkGemm name "Auto" GpuTensor.gemmAuto A B iters

  IO.println "\n═══════════════════════════════════════════════════════════════"
  IO.println "Done!"
