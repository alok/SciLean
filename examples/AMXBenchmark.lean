/-
AMX vs MPS GEMM Benchmark

Compares Accelerate/AMX (CPU) vs MPS (GPU) GEMM performance.
-/
import SciLean.FFI.Metal

open SciLean.Metal

def benchmarkGemm (name : String) (gemm : GpuBuffer → GpuBuffer → USize → USize → USize → IO GpuBuffer)
    (A B : GpuBuffer) (m k n : USize) (iters : Nat) : IO Unit := do
  -- Warmup
  for _ in [0:5] do let _ ← gemm A B m k n
  -- Benchmark
  let start ← IO.monoMsNow
  for _ in [0:iters] do let _ ← gemm A B m k n
  let elapsed := (← IO.monoMsNow) - start
  let flops := 2.0 * m.toNat.toFloat * k.toNat.toFloat * n.toNat.toFloat * iters.toFloat
  let gflops := flops / elapsed.toFloat / 1e6
  IO.println s!"  {name}: {gflops} GFLOP/s ({elapsed}ms / {iters} iters)"

def main : IO Unit := do
  IO.println "═══════════════════════════════════════════════════════════════"
  IO.println "           AMX (Accelerate/CPU) vs MPS (GPU) GEMM"
  IO.println "═══════════════════════════════════════════════════════════════"

  -- MNIST-sized matrices (non-square to avoid MPS buffer size issue)
  let tests := [
    ("Layer 1 fwd [256,784]@[784,128]", 256, 784, 128, 500),
    ("Layer 2 fwd [256,128]@[128,10]", 256, 128, 10, 5000),
    ("Backward dW1 [128,256]@[256,784]", 128, 256, 784, 500),
    ("Backward dW2 [10,256]@[256,128]", 10, 256, 128, 5000)
  ]

  for (name, m, k, n, iters) in tests do
    IO.println s!"\n{name}"
    let aData := ByteArray.replicateFloat32 (m * k) 0.5
    let bData := ByteArray.replicateFloat32 (k * n) 0.5
    let A ← (CpuBuffer.mk aData).upload
    let B ← (CpuBuffer.mk bData).upload
    benchmarkGemm "AMX " GpuBuffer.gemmAMX A B m.toUSize k.toUSize n.toUSize iters
    benchmarkGemm "MPS " GpuBuffer.gemm A B m.toUSize k.toUSize n.toUSize iters
    benchmarkGemm "Auto" GpuBuffer.gemmAuto A B m.toUSize k.toUSize n.toUSize iters

  IO.println "\n═══════════════════════════════════════════════════════════════"
  IO.println "Done!"
