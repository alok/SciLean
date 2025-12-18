/-
AMX vs MPS GEMM Benchmark

Compares Accelerate/AMX (CPU) vs MPS (GPU) GEMM performance
at different matrix sizes to find crossover point.

Run: lake build AMXBenchmark && .lake/build/bin/AMXBenchmark
-/
import SciLean.FFI.Metal

open SciLean.Metal

def benchmark (name : String) (m k n : Nat)
    (gemm : GpuBuffer → GpuBuffer → USize → USize → USize → IO GpuBuffer) : IO Unit := do
  -- Create matrices directly on GPU (no CPU overhead)
  let aSize := m * k
  let bSize := k * n

  let A ← GpuBuffer.fill aSize.toUSize 0.5
  let B ← GpuBuffer.fill bSize.toUSize 0.5

  -- Warmup
  for _ in [0:10] do
    let _ ← gemm A B m.toUSize k.toUSize n.toUSize

  -- More iterations for small matrices (need >= 10ms for accurate timing)
  let iterations := if m * n < 100000 then 10000 else if m * n < 1000000 then 1000 else 100

  let start ← IO.monoMsNow
  for _ in [0:iterations] do
    let _ ← gemm A B m.toUSize k.toUSize n.toUSize
  let elapsed := (← IO.monoMsNow) - start

  let flops := 2.0 * m.toFloat * k.toFloat * n.toFloat * iterations.toFloat
  let gflops := flops / elapsed.toFloat / 1e6  -- ms to GFLOP/s

  IO.println s!"  {name}: {gflops} GFLOP/s ({elapsed}ms for {iterations} iters)"

def main : IO Unit := do
  IO.println "AMX vs MPS GEMM Benchmark"
  IO.println "========================="
  IO.println ""

  let sizes := [(64, 64, 64), (128, 128, 128), (256, 256, 256),
                (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]

  for (m, k, n) in sizes do
    IO.println s!"\nMatrix size: {m}x{k} @ {k}x{n}"
    benchmark "AMX " m k n GpuBuffer.gemmAMX
    benchmark "MPS " m k n GpuBuffer.gemm
    benchmark "Auto" m k n GpuBuffer.gemmAuto

  IO.println "\n\nNon-square matrices (typical for MLP):"

  -- 784 -> 128 (MNIST layer 1)
  IO.println "\n784x256 @ 256x128 (MNIST batch layer 1):"
  benchmark "AMX " 784 256 128 GpuBuffer.gemmAMX
  benchmark "MPS " 784 256 128 GpuBuffer.gemm
  benchmark "Auto" 784 256 128 GpuBuffer.gemmAuto

  -- 256 -> 10 (MNIST layer 2)
  IO.println "\n256x128 @ 128x10 (MNIST batch layer 2):"
  benchmark "AMX " 256 128 10 GpuBuffer.gemmAMX
  benchmark "MPS " 256 128 10 GpuBuffer.gemm
  benchmark "Auto" 256 128 10 GpuBuffer.gemmAuto
