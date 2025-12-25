import SciLean.FFI.Metal
import SciLean.FFI.Float32Array

open SciLean

def benchGemmMs (gemm : USize → USize → USize → ByteArray → ByteArray → ByteArray)
    (n : Nat) (numIters warmupIters : Nat) (amat bmat : ByteArray) : IO Float := do
  -- Warmup
  for _ in [:warmupIters] do
    let _ := gemm n.toUSize n.toUSize n.toUSize amat bmat

  -- Benchmark
  let mut sizeAccum := 0
  let start ← IO.monoNanosNow
  for _ in [:numIters] do
    let r := gemm n.toUSize n.toUSize n.toUSize amat bmat
    sizeAccum := sizeAccum + r.size
  let finish ← IO.monoNanosNow

  let totalNs := finish - start
  let avgNs := totalNs / numIters
  let avgMs := avgNs.toFloat / 1_000_000.0
  return avgMs

def benchGemmChunkNs (gemm : USize → USize → USize → ByteArray → ByteArray → ByteArray)
    (n : Nat) (numIters : Nat) (amat bmat : ByteArray) : IO Nat := do
  let mut sizeAccum := 0
  let start ← IO.monoNanosNow
  for _ in [:numIters] do
    let r := gemm n.toUSize n.toUSize n.toUSize amat bmat
    sizeAccum := sizeAccum + r.size
  let finish ← IO.monoNanosNow
  return finish - start

def benchGemmInterleavedMs
    (gemmA gemmB : USize → USize → USize → ByteArray → ByteArray → ByteArray)
    (n : Nat) (numIters warmupIters : Nat) (amat bmat : ByteArray) : IO (Float × Float) := do
  -- Warmup
  for _ in [:warmupIters] do
    let _ := gemmA n.toUSize n.toUSize n.toUSize amat bmat
    let _ := gemmB n.toUSize n.toUSize n.toUSize amat bmat

  -- Interleaved benchmark (single-iter chunks to reduce drift)
  let mut aNs : Nat := 0
  let mut bNs : Nat := 0
  for _ in [:numIters] do
    let aChunk ← benchGemmChunkNs gemmA n 1 amat bmat
    aNs := aNs + aChunk
    let bChunk ← benchGemmChunkNs gemmB n 1 amat bmat
    bNs := bNs + bChunk

  let aMs := aNs.toFloat / numIters.toFloat / 1_000_000.0
  let bMs := bNs.toFloat / numIters.toFloat / 1_000_000.0
  return (aMs, bMs)

def printGemmResult (name : String) (n : Nat) (avgMs : Float) : IO Unit := do
  let avgNs := avgMs * 1_000_000.0
  let flops := 2.0 * n.toFloat * n.toFloat * n.toFloat
  let gflops := if avgNs > 0 then flops / avgNs else 0.0
  let tflops := gflops / 1000.0
  if tflops >= 1.0 then
    IO.println s!"  {name}: {avgMs} ms, {tflops} TFLOP/s"
  else
    IO.println s!"  {name}: {avgMs} ms, {gflops} GFLOP/s"

def main : IO Unit := do
  IO.println "=== Focused GEMM Analysis ==="
  IO.println "Testing M4Pro, MPS, and Accelerate at various sizes\n"

  -- Test at power-of-2 sizes from 128 to 4096
  for log2n in [7, 8, 9, 10, 11, 12] do
    let n := 1 <<< log2n  -- 128, 256, 512, 1024, 2048, 4096
    let baseIters := if n >= 2048 then 10 else if n >= 1024 then 20 else 50
    let numIters := baseIters * 2
    let warmupIters := 10
    IO.println s!"Matrix size: {n}×{n} ({numIters} iterations, warmup {warmupIters})"

    let amat := Metal.Float32.fill (n * n).toUSize (1.0 : Float32)
    let bmat := Metal.Float32.fill (n * n).toUSize (1.0 : Float32)

    -- Compare M4Pro raw vs guarded, plus MPS and Accelerate
    if n % 64 == 0 then
      let (m4RawMs, m4Ms) ←
        benchGemmInterleavedMs Metal.Float32.gemmM4ProRaw Metal.Float32.gemmM4Pro
          n numIters warmupIters amat bmat
      printGemmResult "M4Pro raw  " n m4RawMs
      printGemmResult "M4Pro      " n m4Ms
      let overheadPct :=
        if m4RawMs > 0.0 then
          (m4Ms - m4RawMs) / m4RawMs * 100.0
        else
          0.0
      IO.println s!"  Guard overhead: {overheadPct}%"
    let mpsMs ← benchGemmMs Metal.Float32.gemmMPS n numIters warmupIters amat bmat
    printGemmResult "MPS        " n mpsMs
    let accelMs ← benchGemmMs Metal.Float32.gemmAccelerate n numIters warmupIters amat bmat
    printGemmResult "Accelerate " n accelMs
    IO.println ""

  IO.println "Done!"
