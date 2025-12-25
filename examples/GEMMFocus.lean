import SciLean.FFI.Metal
import SciLean.FFI.Float32Array

open SciLean

def benchGemmMs (gemm : USize → USize → USize → ByteArray → ByteArray → ByteArray)
    (n : Nat) (numIters : Nat) (amat bmat : ByteArray) : IO Float := do
  -- Warmup
  for _ in [:5] do
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
    let numIters := if n >= 2048 then 10 else if n >= 1024 then 20 else 50
    IO.println s!"Matrix size: {n}×{n} ({numIters} iterations)"

    let amat := Metal.Float32.fill (n * n).toUSize (1.0 : Float32)
    let bmat := Metal.Float32.fill (n * n).toUSize (1.0 : Float32)

    -- Compare M4Pro raw vs guarded, plus MPS and Accelerate
    if n % 64 == 0 then
      let m4RawMs ← benchGemmMs Metal.Float32.gemmM4ProRaw n numIters amat bmat
      printGemmResult "M4Pro raw  " n m4RawMs
      let m4Ms ← benchGemmMs Metal.Float32.gemmM4Pro n numIters amat bmat
      printGemmResult "M4Pro      " n m4Ms
      let overheadPct :=
        if m4RawMs > 0.0 then
          (m4Ms - m4RawMs) / m4RawMs * 100.0
        else
          0.0
      IO.println s!"  Guard overhead: {overheadPct}%"
    let mpsMs ← benchGemmMs Metal.Float32.gemmMPS n numIters amat bmat
    printGemmResult "MPS        " n mpsMs
    let accelMs ← benchGemmMs Metal.Float32.gemmAccelerate n numIters amat bmat
    printGemmResult "Accelerate " n accelMs
    IO.println ""

  IO.println "Done!"
