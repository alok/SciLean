import SciLean.Data.Tensor
import SciLean.Data.Tensor.Layout
import SciLean.AD.TensorRevFDeriv
import SciLean.FFI.Metal
import SciLean.FFI.Metal.StridedBuffer
import SciLean.Util.Benchmark
import SciLean.Data.ByteArray

/-!
# GPU GEMM Benchmark

Benchmarks GEMM variants for layout views and transpose metadata.
Tests constant-time transpose views and GEMM backward performance.
-/

open SciLean

namespace StridedGemmBenchmark

/-- Helper to create a ByteArray from random floats -/
def randomByteArray (n : Nat) : IO ByteArray := do
  let N : Nat := 10^9
  let mut arr := ByteArray.replicateFloat32 n 0.0
  for i in [0:n] do
    let r ← IO.rand 0 N
    let val := (r.toFloat / N.toFloat) - 0.5  -- [-0.5, 0.5]
    arr := arr.usetFloat32 (i * 4).toUSize val.toFloat32
  return arr

/-- Benchmark configuration for GPU ops -/
def gpuConfig : Benchmark.Config := { warmupIterations := 5, timedIterations := 20 }

/-- Generic benchmark runner using raw GPU buffers -/
def benchmarkGemmRaw (m k n : Nat) (label : String) : IO Unit := do
  IO.println s!"\n═══════════════════════════════════════════════════════════════"
  IO.println s!"  GEMM Benchmark: {m}×{k} @ {k}×{n} → {m}×{n}"
  IO.println s!"═══════════════════════════════════════════════════════════════"

  -- Create random matrices
  let aData ← randomByteArray (m * k)
  let bData ← randomByteArray (k * n)
  let dcData ← randomByteArray (m * n)

  -- Upload to GPU
  let aGpu ← Metal.GpuBuffer.fromByteArray aData
  let bGpu ← Metal.GpuBuffer.fromByteArray bData
  let dcGpu ← Metal.GpuBuffer.fromByteArray dcData

  -- For transposed tests
  let atData ← randomByteArray (k * m)
  let btData ← randomByteArray (n * k)
  let atGpu ← Metal.GpuBuffer.fromByteArray atData
  let btGpu ← Metal.GpuBuffer.fromByteArray btData

  -- Create strided layouts
  let layoutA := TensorLayout.contiguous #[m, k]
  let layoutB := TensorLayout.contiguous #[k, n]
  let layoutDC := TensorLayout.contiguous #[m, n]

  -- Transposed layouts
  let layoutAT_storage := TensorLayout.contiguous #[k, m]
  let layoutAT := layoutAT_storage.transpose  -- (m, k) but transposed
  let layoutBT_storage := TensorLayout.contiguous #[n, k]
  let layoutBT := layoutBT_storage.transpose  -- (k, n) but transposed

  let mU := m.toUSize
  let kU := k.toUSize
  let nU := n.toUSize

  let mut suite : Benchmark.Suite := { name := s!"GEMM {m}×{k}×{n}" }

  -- 1. Direct gemm (baseline - what legacy would use)
  let directResult ← Benchmark.run "Direct gemm (baseline)" gpuConfig fun () => do
    let _ ← Metal.GpuBuffer.gemm aGpu bGpu mU kU nU
    pure ()
  suite := suite.add directResult

  -- 2. Contiguous view (should be same as direct)
  let stridedContiguousResult ← Benchmark.run "Contiguous view" gpuConfig fun () => do
    -- Contiguous A and B → dispatch to gemm
    if layoutA.isContiguous && layoutB.isContiguous then
      let _ ← Metal.GpuBuffer.gemm aGpu bGpu mU kU nU
    pure ()
  suite := suite.add stridedContiguousResult

  -- 3. gemmTN (A transposed)
  let gemmTNResult ← Benchmark.run "gemmTN (A^T @ B)" gpuConfig fun () => do
    let _ ← Metal.GpuBuffer.gemmTN atGpu bGpu mU kU nU
    pure ()
  suite := suite.add gemmTNResult

  -- 4. gemmNT (B transposed)
  let gemmNTResult ← Benchmark.run "gemmNT (A @ B^T)" gpuConfig fun () => do
    let _ ← Metal.GpuBuffer.gemmNT aGpu btGpu mU kU nU
    pure ()
  suite := suite.add gemmNTResult

  -- 5. Transposed A view
  let stridedTNResult ← Benchmark.run "A^T view" gpuConfig fun () => do
    -- Simulates: detect transposed, dispatch to gemmTN
    if layoutAT.isTransposed && !layoutB.isTransposed then
      let _ ← Metal.GpuBuffer.gemmTN atGpu bGpu mU kU nU
    pure ()
  suite := suite.add stridedTNResult

  -- 6. Transposed B view
  let stridedNTResult ← Benchmark.run "B^T view" gpuConfig fun () => do
    if !layoutA.isTransposed && layoutBT.isTransposed then
      let _ ← Metal.GpuBuffer.gemmNT aGpu btGpu mU kU nU
    pure ()
  suite := suite.add stridedNTResult

  -- 7. GEMM backward (2 gemm calls using O(1) transpose)
  let backwardResult ← Benchmark.run "Backward (dA + dB)" gpuConfig fun () => do
    -- dA = dC @ B^T via gemmNT
    let _ ← Metal.GpuBuffer.gemmNT dcGpu bGpu mU nU kU
    -- dB = A^T @ dC via gemmTN
    let _ ← Metal.GpuBuffer.gemmTN aGpu dcGpu kU mU nU
    pure ()
  suite := suite.add backwardResult

  suite.print

  -- Print performance info
  let flops := 2 * m * k * n
  let gflops := flops.toFloat / 1_000_000_000.0
  let avgTimeS := directResult.avgTimeNs.toFloat / 1_000_000_000.0
  let tflops := if avgTimeS > 0 then gflops / avgTimeS / 1000.0 else 0
  IO.println s!"\n  FLOPs: {Float.round (gflops * 100) / 100} GFLOPs"
  IO.println s!  "  Performance: ~{Float.round (tflops * 100) / 100} TFLOPs/s"

/-- Benchmark constant-time transpose overhead (pure metadata, no GPU). -/
def benchmarkTransposeOverhead : IO Unit := do
  IO.println s!"\n═══════════════════════════════════════════════════════════════"
  IO.println s!"  O(1) Transpose Overhead (metadata only, no GPU)"
  IO.println s!"═══════════════════════════════════════════════════════════════"

  let sizes := [256, 512, 1024, 2048, 4096]

  for size in sizes do
    let layout := TensorLayout.contiguous #[size, size]

    -- Measure transpose time (pure metadata manipulation)
    let transposeConfig : Benchmark.Config := { warmupIterations := 1000, timedIterations := 10000 }

    let result ← Benchmark.run s!"transpose {size}×{size}" transposeConfig fun () => do
      let transposed := layout.transpose
      -- Access to prevent optimization
      let _ := transposed.isTransposed
      pure ()

    IO.println s!"  {size}×{size}: {Benchmark.formatTime result.avgTimeNs} avg"

  IO.println "\n  (Transpose is O(1) - just swaps two array elements)"

/-- Benchmark isTransposed check overhead -/
def benchmarkDispatchOverhead : IO Unit := do
  IO.println s!"\n═══════════════════════════════════════════════════════════════"
  IO.println s!"  Dispatch Overhead (layout checks)"
  IO.println s!"═══════════════════════════════════════════════════════════════"

  let layout := TensorLayout.contiguous #[1024, 1024]
  let transposed := layout.transpose

  let checkConfig : Benchmark.Config := { warmupIterations := 1000, timedIterations := 10000 }

  let contiguousCheck ← Benchmark.run "isContiguous check" checkConfig fun () => do
    let _ := layout.isContiguous
    pure ()

  let transposedCheck ← Benchmark.run "isTransposed check" checkConfig fun () => do
    let _ := transposed.isTransposed
    pure ()

  IO.println s!"  isContiguous: {Benchmark.formatTime contiguousCheck.avgTimeNs}"
  IO.println s!"  isTransposed: {Benchmark.formatTime transposedCheck.avgTimeNs}"

/-- Main benchmark entry point -/
def main : IO Unit := do
  if !Metal.isAvailable () then
    IO.println "Metal GPU not available, cannot run benchmarks"
    return

  IO.println "╔═══════════════════════════════════════════════════════════════╗"
  IO.println "║           GPU GEMM Benchmark (Views vs Legacy)                ║"
  IO.println "║                                                               ║"
  IO.println "║  Tests O(1) transpose views and kernel dispatch overhead      ║"
  IO.println "╚═══════════════════════════════════════════════════════════════╝"

  -- Test multiple sizes
  benchmarkGemmRaw 256 256 256 "small"
  benchmarkGemmRaw 512 512 512 "medium"
  benchmarkGemmRaw 1024 1024 1024 "large"
  benchmarkGemmRaw 2048 2048 2048 "xlarge"

  -- Measure overhead
  benchmarkTransposeOverhead
  benchmarkDispatchOverhead

  IO.println "\n══════════════════════════════════════════════════════════════════"
  IO.println "  Benchmark Complete"
  IO.println ""
  IO.println "  Key findings:"
  IO.println "  • O(1) transpose = just metadata swap, no GPU copy"
  IO.println "  • View dispatch adds ~nanoseconds overhead"
  IO.println "  • GEMM backward uses same kernels, just different dispatch"
  IO.println "══════════════════════════════════════════════════════════════════"

end StridedGemmBenchmark

def main := StridedGemmBenchmark.main
