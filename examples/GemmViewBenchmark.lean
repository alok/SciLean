import SciLean.Data.Tensor
import SciLean.Data.Tensor.Layout
import SciLean.AD.TensorRevFDeriv
import SciLean.FFI.Metal
import SciLean.Util.Benchmark
import SciLean.Data.ByteArray

/-!
# GPU GEMM Benchmark

Benchmarks GEMM variants for layout views and transpose metadata.
Tests constant-time transpose views and GEMM backward performance.
-/

open SciLean

namespace GemmViewBenchmark

/-- Helper to create a ByteArray from random floats -/
def randomByteArray (n : Nat) : IO ByteArray := do
  let N : Nat := 10^9
  let mut arr := ByteArray.replicateFloat32 n 0.0
  for i in [0:n] do
    let r ← IO.rand 0 N
    let val := (r.toFloat / N.toFloat) - 0.5  -- [-0.5, 0.5]
    arr := arr.usetFloat32 (i * 4).toUSize val.toFloat32
  return arr

/-- Upload ByteArray as a contiguous GPU matrix. -/
def toGpuMatrix (m k : Nat) (data : ByteArray) : IO (Float^[Idx m × Idx k]@metal) := do
  let gpuBuf ← (Metal.CpuBuffer.mk data).upload
  return GpuTensor.fromContiguousBuffer (ι:=Idx m × Idx k) gpuBuf #[m, k]

/-- Benchmark configuration for GPU ops -/
def gpuConfig : Benchmark.Config := { warmupIterations := 5, timedIterations := 20 }

/-- Generic benchmark runner using raw GPU buffers -/
def benchmarkGemmRaw (m k n : Nat) : IO Unit := do
  IO.println s!"\n═══════════════════════════════════════════════════════════════"
  IO.println s!"  GEMM Benchmark: {m}×{k} @ {k}×{n} → {m}×{n}"
  IO.println s!"═══════════════════════════════════════════════════════════════"

  -- Create random matrices
  let aData ← randomByteArray (m * k)
  let bData ← randomByteArray (k * n)
  let dcData ← randomByteArray (m * n)

  -- Upload to GPU
  let aGpu ← toGpuMatrix m k aData
  let bGpu ← toGpuMatrix k n bData
  let dcGpu ← toGpuMatrix m n dcData

  -- For transposed tests (stored transposed, then viewed)
  let atData ← randomByteArray (k * m)
  let btData ← randomByteArray (n * k)
  let aStorage ← toGpuMatrix k m atData
  let bStorage ← toGpuMatrix n k btData
  let aT := GpuTensor.transpose aStorage
  let bT := GpuTensor.transpose bStorage

  let mut suite : Benchmark.Suite := { name := s!"GEMM {m}×{k}×{n}" }

  -- 1. Direct gemm (baseline - what contiguous would use)
  let directResult ← Benchmark.run "Direct gemm (baseline)" gpuConfig fun () => do
    let _ ← GpuTensor.gemm aGpu bGpu
    pure ()
  suite := suite.add directResult

  -- 2. Contiguous view (should be same as direct)
  let viewContiguousResult ← Benchmark.run "Contiguous view" gpuConfig fun () => do
    -- Contiguous A and B → dispatch to gemm
    if aGpu.isContiguous && bGpu.isContiguous then
      let _ ← GpuTensor.gemm aGpu bGpu
    pure ()
  suite := suite.add viewContiguousResult

  -- 3. gemmTN (A transposed)
  let gemmTNResult ← Benchmark.run "gemmTN (A^T @ B)" gpuConfig fun () => do
    let _ ← GpuTensor.gemm aT bGpu
    pure ()
  suite := suite.add gemmTNResult

  -- 4. gemmNT (B transposed)
  let gemmNTResult ← Benchmark.run "gemmNT (A @ B^T)" gpuConfig fun () => do
    let _ ← GpuTensor.gemm aGpu bT
    pure ()
  suite := suite.add gemmNTResult

  -- 5. Transposed A view
  let viewTNResult ← Benchmark.run "A^T view" gpuConfig fun () => do
    -- Simulates: detect transposed, dispatch to gemmTN
    if aT.isTransposed && !bGpu.isTransposed then
      let _ ← GpuTensor.gemm aT bGpu
    pure ()
  suite := suite.add viewTNResult

  -- 6. Transposed B view
  let viewNTResult ← Benchmark.run "B^T view" gpuConfig fun () => do
    if !aGpu.isTransposed && bT.isTransposed then
      let _ ← GpuTensor.gemm aGpu bT
    pure ()
  suite := suite.add viewNTResult

  -- 7. GEMM backward (2 gemm calls using O(1) transpose)
  let backwardResult ← Benchmark.run "Backward (dA + dB)" gpuConfig fun () => do
    -- dA = dC @ B^T via gemmNT
    let _ ← GpuTensor.gemm dcGpu (GpuTensor.transpose bGpu)
    -- dB = A^T @ dC via gemmTN
    let _ ← GpuTensor.gemm (GpuTensor.transpose aGpu) dcGpu
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
  IO.println "║           GPU GEMM Benchmark (Views vs Contiguous)                ║"
  IO.println "║                                                               ║"
  IO.println "║  Tests O(1) transpose views and kernel dispatch overhead      ║"
  IO.println "╚═══════════════════════════════════════════════════════════════╝"

  -- Test multiple sizes
  benchmarkGemmRaw 256 256 256
  benchmarkGemmRaw 512 512 512
  benchmarkGemmRaw 1024 1024 1024
  benchmarkGemmRaw 2048 2048 2048

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

end GemmViewBenchmark

def main := GemmViewBenchmark.main
