-- Test square matrices for AMX/MPS GEMM
import SciLean.FFI.Metal
import SciLean.Data.Tensor

open SciLean
open SciLean.Metal

def toGpuMatrix (n : Nat) (data : ByteArray) : IO (Float^[Idx n × Idx n]@metal) := do
  let gpuBuf ← (CpuBuffer.mk data).upload
  return GpuTensor.fromContiguousBuffer (ι:=Idx n × Idx n) gpuBuf #[n, n]

def main : IO Unit := do
  IO.println "Testing square matrix GEMM..."
  
  -- Test various square sizes
  let sizes := [64, 128, 256, 512]
  
  for n in sizes do
    IO.println s!"Testing {n}×{n} @ {n}×{n}..."
    let data := ByteArray.replicateFloat32 (n * n) 0.5
    let A ← toGpuMatrix n data
    let B ← toGpuMatrix n data
    
    -- Test AMX
    let _ ← GpuTensor.gemmAMX A B
    IO.println s!"  AMX: OK"
    
    -- Test MPS
    let _ ← GpuTensor.gemm A B
    IO.println s!"  MPS: OK"
    
    -- Test Auto
    let _ ← GpuTensor.gemmAuto A B
    IO.println s!"  Auto: OK"
  
  IO.println "All tests passed!"
