-- Test square matrices for AMX/MPS GEMM
import SciLean.FFI.Metal

open SciLean.Metal

def main : IO Unit := do
  IO.println "Testing square matrix GEMM..."
  
  -- Test various square sizes
  let sizes := [64, 128, 256, 512]
  
  for n in sizes do
    IO.println s!"Testing {n}×{n} @ {n}×{n}..."
    let data := ByteArray.replicateFloat32 (n * n) 0.5
    let A ← (CpuBuffer.mk data).upload
    let B ← (CpuBuffer.mk data).upload
    
    -- Test AMX
    let _ ← GpuBuffer.gemmAMX A B n.toUSize n.toUSize n.toUSize
    IO.println s!"  AMX: OK"
    
    -- Test MPS
    let _ ← GpuBuffer.gemm A B n.toUSize n.toUSize n.toUSize
    IO.println s!"  MPS: OK"
    
    -- Test Auto
    let _ ← GpuBuffer.gemmAuto A B n.toUSize n.toUSize n.toUSize
    IO.println s!"  Auto: OK"
  
  IO.println "All tests passed!"
