/-
AMX vs MPS GEMM Correctness Test

Compares output of AMX and MPS GEMM implementations to verify correctness.
-/
import SciLean.FFI.Metal

open SciLean.Metal

def testGemm : IO Unit := do
  IO.println "Testing standard GEMM (C = A @ B)"
  IO.println "================================="

  -- Small test: 2x3 @ 3x4 = 2x4
  let m : USize := 2
  let k : USize := 3
  let n : USize := 4

  -- Create test data
  -- A = [[1, 2, 3], [4, 5, 6]]  (2x3)
  -- B = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]  (3x4)
  -- Expected C = A @ B:
  -- C[0,0] = 1*1 + 2*5 + 3*9 = 1 + 10 + 27 = 38
  -- C[0,1] = 1*2 + 2*6 + 3*10 = 2 + 12 + 30 = 44
  -- ...

  let aData : ByteArray := ByteArray.replicateFloat32 6 0.0
    |>.usetFloat32 0 1.0  |>.usetFloat32 4 2.0  |>.usetFloat32 8 3.0
    |>.usetFloat32 12 4.0 |>.usetFloat32 16 5.0 |>.usetFloat32 20 6.0

  let bData : ByteArray := ByteArray.replicateFloat32 12 0.0
    |>.usetFloat32 0 1.0  |>.usetFloat32 4 2.0  |>.usetFloat32 8 3.0  |>.usetFloat32 12 4.0
    |>.usetFloat32 16 5.0 |>.usetFloat32 20 6.0 |>.usetFloat32 24 7.0 |>.usetFloat32 28 8.0
    |>.usetFloat32 32 9.0 |>.usetFloat32 36 10.0 |>.usetFloat32 40 11.0 |>.usetFloat32 44 12.0

  let cpuA := CpuBuffer.mk aData
  let cpuB := CpuBuffer.mk bData

  let gpuA ← cpuA.upload
  let gpuB ← cpuB.upload

  -- Run MPS GEMM
  let mpsC ← GpuBuffer.gemm gpuA gpuB m k n
  let mpsCpu ← mpsC.download

  -- Run AMX GEMM
  let amxC ← GpuBuffer.gemmAMX gpuA gpuB m k n
  let amxCpu ← amxC.download

  IO.println s!"  MPS result:"
  for i in [0:8] do
    let val := mpsCpu.data.ugetFloat32 (i * 4).toUSize
    IO.print s!"  {val}"
  IO.println ""

  IO.println s!"  AMX result:"
  for i in [0:8] do
    let val := amxCpu.data.ugetFloat32 (i * 4).toUSize
    IO.print s!"  {val}"
  IO.println ""

def testGemmNT : IO Unit := do
  IO.println "\nTesting GEMM NT (C = A @ B^T)"
  IO.println "============================="

  -- For MNIST forward: x[batch, 784] @ W1^T[784, 128] = h[batch, 128]
  -- Test: A[2, 3] @ B^T[3, 4] = C[2, 4]
  -- B is stored as [4, 3], transposed becomes [3, 4]

  let m : USize := 2
  let k : USize := 3
  let n : USize := 4

  -- A = [[1, 2, 3], [4, 5, 6]]  (2x3)
  let aData : ByteArray := ByteArray.replicateFloat32 6 0.0
    |>.usetFloat32 0 1.0  |>.usetFloat32 4 2.0  |>.usetFloat32 8 3.0
    |>.usetFloat32 12 4.0 |>.usetFloat32 16 5.0 |>.usetFloat32 20 6.0

  -- B stored as [4, 3] = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
  -- B^T = [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]]  (3x4)
  let bData : ByteArray := ByteArray.replicateFloat32 12 0.0
    |>.usetFloat32 0 1.0  |>.usetFloat32 4 2.0  |>.usetFloat32 8 3.0
    |>.usetFloat32 12 4.0 |>.usetFloat32 16 5.0 |>.usetFloat32 20 6.0
    |>.usetFloat32 24 7.0 |>.usetFloat32 28 8.0 |>.usetFloat32 32 9.0
    |>.usetFloat32 36 10.0 |>.usetFloat32 40 11.0 |>.usetFloat32 44 12.0

  let cpuA := CpuBuffer.mk aData
  let cpuB := CpuBuffer.mk bData

  let gpuA ← cpuA.upload
  let gpuB ← cpuB.upload

  -- Run MPS GEMM NT
  let mpsC ← GpuBuffer.gemmNT gpuA gpuB m k n
  let mpsCpu ← mpsC.download

  -- Run AMX GEMM NT
  let amxC ← GpuBuffer.gemmNT_AMX gpuA gpuB m k n
  let amxCpu ← amxC.download

  IO.println s!"  MPS result:"
  for i in [0:8] do
    let val := mpsCpu.data.ugetFloat32 (i * 4).toUSize
    IO.print s!"  {val}"
  IO.println ""

  IO.println s!"  AMX result:"
  for i in [0:8] do
    let val := amxCpu.data.ugetFloat32 (i * 4).toUSize
    IO.print s!"  {val}"
  IO.println ""

def testGemmTN : IO Unit := do
  IO.println "\nTesting GEMM TN (C = A^T @ B)"
  IO.println "============================="

  -- For backward: d_o^T[10, batch] @ h[batch, 128] = dw2[10, 128]
  -- A is stored as [batch, 10], transposed becomes [10, batch]
  -- Test: A^T[2, 3] @ B[3, 4] = C[2, 4]
  -- A is stored as [3, 2]

  let m : USize := 2
  let k : USize := 3
  let n : USize := 4

  -- A stored as [3, 2] = [[1, 2], [3, 4], [5, 6]]
  -- A^T = [[1, 3, 5], [2, 4, 6]]  (2x3)
  let aData : ByteArray := ByteArray.replicateFloat32 6 0.0
    |>.usetFloat32 0 1.0  |>.usetFloat32 4 2.0
    |>.usetFloat32 8 3.0  |>.usetFloat32 12 4.0
    |>.usetFloat32 16 5.0 |>.usetFloat32 20 6.0

  -- B = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]  (3x4)
  let bData : ByteArray := ByteArray.replicateFloat32 12 0.0
    |>.usetFloat32 0 1.0  |>.usetFloat32 4 2.0  |>.usetFloat32 8 3.0  |>.usetFloat32 12 4.0
    |>.usetFloat32 16 5.0 |>.usetFloat32 20 6.0 |>.usetFloat32 24 7.0 |>.usetFloat32 28 8.0
    |>.usetFloat32 32 9.0 |>.usetFloat32 36 10.0 |>.usetFloat32 40 11.0 |>.usetFloat32 44 12.0

  let cpuA := CpuBuffer.mk aData
  let cpuB := CpuBuffer.mk bData

  let gpuA ← cpuA.upload
  let gpuB ← cpuB.upload

  -- Run MPS GEMM TN
  let mpsC ← GpuBuffer.gemmTN gpuA gpuB m k n
  let mpsCpu ← mpsC.download

  -- Run AMX GEMM TN
  let amxC ← GpuBuffer.gemmTN_AMX gpuA gpuB m k n
  let amxCpu ← amxC.download

  IO.println s!"  MPS result:"
  for i in [0:8] do
    let val := mpsCpu.data.ugetFloat32 (i * 4).toUSize
    IO.print s!"  {val}"
  IO.println ""

  IO.println s!"  AMX result:"
  for i in [0:8] do
    let val := amxCpu.data.ugetFloat32 (i * 4).toUSize
    IO.print s!"  {val}"
  IO.println ""

def main : IO Unit := do
  IO.println "AMX vs MPS GEMM Correctness Test"
  IO.println "================================"

  testGemm
  testGemmNT
  testGemmTN

  IO.println "\nDone!"
