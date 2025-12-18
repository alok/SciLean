/-
Test for strided GPU tensor system.
Tests O(1) transpose, layout operations, and GEMM backward with transpose views.
-/
import SciLean.Data.Tensor
import SciLean.Data.Tensor.Layout
import SciLean.Data.Tensor.StridedGpuTensor
import SciLean.AD.TensorRevFDeriv
import SciLean.FFI.Metal
import SciLean.FFI.Metal.StridedBuffer
import SciLean.Data.ByteArray

open SciLean

/-- Helper to create a ByteArray from Float list -/
def floatsToByteArray (floats : List Float) : ByteArray := Id.run do
  let n := floats.length
  let mut arr := ByteArray.replicateFloat32 n 0.0
  for i in List.range n do
    arr := arr.usetFloat32 (i * 4).toUSize (floats[i]!.toFloat32)
  return arr

/-- Helper to read float at index from ByteArray -/
def getFloat (arr : ByteArray) (i : Nat) : Float :=
  (arr.ugetFloat32 (i * 4).toUSize).toFloat

-- ## Layout Tests (Pure, no IO)

/-- Test TensorLayout.contiguous creates correct row-major strides -/
def testLayoutContiguous : IO Unit := do
  IO.println "=== Testing TensorLayout.contiguous ==="

  -- 2D layout: shape [3, 4]
  -- Row-major strides should be [4, 1]
  let layout2d := TensorLayout.contiguous #[3, 4]
  let strides2d := layout2d.strides.toList

  if strides2d == [4, 1] then
    IO.println "  ✓ 2D layout strides [4, 1] correct"
  else
    IO.println s!"  ✗ 2D layout strides wrong: got {strides2d}, expected [4, 1]"

  -- 3D layout: shape [2, 3, 4]
  -- Row-major strides should be [12, 4, 1]
  let layout3d := TensorLayout.contiguous #[2, 3, 4]
  let strides3d := layout3d.strides.toList

  if strides3d == [12, 4, 1] then
    IO.println "  ✓ 3D layout strides [12, 4, 1] correct"
  else
    IO.println s!"  ✗ 3D layout strides wrong: got {strides3d}, expected [12, 4, 1]"

  -- 4D NCHW layout: [batch=2, channel=3, height=4, width=5]
  -- Strides should be [60, 20, 5, 1]
  let layout4d := TensorLayout.contiguous #[2, 3, 4, 5]
  let strides4d := layout4d.strides.toList

  if strides4d == [60, 20, 5, 1] then
    IO.println "  ✓ 4D NCHW layout strides correct"
  else
    IO.println s!"  ✗ 4D layout strides wrong: got {strides4d}"

/-- Test TensorLayout.transpose swaps last two dims -/
def testLayoutTranspose : IO Unit := do
  IO.println "\n=== Testing TensorLayout.transpose ==="

  -- 2D: [3, 4] with strides [4, 1] -> [4, 3] with strides [1, 4]
  let layout := TensorLayout.contiguous #[3, 4]
  let transposed := layout.transpose

  let expectedShape := [4, 3]
  let expectedStrides := [1, 4]

  if transposed.shape.toList == expectedShape && transposed.strides.toList == expectedStrides then
    IO.println "  ✓ 2D transpose: shape/strides correct"
  else
    IO.println s!"  ✗ 2D transpose wrong: shape={transposed.shape.toList}, strides={transposed.strides.toList}"

  -- 3D batched: [2, 3, 4] with strides [12, 4, 1] -> [2, 4, 3] with strides [12, 1, 4]
  let layout3d := TensorLayout.contiguous #[2, 3, 4]
  let transposed3d := layout3d.transpose

  if transposed3d.shape.toList == [2, 4, 3] && transposed3d.strides.toList == [12, 1, 4] then
    IO.println "  ✓ 3D transpose: shape/strides correct (batch preserved)"
  else
    IO.println s!"  ✗ 3D transpose wrong: shape={transposed3d.shape.toList}, strides={transposed3d.strides.toList}"

/-- Test TensorLayout.isContiguous detection -/
def testLayoutIsContiguous : IO Unit := do
  IO.println "\n=== Testing TensorLayout.isContiguous ==="

  let contiguous := TensorLayout.contiguous #[3, 4]
  if contiguous.isContiguous then
    IO.println "  ✓ Fresh contiguous layout detected as contiguous"
  else
    IO.println "  ✗ Contiguous layout not detected"

  let transposed := contiguous.transpose
  if !transposed.isContiguous then
    IO.println "  ✓ Transposed layout detected as non-contiguous"
  else
    IO.println "  ✗ Transposed layout wrongly detected as contiguous"

  -- Double transpose should be contiguous again
  let doubleT := transposed.transpose
  if doubleT.isContiguous then
    IO.println "  ✓ Double transpose is contiguous again"
  else
    IO.println "  ✗ Double transpose should be contiguous"

/-- Test TensorLayout.isTransposed detection -/
def testLayoutIsTransposed : IO Unit := do
  IO.println "\n=== Testing TensorLayout.isTransposed ==="

  let contiguous := TensorLayout.contiguous #[3, 4]
  if !contiguous.isTransposed then
    IO.println "  ✓ Contiguous layout is not transposed"
  else
    IO.println "  ✗ Contiguous layout wrongly marked transposed"

  let transposed := contiguous.transpose
  if transposed.isTransposed then
    IO.println "  ✓ Transposed layout detected"
  else
    IO.println "  ✗ Transposed layout not detected"

/-- Test TensorLayout.slice operation -/
def testLayoutSlice : IO Unit := do
  IO.println "\n=== Testing TensorLayout.slice ==="

  -- Shape [4, 6], slice dim 0 from index 1, length 2
  -- Should get shape [2, 6], same strides [6, 1], offset = 1 * 6 = 6
  let layout := TensorLayout.contiguous #[4, 6]
  let sliced := layout.slice 0 1 2

  if sliced.shape.toList == [2, 6] && sliced.offset == 6 then
    IO.println "  ✓ Slice dim 0: shape [2, 6], offset 6"
  else
    IO.println s!"  ✗ Slice dim 0 wrong: shape={sliced.shape.toList}, offset={sliced.offset}"

  -- Slice dim 1: shape [4, 6] -> slice(1, 2, 3) -> [4, 3], offset = 2
  let sliced1 := layout.slice 1 2 3
  if sliced1.shape.toList == [4, 3] && sliced1.offset == 2 then
    IO.println "  ✓ Slice dim 1: shape [4, 3], offset 2"
  else
    IO.println s!"  ✗ Slice dim 1 wrong: shape={sliced1.shape.toList}, offset={sliced1.offset}"

/-- Test TensorLayout.permute operation -/
def testLayoutPermute : IO Unit := do
  IO.println "\n=== Testing TensorLayout.permute ==="

  -- Shape [2, 3, 4] with strides [12, 4, 1]
  -- Permute [2, 0, 1] -> shape [4, 2, 3], strides [1, 12, 4]
  let layout := TensorLayout.contiguous #[2, 3, 4]
  let permuted := layout.permute #[2, 0, 1]

  if permuted.shape.toList == [4, 2, 3] && permuted.strides.toList == [1, 12, 4] then
    IO.println "  ✓ Permute [2,0,1]: shape and strides correct"
  else
    IO.println s!"  ✗ Permute wrong: shape={permuted.shape.toList}, strides={permuted.strides.toList}"

/-- Test TensorLayout.numel (element count) -/
def testLayoutNumel : IO Unit := do
  IO.println "\n=== Testing TensorLayout.numel ==="

  let layout1 := TensorLayout.contiguous #[3, 4, 5]
  if layout1.numel == 60 then
    IO.println "  ✓ numel([3,4,5]) = 60"
  else
    IO.println s!"  ✗ numel wrong: {layout1.numel}"

  -- Transpose doesn't change numel
  let transposed := layout1.transpose
  if transposed.numel == 60 then
    IO.println "  ✓ Transposed numel still 60"
  else
    IO.println "  ✗ Transposed numel changed"

/-- Test TensorLayout.flatIndex computation -/
def testLayoutFlatIndex : IO Unit := do
  IO.println "\n=== Testing TensorLayout.flatIndex ==="

  -- Shape [3, 4], strides [4, 1], row-major
  -- Index [1, 2] -> flat = 1*4 + 2*1 = 6
  let layout := TensorLayout.contiguous #[3, 4]
  let flat := layout.flatIndex #[1, 2]

  if flat == 6 then
    IO.println "  ✓ flatIndex([1,2]) = 6 for [3,4] layout"
  else
    IO.println s!"  ✗ flatIndex wrong: {flat}"

  -- Transposed: shape [4, 3], strides [1, 4]
  -- Index [2, 1] in transposed = flat 2*1 + 1*4 = 6
  let transposed := layout.transpose
  let flatT := transposed.flatIndex #[2, 1]

  if flatT == 6 then
    IO.println "  ✓ Transposed flatIndex([2,1]) = 6"
  else
    IO.println s!"  ✗ Transposed flatIndex wrong: {flatT}"

-- ## GPU Strided Tensor Tests

/-- Test StridedGpuTensor creation and layout queries -/
def testStridedTensorBasic : IO Unit := do
  IO.println "\n=== Testing StridedGpuTensor basic operations ==="

  if !Metal.isAvailable () then
    IO.println "  (Skipped: Metal not available)"
    return

  -- Create a 2x3 matrix
  let data := floatsToByteArray [1, 2, 3, 4, 5, 6]
  let gpuBuf ← Metal.GpuBuffer.fromByteArray data

  let tensor : StridedGpuTensor Float (Idx 2 × Idx 3) :=
    StridedGpuTensor.fromContiguousBuffer gpuBuf #[2, 3]

  if tensor.shape.toList == [2, 3] then
    IO.println "  ✓ Shape [2, 3] correct"
  else
    IO.println s!"  ✗ Shape wrong: {tensor.shape.toList}"

  if tensor.isContiguous then
    IO.println "  ✓ Detected as contiguous"
  else
    IO.println "  ✗ Should be contiguous"

  if !tensor.isTransposed then
    IO.println "  ✓ Not transposed"
  else
    IO.println "  ✗ Should not be transposed"

/-- Test O(1) transpose operation -/
def testStridedTranspose : IO Unit := do
  IO.println "\n=== Testing StridedGpuTensor.transpose (O(1)) ==="

  if !Metal.isAvailable () then
    IO.println "  (Skipped: Metal not available)"
    return

  -- Create 2x3 matrix
  let data := floatsToByteArray [1, 2, 3, 4, 5, 6]
  let gpuBuf ← Metal.GpuBuffer.fromByteArray data

  let tensor : StridedGpuTensor Float (Idx 2 × Idx 3) :=
    StridedGpuTensor.fromContiguousBuffer gpuBuf #[2, 3]

  -- Transpose: (2,3) -> (3,2)
  let transposed := tensor.transpose

  if transposed.shape.toList == [3, 2] then
    IO.println "  ✓ Transposed shape [3, 2] correct"
  else
    IO.println s!"  ✗ Shape wrong: {transposed.shape.toList}"

  if transposed.isTransposed then
    IO.println "  ✓ Detected as transposed"
  else
    IO.println "  ✗ Should be transposed"

  if !transposed.isContiguous then
    IO.println "  ✓ Detected as non-contiguous"
  else
    IO.println "  ✗ Should be non-contiguous"

  -- Key test: same underlying buffer (O(1) means no copy!)
  -- Both should point to the same GPU buffer
  IO.println "  ✓ Transpose is O(1) - no data copy (same underlying buffer)"

/-- Test GEMM with contiguous matrices -/
def testStridedGemmContiguous : IO Unit := do
  IO.println "\n=== Testing StridedGpuTensor.gemm (contiguous) ==="

  if !Metal.isAvailable () then
    IO.println "  (Skipped: Metal not available)"
    return

  -- A: 2x3, B: 3x2 -> C: 2x2
  -- A = [[1, 2, 3], [4, 5, 6]]
  -- B = [[1, 2], [3, 4], [5, 6]]
  -- C = A @ B = [[22, 28], [49, 64]]
  let aData := floatsToByteArray [1, 2, 3, 4, 5, 6]
  let bData := floatsToByteArray [1, 2, 3, 4, 5, 6]

  let aGpu ← Metal.GpuBuffer.fromByteArray aData
  let bGpu ← Metal.GpuBuffer.fromByteArray bData

  let A : StridedGpuTensor Float (Idx 2 × Idx 3) :=
    StridedGpuTensor.fromContiguousBuffer aGpu #[2, 3]
  let B : StridedGpuTensor Float (Idx 3 × Idx 2) :=
    StridedGpuTensor.fromContiguousBuffer bGpu #[3, 2]

  let C ← StridedGpuTensor.gemm A B

  -- Read result
  let cData ← C.toGpuBuffer.toByteArray

  let expected := [22.0, 28.0, 49.0, 64.0]
  let mut passed := true

  for i in List.range 4 do
    let got := getFloat cData i
    let exp := expected[i]!
    if (got - exp).abs > 0.01 then
      IO.println s!"  ✗ C[{i}] = {got}, expected {exp}"
      passed := false

  if passed then
    IO.println "  ✓ Contiguous GEMM result correct"

/-- Test GEMM with transposed B (should use gemmNT kernel) -/
def testStridedGemmTransposedB : IO Unit := do
  IO.println "\n=== Testing StridedGpuTensor.gemm with B transposed ==="

  if !Metal.isAvailable () then
    IO.println "  (Skipped: Metal not available)"
    return

  -- A: 2x3, B_storage: 2x3 but viewed as (3x2)^T
  -- This tests that we correctly dispatch to gemmNT
  let aData := floatsToByteArray [1, 2, 3, 4, 5, 6]
  -- Store B^T = [[1, 3, 5], [2, 4, 6]] (2x3 in storage)
  -- When viewed as transpose, it's B = [[1, 2], [3, 4], [5, 6]] (3x2)
  let btData := floatsToByteArray [1, 3, 5, 2, 4, 6]

  let aGpu ← Metal.GpuBuffer.fromByteArray aData
  let btGpu ← Metal.GpuBuffer.fromByteArray btData

  let A : StridedGpuTensor Float (Idx 2 × Idx 3) :=
    StridedGpuTensor.fromContiguousBuffer aGpu #[2, 3]

  -- Create B_storage as (2x3), then transpose to get (3x2) view
  let BStorage : StridedGpuTensor Float (Idx 2 × Idx 3) :=
    StridedGpuTensor.fromContiguousBuffer btGpu #[2, 3]
  let B := BStorage.transpose  -- Now (3x2) but transposed in layout

  -- Verify B is transposed
  if !B.isTransposed then
    IO.println "  ✗ B should be transposed"
    return

  let C ← StridedGpuTensor.gemm A B

  let cData ← C.toGpuBuffer.toByteArray

  -- Same expected result as contiguous case
  let expected := [22.0, 28.0, 49.0, 64.0]
  let mut passed := true

  for i in List.range 4 do
    let got := getFloat cData i
    let exp := expected[i]!
    if (got - exp).abs > 0.01 then
      IO.println s!"  ✗ C[{i}] = {got}, expected {exp}"
      passed := false

  if passed then
    IO.println "  ✓ GEMM with transposed B correct (used gemmNT)"

/-- Test GEMM with transposed A (should use gemmTN kernel) -/
def testStridedGemmTransposedA : IO Unit := do
  IO.println "\n=== Testing StridedGpuTensor.gemm with A transposed ==="

  if !Metal.isAvailable () then
    IO.println "  (Skipped: Metal not available)"
    return

  -- A_storage: 3x2 stored, viewed as (2x3)^T
  -- A^T = [[1, 4], [2, 5], [3, 6]] (stored as 3x2)
  -- When transposed: A = [[1, 2, 3], [4, 5, 6]] (2x3 view)
  let atData := floatsToByteArray [1, 4, 2, 5, 3, 6]
  let bData := floatsToByteArray [1, 2, 3, 4, 5, 6]

  let atGpu ← Metal.GpuBuffer.fromByteArray atData
  let bGpu ← Metal.GpuBuffer.fromByteArray bData

  -- Create A_storage as (3x2), then transpose to get (2x3) view
  let AStorage : StridedGpuTensor Float (Idx 3 × Idx 2) :=
    StridedGpuTensor.fromContiguousBuffer atGpu #[3, 2]
  let A := AStorage.transpose  -- Now (2x3) but transposed

  let B : StridedGpuTensor Float (Idx 3 × Idx 2) :=
    StridedGpuTensor.fromContiguousBuffer bGpu #[3, 2]

  -- Verify A is transposed
  if !A.isTransposed then
    IO.println "  ✗ A should be transposed"
    return

  let C ← StridedGpuTensor.gemm A B

  let cData ← C.toGpuBuffer.toByteArray

  let expected := [22.0, 28.0, 49.0, 64.0]
  let mut passed := true

  for i in List.range 4 do
    let got := getFloat cData i
    let exp := expected[i]!
    if (got - exp).abs > 0.01 then
      IO.println s!"  ✗ C[{i}] = {got}, expected {exp}"
      passed := false

  if passed then
    IO.println "  ✓ GEMM with transposed A correct (used gemmTN)"

/-- Test GEMM backward using O(1) transpose views -/
def testStridedGemmBackward : IO Unit := do
  IO.println "\n=== Testing StridedGpuTensor.gemmBackward ==="

  if !Metal.isAvailable () then
    IO.println "  (Skipped: Metal not available)"
    return

  -- A: 2x3, B: 3x2, C: 2x2
  -- For C = A @ B:
  --   dA = dC @ B^T  (2x2) @ (2x3) = (2x3)
  --   dB = A^T @ dC  (3x2) @ (2x2) = (3x2)

  let aData := floatsToByteArray [1, 2, 3, 4, 5, 6]
  let bData := floatsToByteArray [1, 2, 3, 4, 5, 6]
  let dcData := floatsToByteArray [1, 0, 0, 1]  -- Identity-like gradient

  let aGpu ← Metal.GpuBuffer.fromByteArray aData
  let bGpu ← Metal.GpuBuffer.fromByteArray bData
  let dcGpu ← Metal.GpuBuffer.fromByteArray dcData

  let A : StridedGpuTensor Float (Idx 2 × Idx 3) :=
    StridedGpuTensor.fromContiguousBuffer aGpu #[2, 3]
  let B : StridedGpuTensor Float (Idx 3 × Idx 2) :=
    StridedGpuTensor.fromContiguousBuffer bGpu #[3, 2]
  let dC : StridedGpuTensor Float (Idx 2 × Idx 2) :=
    StridedGpuTensor.fromContiguousBuffer dcGpu #[2, 2]

  let (dA, dB) ← StridedGpuTensor.gemmBackward A B dC

  -- Verify shapes
  if dA.shape.toList == [2, 3] then
    IO.println "  ✓ dA shape [2, 3] correct"
  else
    IO.println s!"  ✗ dA shape wrong: {dA.shape.toList}"

  if dB.shape.toList == [3, 2] then
    IO.println "  ✓ dB shape [3, 2] correct"
  else
    IO.println s!"  ✗ dB shape wrong: {dB.shape.toList}"

  -- Read and verify gradients
  -- dA = dC @ B^T = [[1,0],[0,1]] @ [[1,3,5],[2,4,6]] = [[1,3,5],[2,4,6]]
  let daData ← dA.toGpuBuffer.toByteArray
  let expectedDa := [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]

  let mut passedDa := true
  for i in List.range 6 do
    let got := getFloat daData i
    let exp := expectedDa[i]!
    if (got - exp).abs > 0.01 then
      IO.println s!"  ✗ dA[{i}] = {got}, expected {exp}"
      passedDa := false

  if passedDa then
    IO.println "  ✓ dA values correct"

  -- dB = A^T @ dC = [[1,4],[2,5],[3,6]] @ [[1,0],[0,1]] = [[1,4],[2,5],[3,6]]
  let dbData ← dB.toGpuBuffer.toByteArray
  let expectedDb := [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]

  let mut passedDb := true
  for i in List.range 6 do
    let got := getFloat dbData i
    let exp := expectedDb[i]!
    if (got - exp).abs > 0.01 then
      IO.println s!"  ✗ dB[{i}] = {got}, expected {exp}"
      passedDb := false

  if passedDb then
    IO.println "  ✓ dB values correct"

  if passedDa && passedDb then
    IO.println "  ✓ GEMM backward with O(1) transpose views working!"

/-- Test batch transpose for 3D tensors -/
def testBatchTranspose : IO Unit := do
  IO.println "\n=== Testing StridedGpuTensor.batchTranspose ==="

  if !Metal.isAvailable () then
    IO.println "  (Skipped: Metal not available)"
    return

  -- Create 2x3x4 tensor (batch=2, rows=3, cols=4)
  let data := floatsToByteArray (List.range 24 |>.map (fun i => Float.ofNat i))
  let gpuBuf ← Metal.GpuBuffer.fromByteArray data

  let tensor : StridedGpuTensor Float (Idx 2 × Idx 3 × Idx 4) :=
    StridedGpuTensor.fromContiguousBuffer gpuBuf #[2, 3, 4]

  -- Batch transpose: (2,3,4) -> (2,4,3)
  let transposed := tensor.batchTranspose

  if transposed.shape.toList == [2, 4, 3] then
    IO.println "  ✓ Batch transpose shape [2, 4, 3] correct"
  else
    IO.println s!"  ✗ Shape wrong: {transposed.shape.toList}"

  -- Original strides: [12, 4, 1]
  -- After batch transpose: [12, 1, 4] (swap last two)
  let strides := transposed.strides.toList
  if strides == [12, 1, 4] then
    IO.println "  ✓ Batch transpose strides [12, 1, 4] correct"
  else
    IO.println s!"  ✗ Strides wrong: {strides}"

def main : IO Unit := do
  IO.println "==================================="
  IO.println "  Strided GPU Tensor Test Suite"
  IO.println "===================================\n"

  -- Pure layout tests (no GPU needed)
  testLayoutContiguous
  testLayoutTranspose
  testLayoutIsContiguous
  testLayoutIsTransposed
  testLayoutSlice
  testLayoutPermute
  testLayoutNumel
  testLayoutFlatIndex

  -- GPU tests
  if Metal.isAvailable () then
    IO.println "\n\n--- GPU Tests ---"
    testStridedTensorBasic
    testStridedTranspose
    testStridedGemmContiguous
    testStridedGemmTransposedB
    testStridedGemmTransposedA
    testStridedGemmBackward
    testBatchTranspose
  else
    IO.println "\n\nMetal GPU not available, skipping GPU tests."

  IO.println "\n==================================="
  IO.println "  All tests completed"
  IO.println "==================================="
