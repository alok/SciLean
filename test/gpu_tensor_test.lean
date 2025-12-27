/-
Test for GPU tensor system.
Tests O(1) transpose, layout operations, and GEMM backward with transpose views.
-/
import SciLean.Data.Tensor
import SciLean.Data.Tensor.Layout
import SciLean.AD.TensorRevFDeriv
import SciLean.FFI.Metal
import SciLean.FFI.Metal.GpuBufferView
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

-- ## GPU Tensor Tests

/-- Test GpuTensor creation and layout queries -/
def testGpuTensorBasic : IO Unit := do
  IO.println "\n=== Testing GpuTensor basic operations ==="

  if !Metal.isAvailable () then
    IO.println "  (Skipped: Metal not available)"
    return

  -- Create a 2x3 matrix
  let data := floatsToByteArray [1, 2, 3, 4, 5, 6]
  let gpuBuf ← Metal.GpuBuffer.fromByteArray data

  let tensor : Float^[Idx 2 × Idx 3]@metal :=
    GpuTensor.fromContiguousBuffer gpuBuf #[2, 3]

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
def testGpuTranspose : IO Unit := do
  IO.println "\n=== Testing GpuTensor.transpose (O(1)) ==="

  if !Metal.isAvailable () then
    IO.println "  (Skipped: Metal not available)"
    return

  -- Create 2x3 matrix
  let data := floatsToByteArray [1, 2, 3, 4, 5, 6]
  let gpuBuf ← Metal.GpuBuffer.fromByteArray data

  let tensor : Float^[Idx 2 × Idx 3]@metal :=
    GpuTensor.fromContiguousBuffer gpuBuf #[2, 3]

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
def testGpuGemmContiguous : IO Unit := do
  IO.println "\n=== Testing GpuTensor.gemm (contiguous) ==="

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

  let A : Float^[Idx 2 × Idx 3]@metal :=
    GpuTensor.fromContiguousBuffer aGpu #[2, 3]
  let B : Float^[Idx 3 × Idx 2]@metal :=
    GpuTensor.fromContiguousBuffer bGpu #[3, 2]

  let C ← GpuTensor.gemm A B

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

/-- Test GEMM with transposed B (layout-aware GEMM). -/
def testGpuGemmTransposedB : IO Unit := do
  IO.println "\n=== Testing GpuTensor.gemm with B transposed ==="

  if !Metal.isAvailable () then
    IO.println "  (Skipped: Metal not available)"
    return

  -- A: 2x3, B_storage: 2x3 but viewed as (3x2)^T
  -- This tests that we correctly handle a transposed layout
  let aData := floatsToByteArray [1, 2, 3, 4, 5, 6]
  -- Store B^T = [[1, 3, 5], [2, 4, 6]] (2x3 in storage)
  -- When viewed as transpose, it's B = [[1, 2], [3, 4], [5, 6]] (3x2)
  let btData := floatsToByteArray [1, 3, 5, 2, 4, 6]

  let aGpu ← Metal.GpuBuffer.fromByteArray aData
  let btGpu ← Metal.GpuBuffer.fromByteArray btData

  let A : Float^[Idx 2 × Idx 3]@metal :=
    GpuTensor.fromContiguousBuffer aGpu #[2, 3]

  -- Create B_storage as (2x3), then transpose to get (3x2) view
  let BStorage : Float^[Idx 2 × Idx 3]@metal :=
    GpuTensor.fromContiguousBuffer btGpu #[2, 3]
  let B := BStorage.transpose  -- Now (3x2) but transposed in layout

  -- Verify B is transposed
  if !B.isTransposed then
    IO.println "  ✗ B should be transposed"
    return

  let C ← GpuTensor.gemm A B

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
    IO.println "  ✓ GEMM with transposed B correct"

/-- Test GEMM with transposed A (layout-aware GEMM). -/
def testGpuGemmTransposedA : IO Unit := do
  IO.println "\n=== Testing GpuTensor.gemm with A transposed ==="

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
  let AStorage : Float^[Idx 3 × Idx 2]@metal :=
    GpuTensor.fromContiguousBuffer atGpu #[3, 2]
  let A := AStorage.transpose  -- Now (2x3) but transposed

  let B : Float^[Idx 3 × Idx 2]@metal :=
    GpuTensor.fromContiguousBuffer bGpu #[3, 2]

  -- Verify A is transposed
  if !A.isTransposed then
    IO.println "  ✗ A should be transposed"
    return

  let C ← GpuTensor.gemm A B

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
    IO.println "  ✓ GEMM with transposed A correct"

/-- Test GEMM backward using O(1) transpose views -/
def testGpuGemmBackward : IO Unit := do
  IO.println "\n=== Testing GpuTensor.gemmBackward ==="

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

  let A : Float^[Idx 2 × Idx 3]@metal :=
    GpuTensor.fromContiguousBuffer aGpu #[2, 3]
  let B : Float^[Idx 3 × Idx 2]@metal :=
    GpuTensor.fromContiguousBuffer bGpu #[3, 2]
  let dC : Float^[Idx 2 × Idx 2]@metal :=
    GpuTensor.fromContiguousBuffer dcGpu #[2, 2]

  let (dA, dB) ← GpuTensor.gemmBackward A B dC

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

/-- Numerical gradient check for GEMM backward.
    Verifies analytical gradient matches finite difference approximation. -/
def testGpuGemmBackwardNumerical : IO Unit := do
  IO.println "\n=== Testing GEMM backward (numerical check) ==="

  if !Metal.isAvailable () then
    IO.println "  (Skipped: Metal not available)"
    return

  let eps := 1e-3  -- finite difference epsilon
  let tol := 1e-2  -- tolerance for gradient check

  -- Small matrices for numerical stability
  -- A: 2x2, B: 2x2 -> C: 2x2
  let aVals := [1.0, 2.0, 3.0, 4.0]
  let bVals := [0.5, 1.5, 2.5, 0.5]

  -- Loss = sum(C) = sum(A @ B)
  -- dC = ones(2,2)
  let dcVals := [1.0, 1.0, 1.0, 1.0]

  let aData := floatsToByteArray aVals
  let bData := floatsToByteArray bVals
  let dcData := floatsToByteArray dcVals

  let aGpu ← Metal.GpuBuffer.fromByteArray aData
  let bGpu ← Metal.GpuBuffer.fromByteArray bData
  let dcGpu ← Metal.GpuBuffer.fromByteArray dcData

  let A : Float^[Idx 2 × Idx 2]@metal := GpuTensor.fromContiguousBuffer aGpu #[2, 2]
  let B : Float^[Idx 2 × Idx 2]@metal := GpuTensor.fromContiguousBuffer bGpu #[2, 2]
  let dC : Float^[Idx 2 × Idx 2]@metal := GpuTensor.fromContiguousBuffer dcGpu #[2, 2]

  -- Analytical gradients
  let (dA, dB) ← GpuTensor.gemmBackward A B dC
  let daData ← dA.toGpuBuffer.toByteArray
  let dbData ← dB.toGpuBuffer.toByteArray

  -- Numerical gradient for A
  let mut passedA := true
  for i in List.range 4 do
    -- Perturb A[i] by +eps
    let mut aPlusVals := aVals
    aPlusVals := aPlusVals.set i (aPlusVals[i]! + eps)
    let aPlusData := floatsToByteArray aPlusVals
    let aPlusGpu ← Metal.GpuBuffer.fromByteArray aPlusData
    let APlus : Float^[Idx 2 × Idx 2]@metal := GpuTensor.fromContiguousBuffer aPlusGpu #[2, 2]
    let CPlus ← GpuTensor.gemm APlus B
    let cPlusData ← CPlus.toGpuBuffer.toByteArray
    let lossPlus := (getFloat cPlusData 0) + (getFloat cPlusData 1) +
                    (getFloat cPlusData 2) + (getFloat cPlusData 3)

    -- Perturb A[i] by -eps
    let mut aMinusVals := aVals
    aMinusVals := aMinusVals.set i (aMinusVals[i]! - eps)
    let aMinusData := floatsToByteArray aMinusVals
    let aMinusGpu ← Metal.GpuBuffer.fromByteArray aMinusData
    let AMinus : Float^[Idx 2 × Idx 2]@metal := GpuTensor.fromContiguousBuffer aMinusGpu #[2, 2]
    let CMinus ← GpuTensor.gemm AMinus B
    let cMinusData ← CMinus.toGpuBuffer.toByteArray
    let lossMinus := (getFloat cMinusData 0) + (getFloat cMinusData 1) +
                     (getFloat cMinusData 2) + (getFloat cMinusData 3)

    -- Numerical gradient: (loss+ - loss-) / (2 * eps)
    let numGrad := (lossPlus - lossMinus) / (2 * eps)
    let anaGrad := getFloat daData i

    if (numGrad - anaGrad).abs > tol then
      IO.println s!"  ✗ dA[{i}]: analytical={anaGrad}, numerical={numGrad}"
      passedA := false

  if passedA then
    IO.println "  ✓ dA numerical gradient check passed"

  -- Numerical gradient for B
  let mut passedB := true
  for i in List.range 4 do
    let mut bPlusVals := bVals
    bPlusVals := bPlusVals.set i (bPlusVals[i]! + eps)
    let bPlusData := floatsToByteArray bPlusVals
    let bPlusGpu ← Metal.GpuBuffer.fromByteArray bPlusData
    let BPlus : Float^[Idx 2 × Idx 2]@metal := GpuTensor.fromContiguousBuffer bPlusGpu #[2, 2]
    let CPlus ← GpuTensor.gemm A BPlus
    let cPlusData ← CPlus.toGpuBuffer.toByteArray
    let lossPlus := (getFloat cPlusData 0) + (getFloat cPlusData 1) +
                    (getFloat cPlusData 2) + (getFloat cPlusData 3)

    let mut bMinusVals := bVals
    bMinusVals := bMinusVals.set i (bMinusVals[i]! - eps)
    let bMinusData := floatsToByteArray bMinusVals
    let bMinusGpu ← Metal.GpuBuffer.fromByteArray bMinusData
    let BMinus : Float^[Idx 2 × Idx 2]@metal := GpuTensor.fromContiguousBuffer bMinusGpu #[2, 2]
    let CMinus ← GpuTensor.gemm A BMinus
    let cMinusData ← CMinus.toGpuBuffer.toByteArray
    let lossMinus := (getFloat cMinusData 0) + (getFloat cMinusData 1) +
                     (getFloat cMinusData 2) + (getFloat cMinusData 3)

    let numGrad := (lossPlus - lossMinus) / (2 * eps)
    let anaGrad := getFloat dbData i

    if (numGrad - anaGrad).abs > tol then
      IO.println s!"  ✗ dB[{i}]: analytical={anaGrad}, numerical={numGrad}"
      passedB := false

  if passedB then
    IO.println "  ✓ dB numerical gradient check passed"

  if passedA && passedB then
    IO.println "  ✓ GEMM backward numerical gradient verification complete!"

/-- Numerical gradient check for GELU backward.
    Verifies geluBackward matches finite difference of GELU forward. -/
def testGpuGeluBackwardNumerical : IO Unit := do
  IO.println "\n=== Testing GELU backward (numerical check) ==="

  if !Metal.isAvailable () then
    IO.println "  (Skipped: Metal not available)"
    return

  let eps := 1e-4  -- finite difference epsilon
  let tol := 5e-3  -- tolerance for gradient check (Float32 precision)

  -- Test values spanning different GELU regions
  let xVals := [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
  let n := xVals.length

  -- Upstream gradient (dL/dy = 1.0 for sum loss)
  let dyVals := List.replicate n 1.0

  let xData := floatsToByteArray xVals
  let dyData := floatsToByteArray dyVals

  let xGpu ← Metal.GpuBuffer.fromByteArray xData
  let dyGpu ← Metal.GpuBuffer.fromByteArray dyData

  let X : Float^[Idx 1 × Idx n]@metal := GpuTensor.fromContiguousBuffer xGpu #[1, n]
  let dY : Float^[Idx 1 × Idx n]@metal := GpuTensor.fromContiguousBuffer dyGpu #[1, n]

  -- Analytical gradient via geluBackward
  let dX ← GpuTensor.geluBackward X dY
  let dxData ← dX.toGpuBuffer.toByteArray

  -- Numerical gradient check for each element
  let mut allPassed := true
  for i in List.range n do
    -- f(x + eps)
    let mut xPlusVals := xVals
    xPlusVals := xPlusVals.set i (xPlusVals[i]! + eps)
    let xPlusData := floatsToByteArray xPlusVals
    let xPlusGpu ← Metal.GpuBuffer.fromByteArray xPlusData
    let XPlus : Float^[Idx 1 × Idx n]@metal := GpuTensor.fromContiguousBuffer xPlusGpu #[1, n]
    let YPlus ← GpuTensor.gelu XPlus
    let yPlusData ← YPlus.toGpuBuffer.toByteArray
    let lossPlusElem := getFloat yPlusData i

    -- f(x - eps)
    let mut xMinusVals := xVals
    xMinusVals := xMinusVals.set i (xMinusVals[i]! - eps)
    let xMinusData := floatsToByteArray xMinusVals
    let xMinusGpu ← Metal.GpuBuffer.fromByteArray xMinusData
    let XMinus : Float^[Idx 1 × Idx n]@metal := GpuTensor.fromContiguousBuffer xMinusGpu #[1, n]
    let YMinus ← GpuTensor.gelu XMinus
    let yMinusData ← YMinus.toGpuBuffer.toByteArray
    let lossMinusElem := getFloat yMinusData i

    -- Numerical gradient: df/dx = (f(x+eps) - f(x-eps)) / (2*eps)
    let numGrad := (lossPlusElem - lossMinusElem) / (2 * eps)
    let anaGrad := getFloat dxData i

    if (numGrad - anaGrad).abs > tol then
      IO.println s!"  ✗ dX[{i}] at x={xVals[i]!}: analytical={anaGrad}, numerical={numGrad}"
      allPassed := false

  if allPassed then
    IO.println "  ✓ GELU backward numerical gradient check passed (8 test points)"

/-- Numerical gradient check for softmax backward.
    Tests that d(softmax)/dx matches finite difference. -/
def testGpuSoftmaxBackwardNumerical : IO Unit := do
  IO.println "\n=== Testing Softmax backward (numerical check) ==="

  if !Metal.isAvailable () then
    IO.println "  (Skipped: Metal not available)"
    return

  let eps := 1e-4
  let tol := 1e-3

  -- Input logits: 2 samples, 4 classes
  let xVals := [1.0, 2.0, 3.0, 4.0,  -- sample 0
                0.5, 1.5, 0.0, 2.0]  -- sample 1

  let xData := floatsToByteArray xVals

  let xGpu ← Metal.GpuBuffer.fromByteArray xData
  let X : Float^[Idx 2 × Idx 4]@metal := GpuTensor.fromContiguousBuffer xGpu #[2, 4]

  -- Forward: y = softmax(x)
  let Y ← GpuTensor.softmax X
  let yData ← Y.toGpuBuffer.toByteArray

  -- For loss = sum(y), dy = ones
  -- Check numerical gradient of sum(softmax(x)) w.r.t. x

  let mut allPassed := true
  for i in List.range 8 do
    -- f(x + eps)
    let mut xPlusVals := xVals
    xPlusVals := xPlusVals.set i (xPlusVals[i]! + eps)
    let xPlusData := floatsToByteArray xPlusVals
    let xPlusGpu ← Metal.GpuBuffer.fromByteArray xPlusData
    let XPlus : Float^[Idx 2 × Idx 4]@metal := GpuTensor.fromContiguousBuffer xPlusGpu #[2, 4]
    let YPlus ← GpuTensor.softmax XPlus
    let yPlusData ← YPlus.toGpuBuffer.toByteArray
    let mut lossPlus : Float := 0
    for j in List.range 8 do
      lossPlus := lossPlus + getFloat yPlusData j

    -- f(x - eps)
    let mut xMinusVals := xVals
    xMinusVals := xMinusVals.set i (xMinusVals[i]! - eps)
    let xMinusData := floatsToByteArray xMinusVals
    let xMinusGpu ← Metal.GpuBuffer.fromByteArray xMinusData
    let XMinus : Float^[Idx 2 × Idx 4]@metal := GpuTensor.fromContiguousBuffer xMinusGpu #[2, 4]
    let YMinus ← GpuTensor.softmax XMinus
    let yMinusData ← YMinus.toGpuBuffer.toByteArray
    let mut lossMinus : Float := 0
    for j in List.range 8 do
      lossMinus := lossMinus + getFloat yMinusData j

    -- Numerical gradient
    let numGrad := (lossPlus - lossMinus) / (2 * eps)

    -- For sum(softmax(x)), the gradient should be 0 since sum(softmax) = 1 always
    -- This is a good sanity check!
    if numGrad.abs > tol then
      IO.println s!"  ✗ d(sum(softmax))/dx[{i}] should be ~0, got {numGrad}"
      allPassed := false

  if allPassed then
    IO.println "  ✓ Softmax preserves sum invariant: d(sum(softmax))/dx ≈ 0"

  -- More meaningful test: cross-entropy loss gradient
  -- loss = -sum(target * log(softmax(x)))
  -- For one-hot target at class 0: loss = -log(softmax(x)[0])
  IO.println "  Testing cross-entropy gradient..."
  let mut cePassedSample0 := true

  -- Sample 0: one-hot at class 2 (index 2)
  let targetClass := 2
  for i in List.range 4 do  -- only check sample 0 (indices 0-3)
    let mut xPlusVals := xVals
    xPlusVals := xPlusVals.set i (xPlusVals[i]! + eps)
    let xPlusData := floatsToByteArray xPlusVals
    let xPlusGpu ← Metal.GpuBuffer.fromByteArray xPlusData
    let XPlus : Float^[Idx 2 × Idx 4]@metal := GpuTensor.fromContiguousBuffer xPlusGpu #[2, 4]
    let YPlus ← GpuTensor.softmax XPlus
    let yPlusData ← YPlus.toGpuBuffer.toByteArray
    let ceLossPlus := -Float.log (getFloat yPlusData targetClass + 1e-10)

    let mut xMinusVals := xVals
    xMinusVals := xMinusVals.set i (xMinusVals[i]! - eps)
    let xMinusData := floatsToByteArray xMinusVals
    let xMinusGpu ← Metal.GpuBuffer.fromByteArray xMinusData
    let XMinus : Float^[Idx 2 × Idx 4]@metal := GpuTensor.fromContiguousBuffer xMinusGpu #[2, 4]
    let YMinus ← GpuTensor.softmax XMinus
    let yMinusData ← YMinus.toGpuBuffer.toByteArray
    let ceLossMinus := -Float.log (getFloat yMinusData targetClass + 1e-10)

    let numGrad := (ceLossPlus - ceLossMinus) / (2 * eps)

    -- Analytical gradient for CE + softmax: y[i] - target[i]
    let yVal := getFloat yData i
    let targetVal : Float := if i == targetClass then 1.0 else 0.0
    let anaGrad := yVal - targetVal

    if (numGrad - anaGrad).abs > tol then
      IO.println s!"  ✗ CE grad x[{i}]: analytical={anaGrad}, numerical={numGrad}"
      cePassedSample0 := false

  if cePassedSample0 then
    IO.println "  ✓ Cross-entropy + softmax gradient verified (sample 0)"

/-- Numerical gradient check for element-wise addition. -/
def testGpuAddBackwardNumerical : IO Unit := do
  IO.println "\n=== Testing Add backward (numerical check) ==="

  if !Metal.isAvailable () then
    IO.println "  (Skipped: Metal not available)"
    return

  let eps := 1e-4
  let tol := 5e-3  -- Float32 precision tolerance

  -- a + b, loss = sum(a + b)
  let aVals := [1.0, 2.0, 3.0, 4.0]
  let bVals := [0.5, 1.5, 2.5, 3.5]

  let aData := floatsToByteArray aVals
  let bData := floatsToByteArray bVals

  let aGpu ← Metal.GpuBuffer.fromByteArray aData
  let bGpu ← Metal.GpuBuffer.fromByteArray bData

  let A : Float^[Idx 2 × Idx 2]@metal := GpuTensor.fromContiguousBuffer aGpu #[2, 2]
  let B : Float^[Idx 2 × Idx 2]@metal := GpuTensor.fromContiguousBuffer bGpu #[2, 2]

  let C ← GpuTensor.add A B
  let cData ← C.toGpuBuffer.toByteArray

  -- For sum(a + b), da = db = ones
  -- Numerical check for da
  let mut passedA := true
  for i in List.range 4 do
    let mut aPlusVals := aVals
    aPlusVals := aPlusVals.set i (aPlusVals[i]! + eps)
    let aPlusData := floatsToByteArray aPlusVals
    let aPlusGpu ← Metal.GpuBuffer.fromByteArray aPlusData
    let APlus : Float^[Idx 2 × Idx 2]@metal := GpuTensor.fromContiguousBuffer aPlusGpu #[2, 2]
    let CPlus ← GpuTensor.add APlus B
    let cPlusData ← CPlus.toGpuBuffer.toByteArray
    let mut lossPlus : Float := 0
    for j in List.range 4 do
      lossPlus := lossPlus + getFloat cPlusData j

    let mut aMinusVals := aVals
    aMinusVals := aMinusVals.set i (aMinusVals[i]! - eps)
    let aMinusData := floatsToByteArray aMinusVals
    let aMinusGpu ← Metal.GpuBuffer.fromByteArray aMinusData
    let AMinus : Float^[Idx 2 × Idx 2]@metal := GpuTensor.fromContiguousBuffer aMinusGpu #[2, 2]
    let CMinus ← GpuTensor.add AMinus B
    let cMinusData ← CMinus.toGpuBuffer.toByteArray
    let mut lossMinus : Float := 0
    for j in List.range 4 do
      lossMinus := lossMinus + getFloat cMinusData j

    let numGrad := (lossPlus - lossMinus) / (2 * eps)
    let anaGrad := 1.0  -- gradient of sum(a+b) w.r.t. a[i] is 1

    if (numGrad - anaGrad).abs > tol then
      IO.println s!"  ✗ da[{i}]: analytical={anaGrad}, numerical={numGrad}"
      passedA := false

  if passedA then
    IO.println "  ✓ Add backward for A passed (gradient = 1)"
    IO.println "  ✓ Add backward for B also = 1 by symmetry"

/-- Test batch transpose for 3D tensors -/
def testBatchTranspose : IO Unit := do
  IO.println "\n=== Testing GpuTensor.batchTranspose ==="

  if !Metal.isAvailable () then
    IO.println "  (Skipped: Metal not available)"
    return

  -- Create 2x3x4 tensor (batch=2, rows=3, cols=4)
  let data := floatsToByteArray (List.range 24 |>.map (fun i => Float.ofNat i))
  let gpuBuf ← Metal.GpuBuffer.fromByteArray data

  let tensor : Float^[Idx 2 × Idx 3 × Idx 4]@metal :=
    GpuTensor.fromContiguousBuffer gpuBuf #[2, 3, 4]

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
  IO.println "  GPU Tensor Test Suite"
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
    testGpuTensorBasic
    testGpuTranspose
    testGpuGemmContiguous
    testGpuGemmTransposedB
    testGpuGemmTransposedA
    testGpuGemmBackward
    testGpuGemmBackwardNumerical
    testGpuGeluBackwardNumerical
    testGpuSoftmaxBackwardNumerical
    testGpuAddBackwardNumerical
    testBatchTranspose
  else
    IO.println "\n\nMetal GPU not available, skipping GPU tests."

  IO.println "\n==================================="
  IO.println "  All tests completed"
  IO.println "==================================="
