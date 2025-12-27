-- Verification tests with known mathematical formulas
-- Results written to file to ensure correctness

import LeanBLAS

open BLAS.CBLAS

-- Sum of 1..n = n*(n+1)/2
def expectedSum (n : Nat) : Float := n.toFloat * (n.toFloat + 1) / 2

-- Sum of squares 1^2 + 2^2 + ... + n^2 = n*(n+1)*(2n+1)/6
def expectedSumSquares (n : Nat) : Float :=
  n.toFloat * (n.toFloat + 1) * (2 * n.toFloat + 1) / 6

-- Helper to compute relative error
def relError (expected actual : Float) : Float :=
  if expected == 0 then actual.abs else ((actual - expected) / expected).abs

-- Test harness
structure TestResult where
  name : String
  expected : Float
  actual : Float
  relErr : Float
  passed : Bool
  deriving Repr

def runTest (name : String) (expected : Float) (actual : Float) (tol : Float := 1e-6) : TestResult :=
  let err := relError expected actual
  { name, expected, actual, relErr := err, passed := err < tol }

def formatResult (r : TestResult) : String :=
  let status := if r.passed then "PASS" else "FAIL"
  s!"[{status}] {r.name}: expected={r.expected}, actual={r.actual}, relErr={r.relErr}"

-- Create Float64Array from list of floats
def mkFloat64Array (xs : List Float) : BLAS.Float64Array :=
  (FloatArray.mk (xs.toArray)).toFloat64Array

def main : IO Unit := do
  let mut results : Array TestResult := #[]
  let mut output := "=== SciLean Verification Tests ===\n"
  output := output ++ s!"Timestamp: {← IO.monoNanosNow}\n\n"

  -- Test 1: Sum using dsum (1..1_000_000)
  output := output ++ "--- Test 1: BLAS dsum of 1..1,000,000 ---\n"
  let n1 : Nat := 1_000_000
  let arr1 := mkFloat64Array ((List.range n1).map fun i => (i + 1).toFloat)
  let sum1 := dsum n1.toUSize arr1 0 1
  let exp1 := expectedSum n1
  let r1 := runTest "dsum_1M" exp1 sum1
  results := results.push r1
  output := output ++ formatResult r1 ++ "\n"
  output := output ++ s!"  Formula: n*(n+1)/2 = {n1}*{n1+1}/2 = {exp1}\n\n"

  -- Test 2: Dot product equals sum of squares
  output := output ++ "--- Test 2: BLAS ddot (sum of squares) ---\n"
  let n2 := 10_000
  let arr2 := mkFloat64Array ((List.range n2).map fun i => (i + 1).toFloat)
  let dot2 := ddot n2.toUSize arr2 0 1 arr2 0 1
  let exp2 := expectedSumSquares n2
  let r2 := runTest "ddot_10K" exp2 dot2
  results := results.push r2
  output := output ++ formatResult r2 ++ "\n"
  output := output ++ s!"  dot(1..n, 1..n) = sum of squares = {exp2}\n\n"

  -- Test 3: Euclidean norm = sqrt(sum of squares)
  output := output ++ "--- Test 3: BLAS dnrm2 ---\n"
  let nrm3 := dnrm2 n2.toUSize arr2 0 1
  let exp3 := (expectedSumSquares n2).sqrt
  let r3 := runTest "dnrm2_10K" exp3 nrm3
  results := results.push r3
  output := output ++ formatResult r3 ++ "\n"
  output := output ++ s!"  ||1..n||_2 = sqrt(sum of squares) = {exp3}\n\n"

  -- Test 4: Sum of absolute values = sum 1..n (all positive)
  output := output ++ "--- Test 4: BLAS dasum ---\n"
  let asum4 := dasum n2.toUSize arr2 0 1
  let exp4 := expectedSum n2
  let r4 := runTest "dasum_10K" exp4 asum4
  results := results.push r4
  output := output ++ formatResult r4 ++ "\n"
  output := output ++ s!"  sum(|1..n|) = n*(n+1)/2 = {exp4}\n\n"

  -- Test 5: daxpy (y = alpha*x + y)
  output := output ++ "--- Test 5: BLAS daxpy ---\n"
  let n5 := 1000
  let x5 := mkFloat64Array ((List.range n5).map fun _ => 2.0)
  let y5 := mkFloat64Array ((List.range n5).map fun _ => 3.0)
  let alpha5 := 5.0
  let y5' := daxpy n5.toUSize alpha5 x5 0 1 y5 0 1
  -- y' should be 5*2 + 3 = 13 for each element
  let sum5 := dsum n5.toUSize y5' 0 1
  let exp5 := 13.0 * n5.toFloat
  let r5 := runTest "daxpy_1K" exp5 sum5
  results := results.push r5
  output := output ++ formatResult r5 ++ "\n"
  output := output ++ s!"  y = 5*[2,2,...] + [3,3,...] = [13,13,...], sum = {exp5}\n\n"

  -- Test 6: dscal (x = alpha*x)
  output := output ++ "--- Test 6: BLAS dscal ---\n"
  let n6 := 1000
  let x6 := mkFloat64Array ((List.range n6).map fun i => (i + 1).toFloat)
  let alpha6 := 2.0
  let x6' := dscal n6.toUSize alpha6 x6 0 1
  let sum6 := dsum n6.toUSize x6' 0 1
  let exp6 := 2.0 * expectedSum n6
  let r6 := runTest "dscal_1K" exp6 sum6
  results := results.push r6
  output := output ++ formatResult r6 ++ "\n"
  output := output ++ s!"  2 * sum(1..{n6}) = 2 * {expectedSum n6} = {exp6}\n\n"

  -- Test 7: Large sum (10M elements)
  output := output ++ "--- Test 7: BLAS dsum of 1..10,000,000 ---\n"
  let n7 : Nat := 10_000_000
  let arr7 := mkFloat64Array ((List.range n7).map fun i => (i + 1).toFloat)
  let sum7 := dsum n7.toUSize arr7 0 1
  let exp7 := expectedSum n7
  let r7 := runTest "dsum_10M" exp7 sum7 1e-5  -- slightly relaxed tolerance
  results := results.push r7
  output := output ++ formatResult r7 ++ "\n"
  output := output ++ s!"  Formula: {n7}*{n7+1}/2 = {exp7}\n\n"

  -- Test 8: dconst creates constant array
  output := output ++ "--- Test 8: BLAS dconst ---\n"
  let n8 := 5000
  let val8 := 7.5
  let arr8 := dconst n8.toUSize val8
  let sum8 := dsum n8.toUSize arr8 0 1
  let exp8 := val8 * n8.toFloat
  let r8 := runTest "dconst_5K" exp8 sum8
  results := results.push r8
  output := output ++ formatResult r8 ++ "\n"
  output := output ++ s!"  sum([7.5, 7.5, ...] * 5000) = {exp8}\n\n"

  -- Summary
  let passed := results.filter (·.passed) |>.size
  let total := results.size
  output := output ++ s!"=== Summary: {passed}/{total} tests passed ===\n"

  -- Write to file
  let outPath := "test_verification_results.txt"
  IO.FS.writeFile outPath output
  IO.println output
  IO.println s!"Results written to {outPath}"

  -- Exit with error if any test failed
  if passed < total then
    IO.Process.exit 1
