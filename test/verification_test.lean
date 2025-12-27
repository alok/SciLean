-- Verification tests with known mathematical formulas
-- Benchmarks with W&B logging

import LeanBLAS
import SciLean.Util.Benchmark

open BLAS.CBLAS
open SciLean.Benchmark

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
  timeNs : Nat := 0
  deriving Repr

def runTest (name : String) (expected : Float) (actual : Float) (tol : Float := 1e-6) (timeNs : Nat := 0) : TestResult :=
  let err := relError expected actual
  { name, expected, actual, relErr := err, passed := err < tol, timeNs }

def formatResult (r : TestResult) : String :=
  let status := if r.passed then "PASS" else "FAIL"
  let timeStr := if r.timeNs > 0 then s!" ({formatTime r.timeNs})" else ""
  s!"[{status}] {r.name}: expected={r.expected}, actual={r.actual}, relErr={r.relErr}{timeStr}"

-- Create Float64Array from list of floats
def mkFloat64Array (xs : List Float) : BLAS.Float64Array :=
  (FloatArray.mk (xs.toArray)).toFloat64Array

-- Timed operation that returns (result, time_ns)
def timed (f : Unit → α) : IO (α × Nat) := do
  let start ← IO.monoNanosNow
  let result := f ()
  let elapsed := (← IO.monoNanosNow) - start
  pure (result, elapsed)

-- Benchmark config for BLAS ops
def blasBenchConfig : Config := { warmupIterations := 3, timedIterations := 10 }

def main : IO Unit := do
  IO.println "=== SciLean BLAS Verification & Benchmarks ==="
  IO.println s!"Timestamp: {← IO.monoNanosNow}"
  IO.println ""

  let mut results : Array TestResult := #[]
  let mut output := "=== SciLean Verification Tests ===\n"
  output := output ++ s!"Timestamp: {← IO.monoNanosNow}\n\n"

  -- Prepare test arrays upfront
  let n1M : Nat := 1_000_000
  let n10M : Nat := 10_000_000
  let n10K : Nat := 10_000
  let n1K : Nat := 1000

  IO.println "Preparing test arrays..."
  let arr1M := mkFloat64Array ((List.range n1M).map fun i => (i + 1).toFloat)
  let arr10M := mkFloat64Array ((List.range n10M).map fun i => (i + 1).toFloat)
  let arr10K := mkFloat64Array ((List.range n10K).map fun i => (i + 1).toFloat)
  IO.println "Arrays prepared.\n"

  -- ============================================
  -- Test 1: dsum 1M elements
  -- ============================================
  IO.println "--- Benchmark 1: BLAS dsum (1M elements) ---"
  let exp1 := expectedSum n1M
  let bench1 ← run "dsum_1M" blasBenchConfig fun () => pure (dsum n1M.toUSize arr1M 0 1)
  let sum1 := dsum n1M.toUSize arr1M 0 1
  let r1 := runTest "dsum_1M" exp1 sum1 1e-10 bench1.avgTimeNs
  results := results.push r1
  IO.println (formatResult r1)
  logMetric "BLAS" "dsum_1M" (bench1.avgTimeNs.toFloat / 1e6) (unit? := some "ms")
    (params := [paramNat "n" n1M, paramFloat "expected" exp1, paramFloat "actual" sum1])
  IO.println ""

  -- ============================================
  -- Test 2: dsum 10M elements
  -- ============================================
  IO.println "--- Benchmark 2: BLAS dsum (10M elements) ---"
  let exp2 := expectedSum n10M
  let bench2 ← run "dsum_10M" blasBenchConfig fun () => pure (dsum n10M.toUSize arr10M 0 1)
  let sum2 := dsum n10M.toUSize arr10M 0 1
  let r2 := runTest "dsum_10M" exp2 sum2 1e-5 bench2.avgTimeNs
  results := results.push r2
  IO.println (formatResult r2)
  logMetric "BLAS" "dsum_10M" (bench2.avgTimeNs.toFloat / 1e6) (unit? := some "ms")
    (params := [paramNat "n" n10M, paramFloat "expected" exp2, paramFloat "actual" sum2])
  IO.println ""

  -- ============================================
  -- Test 3: ddot (sum of squares)
  -- ============================================
  IO.println "--- Benchmark 3: BLAS ddot (10K elements) ---"
  let exp3 := expectedSumSquares n10K
  let bench3 ← run "ddot_10K" blasBenchConfig fun () => pure (ddot n10K.toUSize arr10K 0 1 arr10K 0 1)
  let dot3 := ddot n10K.toUSize arr10K 0 1 arr10K 0 1
  let r3 := runTest "ddot_10K" exp3 dot3 1e-10 bench3.avgTimeNs
  results := results.push r3
  IO.println (formatResult r3)
  logMetric "BLAS" "ddot_10K" (bench3.avgTimeNs.toFloat / 1e6) (unit? := some "ms")
    (params := [paramNat "n" n10K, paramFloat "expected" exp3, paramFloat "actual" dot3])
  IO.println ""

  -- ============================================
  -- Test 4: dnrm2
  -- ============================================
  IO.println "--- Benchmark 4: BLAS dnrm2 (10K elements) ---"
  let exp4 := (expectedSumSquares n10K).sqrt
  let bench4 ← run "dnrm2_10K" blasBenchConfig fun () => pure (dnrm2 n10K.toUSize arr10K 0 1)
  let nrm4 := dnrm2 n10K.toUSize arr10K 0 1
  let r4 := runTest "dnrm2_10K" exp4 nrm4 1e-10 bench4.avgTimeNs
  results := results.push r4
  IO.println (formatResult r4)
  logMetric "BLAS" "dnrm2_10K" (bench4.avgTimeNs.toFloat / 1e6) (unit? := some "ms")
    (params := [paramNat "n" n10K, paramFloat "expected" exp4, paramFloat "actual" nrm4])
  IO.println ""

  -- ============================================
  -- Test 5: dasum
  -- ============================================
  IO.println "--- Benchmark 5: BLAS dasum (10K elements) ---"
  let exp5 := expectedSum n10K
  let bench5 ← run "dasum_10K" blasBenchConfig fun () => pure (dasum n10K.toUSize arr10K 0 1)
  let asum5 := dasum n10K.toUSize arr10K 0 1
  let r5 := runTest "dasum_10K" exp5 asum5 1e-10 bench5.avgTimeNs
  results := results.push r5
  IO.println (formatResult r5)
  logMetric "BLAS" "dasum_10K" (bench5.avgTimeNs.toFloat / 1e6) (unit? := some "ms")
    (params := [paramNat "n" n10K, paramFloat "expected" exp5, paramFloat "actual" asum5])
  IO.println ""

  -- ============================================
  -- Test 6: daxpy
  -- ============================================
  IO.println "--- Benchmark 6: BLAS daxpy (1K elements) ---"
  let x6 := mkFloat64Array ((List.range n1K).map fun _ => 2.0)
  let y6 := mkFloat64Array ((List.range n1K).map fun _ => 3.0)
  let alpha6 := 5.0
  let exp6 := 13.0 * n1K.toFloat  -- 5*2 + 3 = 13 per element
  let bench6 ← run "daxpy_1K" blasBenchConfig fun () => pure (
    let y' := daxpy n1K.toUSize alpha6 x6 0 1 y6 0 1
    dsum n1K.toUSize y' 0 1)
  let y6' := daxpy n1K.toUSize alpha6 x6 0 1 y6 0 1
  let sum6 := dsum n1K.toUSize y6' 0 1
  let r6 := runTest "daxpy_1K" exp6 sum6 1e-10 bench6.avgTimeNs
  results := results.push r6
  IO.println (formatResult r6)
  logMetric "BLAS" "daxpy_1K" (bench6.avgTimeNs.toFloat / 1e6) (unit? := some "ms")
    (params := [paramNat "n" n1K, paramFloat "alpha" alpha6])
  IO.println ""

  -- ============================================
  -- Test 7: dscal
  -- ============================================
  IO.println "--- Benchmark 7: BLAS dscal (1K elements) ---"
  let x7 := mkFloat64Array ((List.range n1K).map fun i => (i + 1).toFloat)
  let alpha7 := 2.0
  let exp7 := 2.0 * expectedSum n1K
  let bench7 ← run "dscal_1K" blasBenchConfig fun () => pure (
    let x' := dscal n1K.toUSize alpha7 x7 0 1
    dsum n1K.toUSize x' 0 1)
  let x7' := dscal n1K.toUSize alpha7 x7 0 1
  let sum7 := dsum n1K.toUSize x7' 0 1
  let r7 := runTest "dscal_1K" exp7 sum7 1e-10 bench7.avgTimeNs
  results := results.push r7
  IO.println (formatResult r7)
  logMetric "BLAS" "dscal_1K" (bench7.avgTimeNs.toFloat / 1e6) (unit? := some "ms")
    (params := [paramNat "n" n1K, paramFloat "alpha" alpha7])
  IO.println ""

  -- ============================================
  -- Test 8: dconst
  -- ============================================
  IO.println "--- Benchmark 8: BLAS dconst (5K elements) ---"
  let n8 := 5000
  let val8 := 7.5
  let exp8 := val8 * n8.toFloat
  let bench8 ← run "dconst_5K" blasBenchConfig fun () => pure (
    let arr := dconst n8.toUSize val8
    dsum n8.toUSize arr 0 1)
  let arr8 := dconst n8.toUSize val8
  let sum8 := dsum n8.toUSize arr8 0 1
  let r8 := runTest "dconst_5K" exp8 sum8 1e-10 bench8.avgTimeNs
  results := results.push r8
  IO.println (formatResult r8)
  logMetric "BLAS" "dconst_5K" (bench8.avgTimeNs.toFloat / 1e6) (unit? := some "ms")
    (params := [paramNat "n" n8, paramFloat "value" val8])
  IO.println ""

  -- ============================================
  -- Test 9: Large ddot (1M elements)
  -- ============================================
  IO.println "--- Benchmark 9: BLAS ddot (1M elements) ---"
  let exp9 := expectedSumSquares n1M
  let bench9 ← run "ddot_1M" blasBenchConfig fun () => pure (ddot n1M.toUSize arr1M 0 1 arr1M 0 1)
  let dot9 := ddot n1M.toUSize arr1M 0 1 arr1M 0 1
  let r9 := runTest "ddot_1M" exp9 dot9 1e-6 bench9.avgTimeNs
  results := results.push r9
  IO.println (formatResult r9)
  logMetric "BLAS" "ddot_1M" (bench9.avgTimeNs.toFloat / 1e6) (unit? := some "ms")
    (params := [paramNat "n" n1M, paramFloat "expected" exp9])
  IO.println ""

  -- ============================================
  -- Summary
  -- ============================================
  let passed := results.filter (·.passed) |>.size
  let total := results.size

  IO.println "============================================"
  IO.println s!"Summary: {passed}/{total} tests passed"
  IO.println "============================================"

  -- Print benchmark table
  IO.println "\nBenchmark Summary:"
  IO.println "┌────────────────────┬────────────────┬────────────────┐"
  IO.println "│ Test               │ Time           │ Throughput     │"
  IO.println "├────────────────────┼────────────────┼────────────────┤"
  for r in results do
    let name := padRight r.name 18
    let time := padRight (formatTime r.timeNs) 14
    let status := if r.passed then "✓" else "✗"
    IO.println s!"│ {name} │ {time} │ {status}              │"
  IO.println "└────────────────────┴────────────────┴────────────────┘"

  -- Write to file
  output := output ++ "\nBenchmark Results:\n"
  for r in results do
    output := output ++ formatResult r ++ "\n"
  output := output ++ s!"\n=== Summary: {passed}/{total} tests passed ===\n"

  let outPath := "test_verification_results.txt"
  IO.FS.writeFile outPath output
  IO.println s!"\nResults written to {outPath}"

  -- Log summary to W&B
  logMetric "summary" "tests_passed" passed.toFloat
  logMetric "summary" "tests_total" total.toFloat
  logMetric "summary" "pass_rate" (passed.toFloat / total.toFloat * 100) (unit? := some "%")

  -- Exit with error if any test failed
  if passed < total then
    IO.Process.exit 1
