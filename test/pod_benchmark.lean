import SciLean.Data.DataArray.PlainDataType

/-!
# PlainDataType Specialization Benchmark

Compares raw FFI access vs PlainDataType abstraction to verify specialization.
-/

open SciLean

-- ============================================================================
-- Benchmark utilities
-- ============================================================================

-- Use a mutable ref to prevent optimization
@[inline]
def runBenchFloat32 (name : String) (iters : Nat) (f : Unit → Float32) : IO Unit := do
  -- Warmup
  let mut acc : Float32 := 0.0
  for _ in [0:10] do acc := acc + f ()
  -- Timed run
  let start ← IO.monoMsNow
  for _ in [0:iters] do acc := acc + f ()
  let elapsed := (← IO.monoMsNow) - start
  let opsPerSec := (iters.toFloat * 1000.0) / elapsed.toFloat
  -- Use acc to prevent DCE
  if acc.isNaN then IO.println "NaN!"
  IO.println s!"  {name}: {elapsed}ms ({opsPerSec / 1e6} M ops/s)"

@[inline]
def runBenchUInt32 (name : String) (iters : Nat) (f : Unit → UInt32) : IO Unit := do
  -- Warmup
  let mut acc : UInt32 := 0
  for _ in [0:10] do acc := acc + f ()
  -- Timed run
  let start ← IO.monoMsNow
  for _ in [0:iters] do acc := acc + f ()
  let elapsed := (← IO.monoMsNow) - start
  let opsPerSec := (iters.toFloat * 1000.0) / elapsed.toFloat
  -- Use acc to prevent DCE
  if acc == 0 then IO.println "Zero!"
  IO.println s!"  {name}: {elapsed}ms ({opsPerSec / 1e6} M ops/s)"

@[inline]
def runBenchUInt64 (name : String) (iters : Nat) (f : Unit → UInt64) : IO Unit := do
  -- Warmup
  let mut acc : UInt64 := 0
  for _ in [0:10] do acc := acc + f ()
  -- Timed run
  let start ← IO.monoMsNow
  for _ in [0:iters] do acc := acc + f ()
  let elapsed := (← IO.monoMsNow) - start
  let opsPerSec := (iters.toFloat * 1000.0) / elapsed.toFloat
  -- Use acc to prevent DCE
  if acc == 0 then IO.println "Zero!"
  IO.println s!"  {name}: {elapsed}ms ({opsPerSec / 1e6} M ops/s)"

-- ============================================================================
-- Float32 benchmarks
-- ============================================================================

/-- Raw FFI Float32 sum - baseline -/
@[specialize, inline]
def rawFloat32Sum (arr : ByteArray) (n : Nat) : Float32 := Id.run do
  let mut sum : Float32 := 0.0
  for i in [0:n] do
    sum := sum + arr.ugetFloat32 (i * 4).toUSize
  return sum

/-- PlainDataType Float32 sum - should specialize to same code -/
@[specialize, inline]
def podFloat32Sum (arr : ByteArray) (n : Nat) : Float32 := Id.run do
  let pd : PlainDataType Float32 := inferInstance
  let mut sum : Float32 := 0.0
  for i in [0:n] do
    sum := sum + pd.btype.fromByteArray arr (i * 4).toUSize sorry_proof
  return sum

-- ============================================================================
-- UInt32 benchmarks
-- ============================================================================

/-- Raw FFI UInt32 sum - baseline -/
@[specialize, inline]
def rawUInt32Sum (arr : ByteArray) (n : Nat) : UInt32 := Id.run do
  let mut sum : UInt32 := 0
  for i in [0:n] do
    sum := sum + arr.ugetUInt32 (i * 4).toUSize
  return sum

/-- PlainDataType UInt32 sum - should specialize to same code -/
@[specialize, inline]
def podUInt32Sum (arr : ByteArray) (n : Nat) : UInt32 := Id.run do
  let pd : PlainDataType UInt32 := inferInstance
  let mut sum : UInt32 := 0
  for i in [0:n] do
    sum := sum + pd.btype.fromByteArray arr (i * 4).toUSize sorry_proof
  return sum

-- ============================================================================
-- UInt64 benchmarks
-- ============================================================================

/-- Raw FFI UInt64 sum - baseline -/
@[specialize, inline]
def rawUInt64Sum (arr : ByteArray) (n : Nat) : UInt64 := Id.run do
  let mut sum : UInt64 := 0
  for i in [0:n] do
    sum := sum + arr.ugetUInt64 (i * 8).toUSize
  return sum

/-- PlainDataType UInt64 sum - should specialize to same code -/
@[specialize, inline]
def podUInt64Sum (arr : ByteArray) (n : Nat) : UInt64 := Id.run do
  let pd : PlainDataType UInt64 := inferInstance
  let mut sum : UInt64 := 0
  for i in [0:n] do
    sum := sum + pd.btype.fromByteArray arr (i * 8).toUSize sorry_proof
  return sum

-- ============================================================================
-- Main benchmark
-- ============================================================================

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════════════════════╗"
  IO.println "║  PlainDataType Specialization Benchmark                       ║"
  IO.println "║  Comparing raw FFI vs PlainDataType abstraction               ║"
  IO.println "╚══════════════════════════════════════════════════════════════╝"

  let n := 1000000  -- 1M elements
  let iters := 100

  -- Create test arrays
  let f32Arr := ByteArray.replicateFloat32 n 1.5
  let u32Arr := ByteArray.replicateUInt32 n 42
  let u64Arr := ByteArray.replicateUInt64 n 12345

  IO.println s!"\nArray size: {n} elements, {iters} iterations each"

  IO.println "\n═══ Float32 (4 bytes) ═══"
  runBenchFloat32 "Raw FFI " iters (fun _ => rawFloat32Sum f32Arr n)
  runBenchFloat32 "POD     " iters (fun _ => podFloat32Sum f32Arr n)

  IO.println "\n═══ UInt32 (4 bytes) ═══"
  runBenchUInt32 "Raw FFI " iters (fun _ => rawUInt32Sum u32Arr n)
  runBenchUInt32 "POD     " iters (fun _ => podUInt32Sum u32Arr n)

  IO.println "\n═══ UInt64 (8 bytes) ═══"
  runBenchUInt64 "Raw FFI " iters (fun _ => rawUInt64Sum u64Arr n)
  runBenchUInt64 "POD     " iters (fun _ => podUInt64Sum u64Arr n)

  -- Verify correctness
  IO.println "\n═══ Correctness Verification ═══"
  let f32Raw := rawFloat32Sum f32Arr n
  let f32Pod := podFloat32Sum f32Arr n
  IO.println s!"Float32: raw={f32Raw}, pod={f32Pod}, match={f32Raw == f32Pod}"

  let u32Raw := rawUInt32Sum u32Arr n
  let u32Pod := podUInt32Sum u32Arr n
  IO.println s!"UInt32:  raw={u32Raw}, pod={u32Pod}, match={u32Raw == u32Pod}"

  let u64Raw := rawUInt64Sum u64Arr n
  let u64Pod := podUInt64Sum u64Arr n
  IO.println s!"UInt64:  raw={u64Raw}, pod={u64Pod}, match={u64Raw == u64Pod}"

  IO.println "\n✓ Done! If POD matches Raw FFI performance, specialization works."
