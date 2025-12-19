import SciLean.Data.DataArray.DerivePlainDataType

-- Benchmark: Struct vs Tuple PlainDataType Performance
-- Compares raw tuples vs struct with deriving PlainDataType
-- Goal: Verify struct wrapper has zero overhead vs raw tuples

open SciLean

-- ============================================================================
-- Type definitions
-- ============================================================================

-- Struct approach (user-friendly) - derived
structure Vec3 where
  x : Float32
  y : Float32
  z : Float32
  deriving PlainDataType, Repr, Inhabited

structure Vec4 where
  x : Float32
  y : Float32
  z : Float32
  w : Float32
  deriving PlainDataType, Repr, Inhabited

-- Manually inlined Vec3 instance for comparison
structure Vec3Manual where
  x : Float32
  y : Float32
  z : Float32
  deriving Repr, Inhabited

-- Write a fully inlined ByteType for Vec3Manual
@[inline]
def Vec3Manual.byteType : ByteType Vec3Manual where
  bytes := 12
  h_size := sorry_proof
  fromByteArray arr i _ :=
    let x := arr.ugetFloat32 i
    let y := arr.ugetFloat32 (i + 4)
    let z := arr.ugetFloat32 (i + 8)
    ⟨x, y, z⟩
  toByteArray arr i _ v :=
    let arr := arr.usetFloat32 i v.x
    let arr := arr.usetFloat32 (i + 4) v.y
    arr.usetFloat32 (i + 8) v.z
  toByteArray_size := sorry_proof
  fromByteArray_toByteArray := sorry_proof
  fromByteArray_toByteArray_other := sorry_proof

instance : PlainDataType Vec3Manual where
  btype := Vec3Manual.byteType

-- Tuple approach (raw products)
abbrev Tuple3 := Float32 × Float32 × Float32
abbrev Tuple4 := Float32 × Float32 × Float32 × Float32

-- ============================================================================
-- Benchmark utilities
-- ============================================================================

@[inline]
def runBench (name : String) (iters : Nat) (f : Unit → Float32) : IO Unit := do
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

-- ============================================================================
-- Vec3 vs Tuple3 benchmarks
-- ============================================================================

-- Serialize array of Vec3
@[specialize, inline]
def serializeVec3Array (arr : Array Vec3) : ByteArray := Id.run do
  let pd : PlainDataType Vec3 := inferInstance
  let elemSize := pd.btype.bytes.toNat
  let mut bytes := ByteArray.replicate (arr.size * elemSize) 0
  for i in [0:arr.size] do
    bytes := pd.btype.toByteArray bytes (i * elemSize).toUSize sorry_proof arr[i]!
  return bytes

-- Deserialize array of Vec3
@[specialize, inline]
def deserializeVec3Array (bytes : ByteArray) (n : Nat) : Array Vec3 := Id.run do
  let pd : PlainDataType Vec3 := inferInstance
  let elemSize := pd.btype.bytes.toNat
  let mut arr : Array Vec3 := #[]
  for i in [0:n] do
    let elem := pd.btype.fromByteArray bytes (i * elemSize).toUSize sorry_proof
    arr := arr.push elem
  return arr

-- Serialize array of Tuple3
@[specialize, inline]
def serializeTuple3Array (arr : Array Tuple3) : ByteArray := Id.run do
  let pd : PlainDataType Tuple3 := inferInstance
  let elemSize := pd.btype.bytes.toNat
  let mut bytes := ByteArray.replicate (arr.size * elemSize) 0
  for i in [0:arr.size] do
    bytes := pd.btype.toByteArray bytes (i * elemSize).toUSize sorry_proof arr[i]!
  return bytes

-- Deserialize array of Tuple3
@[specialize, inline]
def deserializeTuple3Array (bytes : ByteArray) (n : Nat) : Array Tuple3 := Id.run do
  let pd : PlainDataType Tuple3 := inferInstance
  let elemSize := pd.btype.bytes.toNat
  let mut arr : Array Tuple3 := #[]
  for i in [0:n] do
    let elem := pd.btype.fromByteArray bytes (i * elemSize).toUSize sorry_proof
    arr := arr.push elem
  return arr

-- Sum all x components (Vec3 - derived)
@[specialize, inline]
def sumVec3X (bytes : ByteArray) (n : Nat) : Float32 := Id.run do
  let pd : PlainDataType Vec3 := inferInstance
  let elemSize := pd.btype.bytes.toNat
  let mut sum : Float32 := 0.0
  for i in [0:n] do
    let v := pd.btype.fromByteArray bytes (i * elemSize).toUSize sorry_proof
    sum := sum + v.x
  return sum

-- Sum all x components (Vec3Manual - manually inlined)
@[specialize, inline]
def sumVec3ManualX (bytes : ByteArray) (n : Nat) : Float32 := Id.run do
  let pd : PlainDataType Vec3Manual := inferInstance
  let elemSize := pd.btype.bytes.toNat
  let mut sum : Float32 := 0.0
  for i in [0:n] do
    let v := pd.btype.fromByteArray bytes (i * elemSize).toUSize sorry_proof
    sum := sum + v.x
  return sum

-- Sum all first components (Tuple3)
@[specialize, inline]
def sumTuple3First (bytes : ByteArray) (n : Nat) : Float32 := Id.run do
  let pd : PlainDataType Tuple3 := inferInstance
  let elemSize := pd.btype.bytes.toNat
  let mut sum : Float32 := 0.0
  for i in [0:n] do
    let t := pd.btype.fromByteArray bytes (i * elemSize).toUSize sorry_proof
    sum := sum + t.1
  return sum

-- ============================================================================
-- Vec4 vs Tuple4 benchmarks
-- ============================================================================

@[specialize, inline]
def sumVec4X (bytes : ByteArray) (n : Nat) : Float32 := Id.run do
  let pd : PlainDataType Vec4 := inferInstance
  let elemSize := pd.btype.bytes.toNat
  let mut sum : Float32 := 0.0
  for i in [0:n] do
    let v := pd.btype.fromByteArray bytes (i * elemSize).toUSize sorry_proof
    sum := sum + v.x
  return sum

@[specialize, inline]
def sumTuple4First (bytes : ByteArray) (n : Nat) : Float32 := Id.run do
  let pd : PlainDataType Tuple4 := inferInstance
  let elemSize := pd.btype.bytes.toNat
  let mut sum : Float32 := 0.0
  for i in [0:n] do
    let t := pd.btype.fromByteArray bytes (i * elemSize).toUSize sorry_proof
    sum := sum + t.1
  return sum

-- ============================================================================
-- Main benchmark
-- ============================================================================

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════════════════════╗"
  IO.println "║     Struct vs Tuple PlainDataType Benchmark                   ║"
  IO.println "╚══════════════════════════════════════════════════════════════╝"

  let n := 100000  -- 100K elements
  let iters := 100

  -- Create test data
  let vec3Arr : Array Vec3 := Array.range n |>.map fun i =>
    let f := i.toFloat.toFloat32
    ⟨f, f + 1, f + 2⟩

  let tuple3Arr : Array Tuple3 := Array.range n |>.map fun i =>
    let f := i.toFloat.toFloat32
    (f, f + 1, f + 2)

  let vec4Arr : Array Vec4 := Array.range n |>.map fun i =>
    let f := i.toFloat.toFloat32
    ⟨f, f + 1, f + 2, f + 3⟩

  let tuple4Arr : Array Tuple4 := Array.range n |>.map fun i =>
    let f := i.toFloat.toFloat32
    (f, f + 1, f + 2, f + 3)

  IO.println s!"\nArray size: {n} elements, {iters} iterations each"

  -- Verify byte sizes
  let pdVec3 : PlainDataType Vec3 := inferInstance
  let pdTuple3 : PlainDataType Tuple3 := inferInstance
  let pdVec4 : PlainDataType Vec4 := inferInstance
  let pdTuple4 : PlainDataType Tuple4 := inferInstance

  IO.println s!"\nByte sizes:"
  IO.println s!"  Vec3:   {pdVec3.btype.bytes} bytes"
  IO.println s!"  Tuple3: {pdTuple3.btype.bytes} bytes"
  IO.println s!"  Vec4:   {pdVec4.btype.bytes} bytes"
  IO.println s!"  Tuple4: {pdTuple4.btype.bytes} bytes"

  -- Serialize once
  let vec3Bytes := serializeVec3Array vec3Arr
  let tuple3Bytes := serializeTuple3Array tuple3Arr
  let vec4Bytes := Id.run do
    let pd := pdVec4
    let elemSize := pd.btype.bytes.toNat
    let mut bytes := ByteArray.replicate (vec4Arr.size * elemSize) 0
    for i in [0:vec4Arr.size] do
      bytes := pd.btype.toByteArray bytes (i * elemSize).toUSize sorry_proof vec4Arr[i]!
    return bytes
  let tuple4Bytes := Id.run do
    let pd := pdTuple4
    let elemSize := pd.btype.bytes.toNat
    let mut bytes := ByteArray.replicate (tuple4Arr.size * elemSize) 0
    for i in [0:tuple4Arr.size] do
      bytes := pd.btype.toByteArray bytes (i * elemSize).toUSize sorry_proof tuple4Arr[i]!
    return bytes

  -- Create Vec3Manual data (same as vec3 but different type)
  let vec3ManualArr : Array Vec3Manual := Array.range n |>.map fun i =>
    let f := i.toFloat.toFloat32
    ⟨f, f + 1, f + 2⟩
  let vec3ManualBytes := Id.run do
    let pd : PlainDataType Vec3Manual := inferInstance
    let elemSize := pd.btype.bytes.toNat
    let mut bytes := ByteArray.replicate (vec3ManualArr.size * elemSize) 0
    for i in [0:vec3ManualArr.size] do
      bytes := pd.btype.toByteArray bytes (i * elemSize).toUSize sorry_proof vec3ManualArr[i]!
    return bytes

  IO.println s!"\n═══ Vec3 vs Tuple3 (12 bytes each) ═══"
  runBench "Vec3   (derived)  " iters (fun _ => sumVec3X vec3Bytes n)
  runBench "Vec3   (manual)   " iters (fun _ => sumVec3ManualX vec3ManualBytes n)
  runBench "Tuple3 (raw)      " iters (fun _ => sumTuple3First tuple3Bytes n)

  IO.println s!"\n═══ Vec4 vs Tuple4 (16 bytes each) ═══"
  runBench "Vec4   (struct)" iters (fun _ => sumVec4X vec4Bytes n)
  runBench "Tuple4 (raw)   " iters (fun _ => sumTuple4First tuple4Bytes n)

  -- Correctness check
  IO.println s!"\n═══ Correctness Verification ═══"
  let vec3Sum := sumVec3X vec3Bytes n
  let tuple3Sum := sumTuple3First tuple3Bytes n
  IO.println s!"Vec3 sum:   {vec3Sum}"
  IO.println s!"Tuple3 sum: {tuple3Sum}"
  IO.println s!"Match: {vec3Sum == tuple3Sum}"

  let vec4Sum := sumVec4X vec4Bytes n
  let tuple4Sum := sumTuple4First tuple4Bytes n
  IO.println s!"Vec4 sum:   {vec4Sum}"
  IO.println s!"Tuple4 sum: {tuple4Sum}"
  IO.println s!"Match: {vec4Sum == tuple4Sum}"

  IO.println "\n✓ Done! If struct matches tuple performance, abstraction is zero-cost."
