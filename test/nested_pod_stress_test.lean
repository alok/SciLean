import SciLean.Data.DataArray.DerivePlainDataTypeFast

/-!
# Deeply Nested POD Stress Test

Test that the fast PlainDataType derive handler correctly handles:
1. Multiple levels of nesting (3-4 deep)
2. Mixed field types at each level
3. Correctness of byte offsets across nesting
-/

open SciLean

-- ============================================================================
-- Level 1: Basic types
-- ============================================================================

structure Point2D where
  x : Float32
  y : Float32
  deriving PlainDataType, Repr, Inhabited

structure Point3D where
  x : Float32
  y : Float32
  z : Float32
  deriving PlainDataType, Repr, Inhabited

-- ============================================================================
-- Level 2: Structures containing Level 1
-- ============================================================================

structure Line2D where
  start : Point2D
  stop : Point2D
  deriving PlainDataType, Repr, Inhabited

structure Triangle3D where
  a : Point3D
  b : Point3D
  c : Point3D
  deriving PlainDataType, Repr, Inhabited

structure AABB where  -- Axis-aligned bounding box
  min : Point3D
  max : Point3D
  deriving PlainDataType, Repr, Inhabited

-- ============================================================================
-- Level 3: Structures containing Level 2
-- ============================================================================

structure Ray where
  origin : Point3D
  direction : Point3D
  tMin : Float32
  tMax : Float32
  deriving PlainDataType, Repr, Inhabited

structure BoundedTriangle where
  triangle : Triangle3D
  bounds : AABB
  materialId : Float32  -- Using Float32 as placeholder for ID
  deriving PlainDataType, Repr, Inhabited

-- ============================================================================
-- Level 4: Deep nesting
-- ============================================================================

structure SceneObject where
  mesh : BoundedTriangle
  transform : Point3D  -- Simplified transform (just position)
  scale : Float32
  deriving PlainDataType, Repr, Inhabited

structure RayHit where
  ray : Ray
  hitPoint : Point3D
  normal : Point3D
  distance : Float32
  objectId : Float32
  deriving PlainDataType, Repr, Inhabited

-- ============================================================================
-- Serialization helpers
-- ============================================================================

def serializeArray {α : Type} [Inhabited α] [pd : PlainDataType α] (arr : Array α) : ByteArray := Id.run do
  let elemSize := pd.btype.bytes.toNat
  let totalSize := arr.size * elemSize
  let mut bytes := ByteArray.replicate totalSize 0
  for i in [0:arr.size] do
    bytes := pd.btype.toByteArray bytes (i * elemSize).toUSize sorry_proof arr[i]!
  return bytes

def deserializeArray {α : Type} [pd : PlainDataType α] (bytes : ByteArray) (n : Nat) : Array α := Id.run do
  let elemSize := pd.btype.bytes.toNat
  let mut arr : Array α := #[]
  for i in [0:n] do
    let elem := pd.btype.fromByteArray bytes (i * elemSize).toUSize sorry_proof
    arr := arr.push elem
  return arr

-- ============================================================================
-- Roundtrip tests
-- ============================================================================

def testLevel1 : IO Bool := do
  IO.println "\n=== Level 1: Basic types ==="

  let pt2 : Point2D := ⟨1.5, 2.5⟩
  let pt3 : Point3D := ⟨1.0, 2.0, 3.0⟩

  let pd2 : PlainDataType Point2D := inferInstance
  let pd3 : PlainDataType Point3D := inferInstance

  IO.println s!"Point2D size: {pd2.btype.bytes} bytes (expected 8)"
  IO.println s!"Point3D size: {pd3.btype.bytes} bytes (expected 12)"

  -- Roundtrip
  let arr2 := #[pt2, ⟨3.5, 4.5⟩]
  let bytes2 := serializeArray arr2
  let recovered2 : Array Point2D := deserializeArray bytes2 2

  let arr3 := #[pt3, ⟨4.0, 5.0, 6.0⟩]
  let bytes3 := serializeArray arr3
  let recovered3 : Array Point3D := deserializeArray bytes3 2

  let ok2 := recovered2[0]!.x == pt2.x && recovered2[0]!.y == pt2.y
  let ok3 := recovered3[0]!.x == pt3.x && recovered3[0]!.y == pt3.y && recovered3[0]!.z == pt3.z

  IO.println s!"Point2D roundtrip: {if ok2 then "✓" else "✗"}"
  IO.println s!"Point3D roundtrip: {if ok3 then "✓" else "✗"}"

  return ok2 && ok3

def testLevel2 : IO Bool := do
  IO.println "\n=== Level 2: Nested once ==="

  let line : Line2D := ⟨⟨0, 0⟩, ⟨10, 10⟩⟩
  let tri : Triangle3D := ⟨⟨0, 0, 0⟩, ⟨1, 0, 0⟩, ⟨0, 1, 0⟩⟩
  let aabb : AABB := ⟨⟨-1, -1, -1⟩, ⟨1, 1, 1⟩⟩

  let pdLine : PlainDataType Line2D := inferInstance
  let pdTri : PlainDataType Triangle3D := inferInstance
  let pdAABB : PlainDataType AABB := inferInstance

  IO.println s!"Line2D size: {pdLine.btype.bytes} bytes (expected 16 = 2×8)"
  IO.println s!"Triangle3D size: {pdTri.btype.bytes} bytes (expected 36 = 3×12)"
  IO.println s!"AABB size: {pdAABB.btype.bytes} bytes (expected 24 = 2×12)"

  -- Roundtrip
  let arrLine := #[line]
  let bytesLine := serializeArray arrLine
  let recoveredLine : Array Line2D := deserializeArray bytesLine 1

  let arrTri := #[tri]
  let bytesTri := serializeArray arrTri
  let recoveredTri : Array Triangle3D := deserializeArray bytesTri 1

  let okLine := recoveredLine[0]!.start.x == line.start.x &&
                recoveredLine[0]!.stop.y == line.stop.y
  let okTri := recoveredTri[0]!.a.x == tri.a.x &&
               recoveredTri[0]!.b.x == tri.b.x &&
               recoveredTri[0]!.c.y == tri.c.y

  IO.println s!"Line2D roundtrip: {if okLine then "✓" else "✗"}"
  IO.println s!"Triangle3D roundtrip: {if okTri then "✓" else "✗"}"

  return okLine && okTri

def testLevel3 : IO Bool := do
  IO.println "\n=== Level 3: Nested twice ==="

  let ray : Ray := ⟨⟨0, 0, 0⟩, ⟨0, 0, 1⟩, 0.001, 1000.0⟩
  let bt : BoundedTriangle := ⟨
    ⟨⟨0, 0, 0⟩, ⟨1, 0, 0⟩, ⟨0, 1, 0⟩⟩,  -- triangle
    ⟨⟨0, 0, 0⟩, ⟨1, 1, 0⟩⟩,              -- bounds
    42.0                                   -- materialId
  ⟩

  let pdRay : PlainDataType Ray := inferInstance
  let pdBT : PlainDataType BoundedTriangle := inferInstance

  IO.println s!"Ray size: {pdRay.btype.bytes} bytes (expected 32 = 2×12 + 2×4)"
  IO.println s!"BoundedTriangle size: {pdBT.btype.bytes} bytes (expected 64 = 36 + 24 + 4)"

  -- Roundtrip
  let arrRay := #[ray]
  let bytesRay := serializeArray arrRay
  let recoveredRay : Array Ray := deserializeArray bytesRay 1

  let arrBT := #[bt]
  let bytesBT := serializeArray arrBT
  let recoveredBT : Array BoundedTriangle := deserializeArray bytesBT 1

  let okRay := recoveredRay[0]!.origin.x == ray.origin.x &&
               recoveredRay[0]!.direction.z == ray.direction.z &&
               recoveredRay[0]!.tMax == ray.tMax
  let okBT := recoveredBT[0]!.triangle.a.x == bt.triangle.a.x &&
              recoveredBT[0]!.bounds.max.y == bt.bounds.max.y &&
              recoveredBT[0]!.materialId == bt.materialId

  IO.println s!"Ray roundtrip: {if okRay then "✓" else "✗"}"
  IO.println s!"BoundedTriangle roundtrip: {if okBT then "✓" else "✗"}"

  return okRay && okBT

def testLevel4 : IO Bool := do
  IO.println "\n=== Level 4: Deep nesting (3+ levels) ==="

  let sceneObj : SceneObject := ⟨
    ⟨⟨⟨0, 0, 0⟩, ⟨1, 0, 0⟩, ⟨0, 1, 0⟩⟩, ⟨⟨0, 0, 0⟩, ⟨1, 1, 0⟩⟩, 1.0⟩,  -- mesh
    ⟨5, 5, 5⟩,  -- transform
    2.0         -- scale
  ⟩

  let rayHit : RayHit := ⟨
    ⟨⟨0, 0, -5⟩, ⟨0, 0, 1⟩, 0.001, 100.0⟩,  -- ray
    ⟨0.5, 0.5, 0⟩,                           -- hitPoint
    ⟨0, 0, -1⟩,                              -- normal
    5.0,                                      -- distance
    7.0                                       -- objectId
  ⟩

  let pdSO : PlainDataType SceneObject := inferInstance
  let pdRH : PlainDataType RayHit := inferInstance

  IO.println s!"SceneObject size: {pdSO.btype.bytes} bytes (expected 80 = 64 + 12 + 4)"
  IO.println s!"RayHit size: {pdRH.btype.bytes} bytes (expected 64 = 32 + 12 + 12 + 4 + 4)"

  -- Roundtrip
  let arrSO := #[sceneObj]
  let bytesSO := serializeArray arrSO
  let recoveredSO : Array SceneObject := deserializeArray bytesSO 1

  let arrRH := #[rayHit]
  let bytesRH := serializeArray arrRH
  let recoveredRH : Array RayHit := deserializeArray bytesRH 1

  -- Deep field access test
  let okSO := recoveredSO[0]!.mesh.triangle.b.x == sceneObj.mesh.triangle.b.x &&
              recoveredSO[0]!.mesh.bounds.max.y == sceneObj.mesh.bounds.max.y &&
              recoveredSO[0]!.transform.z == sceneObj.transform.z &&
              recoveredSO[0]!.scale == sceneObj.scale

  let okRH := recoveredRH[0]!.ray.origin.z == rayHit.ray.origin.z &&
              recoveredRH[0]!.ray.tMax == rayHit.ray.tMax &&
              recoveredRH[0]!.hitPoint.x == rayHit.hitPoint.x &&
              recoveredRH[0]!.distance == rayHit.distance

  IO.println s!"SceneObject roundtrip: {if okSO then "✓" else "✗"}"
  IO.println s!"RayHit roundtrip: {if okRH then "✓" else "✗"}"

  return okSO && okRH

def testPerformance : IO Unit := do
  IO.println "\n=== Performance: Deep nested serialization ==="

  let n := 10000
  let iters := 100

  -- Create test data
  let sceneObjects : Array SceneObject := Array.range n |>.map fun i =>
    let f := i.toFloat.toFloat32
    ⟨⟨⟨⟨f, f+1, f+2⟩, ⟨f+3, f+4, f+5⟩, ⟨f+6, f+7, f+8⟩⟩,
      ⟨⟨f, f, f⟩, ⟨f+10, f+10, f+10⟩⟩, f⟩,
     ⟨f, f, f⟩, 1.0⟩

  let pd : PlainDataType SceneObject := inferInstance
  IO.println s!"SceneObject size: {pd.btype.bytes} bytes"
  IO.println s!"Array size: {n} elements = {n * pd.btype.bytes.toNat} bytes"

  -- Warmup
  let mut bytes := serializeArray sceneObjects
  let _ : Array SceneObject := deserializeArray bytes n

  -- Timed serialization
  let start ← IO.monoMsNow
  for _ in [0:iters] do
    bytes := serializeArray sceneObjects
  let serTime := (← IO.monoMsNow) - start

  -- Timed deserialization
  let start2 ← IO.monoMsNow
  for _ in [0:iters] do
    let _ : Array SceneObject := deserializeArray bytes n
  let deserTime := (← IO.monoMsNow) - start2

  let totalBytes := n * pd.btype.bytes.toNat * iters
  let serThroughput := (totalBytes.toFloat / 1e6) / (serTime.toFloat / 1000.0)
  let deserThroughput := (totalBytes.toFloat / 1e6) / (deserTime.toFloat / 1000.0)

  IO.println s!"Serialize {iters}×{n} objects: {serTime}ms ({serThroughput} MB/s)"
  IO.println s!"Deserialize {iters}×{n} objects: {deserTime}ms ({deserThroughput} MB/s)"

-- ============================================================================
-- Main
-- ============================================================================

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════════════════════╗"
  IO.println "║         Deeply Nested POD Stress Test                         ║"
  IO.println "╚══════════════════════════════════════════════════════════════╝"

  let ok1 ← testLevel1
  let ok2 ← testLevel2
  let ok3 ← testLevel3
  let ok4 ← testLevel4

  testPerformance

  IO.println "\n═══ Summary ═══"
  if ok1 && ok2 && ok3 && ok4 then
    IO.println "✓ All nested POD tests passed!"
  else
    IO.println "✗ Some tests failed"
    IO.println s!"  Level 1: {if ok1 then "✓" else "✗"}"
    IO.println s!"  Level 2: {if ok2 then "✓" else "✗"}"
    IO.println s!"  Level 3: {if ok3 then "✓" else "✗"}"
    IO.println s!"  Level 4: {if ok4 then "✓" else "✗"}"
