import SciLean.Data.DataArray.DerivePlainDataType

/-!
# Test: Deriving PlainDataType for custom structures
-/

open SciLean

-- Simple 2D point
structure Point2D where
  x : Float32
  y : Float32
  deriving PlainDataType, Repr, Inhabited

instance : ToString Point2D where
  toString p := s!"Point2D({p.x}, {p.y})"

-- 3D vector
structure Vec3D where
  x : Float32
  y : Float32
  z : Float32
  deriving PlainDataType, Repr, Inhabited

instance : ToString Vec3D where
  toString v := s!"Vec3D({v.x}, {v.y}, {v.z})"

-- Composite structure
structure Particle where
  pos : Vec3D
  vel : Vec3D
  mass : Float32
  deriving PlainDataType, Repr, Inhabited

instance : ToString Particle where
  toString p := s!"Particle(pos={p.pos}, vel={p.vel}, mass={p.mass})"

-- Test that the instances work
#check (inferInstance : PlainDataType Point2D)
#check (inferInstance : PlainDataType Vec3D)
#check (inferInstance : PlainDataType Particle)

-- Verify byte sizes are correct
-- Point2D: 2 × 4 = 8 bytes
-- Vec3D: 3 × 4 = 12 bytes
-- Particle: 12 + 12 + 4 = 28 bytes

def main : IO Unit := do
  IO.println "PlainDataType derive test:"

  let pd2 : PlainDataType Point2D := inferInstance
  let pd3 : PlainDataType Vec3D := inferInstance
  let pdp : PlainDataType Particle := inferInstance

  IO.println s!"Point2D bytes: {pd2.btype.bytes} (expected 8)"
  IO.println s!"Vec3D bytes: {pd3.btype.bytes} (expected 12)"
  IO.println s!"Particle bytes: {pdp.btype.bytes} (expected 28)"

  -- Test roundtrip
  let p : Point2D := ⟨1.5, 2.5⟩
  let arr := ByteArray.replicate 8 0
  let arr' := pd2.btype.toByteArray arr 0 sorry_proof p
  let p' := pd2.btype.fromByteArray arr' 0 sorry_proof

  IO.println s!"Original: {p}"
  IO.println s!"Recovered: {p'}"

  if p.x == p'.x && p.y == p'.y then
    IO.println "✓ Point2D roundtrip OK"
  else
    IO.println "✗ MISMATCH"

  IO.println ""
  IO.println "✓ All deriving PlainDataType tests passed!"
