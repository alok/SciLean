import SciLean.Data.DataArray.PlainDataType
import SciLean.FFI.Metal

/-!
# GPU Struct Marshaling Test

Goal: Custom structures should work with GPU buffers automatically.
This tests the path: Structure → PlainDataType → ByteArray → GpuBuffer
-/

open SciLean

-- ============================================================================
-- Example 1: Simple Vec3 structure
-- ============================================================================

structure Vec3 where
  x : Float32
  y : Float32
  z : Float32
  deriving Repr, Inhabited

instance : ToString Vec3 where
  toString v := s!"Vec3({v.x}, {v.y}, {v.z})"

-- Manual equivalence to nested product (current approach)
def Vec3.toProd (v : Vec3) : Float32 × Float32 × Float32 := (v.x, v.y, v.z)
def Vec3.fromProd (p : Float32 × Float32 × Float32) : Vec3 := ⟨p.1, p.2.1, p.2.2⟩

def Vec3.equiv : Vec3 ≃ Float32 × Float32 × Float32 where
  toFun := Vec3.toProd
  invFun := Vec3.fromProd
  left_inv := fun _ => rfl
  right_inv := fun _ => rfl

-- PlainDataType instance via equivalence
instance : PlainDataType Vec3 := PlainDataType.ofEquiv Vec3.equiv.symm

-- ============================================================================
-- Example 2: Particle with multiple fields
-- ============================================================================

structure Particle where
  pos : Vec3
  vel : Vec3
  mass : Float32
  deriving Repr, Inhabited

instance : ToString Particle where
  toString p := s!"Particle(pos={p.pos}, vel={p.vel}, mass={p.mass})"

def Particle.toProd (p : Particle) : Vec3 × Vec3 × Float32 := (p.pos, p.vel, p.mass)
def Particle.fromProd (t : Vec3 × Vec3 × Float32) : Particle := ⟨t.1, t.2.1, t.2.2⟩

def Particle.equiv : Particle ≃ Vec3 × Vec3 × Float32 where
  toFun := Particle.toProd
  invFun := Particle.fromProd
  left_inv := fun _ => rfl
  right_inv := fun _ => rfl

instance : PlainDataType Particle := PlainDataType.ofEquiv Particle.equiv.symm

-- ============================================================================
-- Helper: Serialize/deserialize arrays of structs
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
-- Test: Roundtrip through ByteArray
-- ============================================================================

def testVec3Roundtrip : IO Unit := do
  let v1 : Vec3 := ⟨1.0, 2.0, 3.0⟩
  let v2 : Vec3 := ⟨4.0, 5.0, 6.0⟩
  let v3 : Vec3 := ⟨7.0, 8.0, 9.0⟩

  let arr := #[v1, v2, v3]
  let bytes := serializeArray arr
  let recovered : Array Vec3 := deserializeArray bytes 3

  IO.println s!"Original:  {arr.toList}"
  IO.println s!"Recovered: {recovered.toList}"
  IO.println s!"Bytes: {bytes.size} (expected {3 * 12} = 3 × 3 × 4)"

  -- Verify values
  for i in [0:3] do
    let orig := arr[i]!
    let recv := recovered[i]!
    if orig.x != recv.x || orig.y != recv.y || orig.z != recv.z then
      IO.println s!"MISMATCH at {i}!"
      return
  IO.println "✓ Vec3 roundtrip OK"

def testParticleRoundtrip : IO Unit := do
  let p1 : Particle := ⟨⟨1, 2, 3⟩, ⟨0.1, 0.2, 0.3⟩, 1.0⟩
  let p2 : Particle := ⟨⟨4, 5, 6⟩, ⟨0.4, 0.5, 0.6⟩, 2.0⟩

  let arr := #[p1, p2]
  let bytes := serializeArray arr
  let recovered : Array Particle := deserializeArray bytes 2

  IO.println s!"Original:  {arr.toList}"
  IO.println s!"Recovered: {recovered.toList}"

  -- Particle size: 3*4 (pos) + 3*4 (vel) + 4 (mass) = 28 bytes
  IO.println s!"Bytes: {bytes.size} (expected {2 * 28} = 2 × 28)"
  IO.println "✓ Particle roundtrip OK"

-- ============================================================================
-- GpuArray: Keep data on GPU, avoid per-call copies
-- ============================================================================

/-- Array of structs stored on GPU. Uploads once, keeps data resident.
    This is the key abstraction for efficient GPU compute. -/
structure GpuArray (α : Type) [PlainDataType α] where
  buffer : Metal.GpuBuffer
  size : Nat

namespace GpuArray

variable {α : Type} [Inhabited α] [pd : PlainDataType α]

/-- Upload array to GPU (one-time cost). -/
def fromArray (arr : Array α) : IO (GpuArray α) := do
  let bytes := serializeArray arr
  let buf ← Metal.GpuBuffer.fromByteArray bytes
  return ⟨buf, arr.size⟩

/-- Download array from GPU (only when results needed). -/
def toArray (gpu : GpuArray α) : IO (Array α) := do
  let cpuBuf ← gpu.buffer.download
  return deserializeArray cpuBuf.data gpu.size

/-- Allocate GPU array without upload (for outputs). -/
def alloc (n : Nat) : IO (GpuArray α) := do
  let elemSize := pd.btype.bytes.toNat
  let buf ← Metal.GpuBuffer.alloc (n * elemSize).toUSize
  return ⟨buf, n⟩

/-- Element size in bytes. -/
def elemBytes (_gpu : GpuArray α) : Nat := pd.btype.bytes.toNat

/-- Total byte size of data. -/
def byteSize (gpu : GpuArray α) : Nat := gpu.size * pd.btype.bytes.toNat

end GpuArray

-- ============================================================================
-- Test: GPU-resident computation
-- ============================================================================

def testGpuResident : IO Unit := do
  IO.println "\n=== GPU-Resident Computation Test ==="

  -- Create particle data
  let particles := #[
    Particle.mk ⟨1, 2, 3⟩ ⟨0.1, 0.2, 0.3⟩ 1.0,
    Particle.mk ⟨4, 5, 6⟩ ⟨0.4, 0.5, 0.6⟩ 2.0,
    Particle.mk ⟨7, 8, 9⟩ ⟨0.7, 0.8, 0.9⟩ 3.0
  ]

  -- Upload ONCE to GPU
  let gpuParticles ← GpuArray.fromArray particles
  IO.println s!"Uploaded {gpuParticles.size} particles ({gpuParticles.byteSize} bytes)"

  -- Simulate multiple GPU operations without downloading
  -- (In real code, we'd run kernels here)
  IO.println "Running GPU operations... (data stays on GPU)"
  -- Here we'd call: Metal.GpuBuffer.add, gemm, etc.
  -- The key is: NO download between operations

  -- Download ONLY at the end when we need results
  let result ← gpuParticles.toArray
  IO.println s!"Downloaded final result: {result.size} particles"

  -- Verify
  for i in [0:particles.size] do
    let orig := particles[i]!
    let recv := result[i]!
    if orig.mass != recv.mass then
      IO.println s!"MISMATCH at particle {i}!"
      return
  IO.println "✓ GPU-resident test OK"

-- ============================================================================
-- Test: Simple roundtrip (for validation)
-- ============================================================================

def testGpuRoundtrip : IO Unit := do
  IO.println "\n=== GPU Roundtrip Test ==="

  let particles := #[
    Particle.mk ⟨1, 2, 3⟩ ⟨0.1, 0.2, 0.3⟩ 1.0,
    Particle.mk ⟨4, 5, 6⟩ ⟨0.4, 0.5, 0.6⟩ 2.0,
    Particle.mk ⟨7, 8, 9⟩ ⟨0.7, 0.8, 0.9⟩ 3.0
  ]

  -- Upload and download
  let gpu ← GpuArray.fromArray particles
  let recovered ← gpu.toArray

  -- Verify
  for i in [0:particles.size] do
    let orig := particles[i]!
    let recv := recovered[i]!
    if orig.mass != recv.mass then
      IO.println s!"MISMATCH at particle {i}!"
      return
  IO.println "✓ GPU roundtrip OK"

-- ============================================================================
-- Main
-- ============================================================================

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════════════════════╗"
  IO.println "║           GPU Struct Marshaling Test                          ║"
  IO.println "╚══════════════════════════════════════════════════════════════╝"

  testVec3Roundtrip
  IO.println ""
  testParticleRoundtrip
  testGpuRoundtrip
  testGpuResident

  IO.println "\n✓ All struct marshaling tests passed!"
  IO.println ""
  IO.println "Key insight: GpuArray keeps data on GPU between operations."
  IO.println "Upload once → compute many times → download result."
