/-!
# Integer Array FFI

Native integer type (UInt16, UInt32, Int32, UInt64, Int64) array operations via FFI.
This provides direct memory access without Nat conversion overhead.
-/

-- ============================================================================
-- UInt16 FFI
-- ============================================================================

/-- Read a UInt16 from ByteArray at byte offset `i`. -/
@[extern "scilean_byte_array_uget_uint16"]
opaque ByteArray.ugetUInt16 (a : @& ByteArray) (i : USize) : UInt16

/-- Set a UInt16 in ByteArray at byte offset `i`. Unsafe: no bounds checking. -/
@[extern "scilean_byte_array_uset_uint16"]
unsafe opaque ByteArray.usetUInt16Unsafe (a : ByteArray) (i : USize) (v : UInt16) : ByteArray

/-- Safe wrapper for setting UInt16 in ByteArray. -/
@[implemented_by ByteArray.usetUInt16Unsafe]
opaque ByteArray.usetUInt16 (a : ByteArray) (i : USize) (v : UInt16) : ByteArray := a

-- ============================================================================
-- UInt32 FFI
-- ============================================================================

/-- Read a UInt32 from ByteArray at byte offset `i`. -/
@[extern "scilean_byte_array_uget_uint32"]
opaque ByteArray.ugetUInt32 (a : @& ByteArray) (i : USize) : UInt32

/-- Set a UInt32 in ByteArray at byte offset `i`. Unsafe: no bounds checking. -/
@[extern "scilean_byte_array_uset_uint32"]
unsafe opaque ByteArray.usetUInt32Unsafe (a : ByteArray) (i : USize) (v : UInt32) : ByteArray

/-- Safe wrapper for setting UInt32 in ByteArray. -/
@[implemented_by ByteArray.usetUInt32Unsafe]
opaque ByteArray.usetUInt32 (a : ByteArray) (i : USize) (v : UInt32) : ByteArray := a

-- ============================================================================
-- Int32 FFI
-- ============================================================================

/-- Read an Int32 from ByteArray at byte offset `i`. -/
@[extern "scilean_byte_array_uget_int32"]
opaque ByteArray.ugetInt32 (a : @& ByteArray) (i : USize) : Int32

/-- Set an Int32 in ByteArray at byte offset `i`. Unsafe: no bounds checking. -/
@[extern "scilean_byte_array_uset_int32"]
unsafe opaque ByteArray.usetInt32Unsafe (a : ByteArray) (i : USize) (v : Int32) : ByteArray

/-- Safe wrapper for setting Int32 in ByteArray. -/
@[implemented_by ByteArray.usetInt32Unsafe]
opaque ByteArray.usetInt32 (a : ByteArray) (i : USize) (v : Int32) : ByteArray := a

-- ============================================================================
-- UInt64 FFI
-- ============================================================================

/-- Read a UInt64 from ByteArray at byte offset `i`. -/
@[extern "scilean_byte_array_uget_uint64"]
opaque ByteArray.ugetUInt64 (a : @& ByteArray) (i : USize) : UInt64

/-- Set a UInt64 in ByteArray at byte offset `i`. Unsafe: no bounds checking. -/
@[extern "scilean_byte_array_uset_uint64"]
unsafe opaque ByteArray.usetUInt64Unsafe (a : ByteArray) (i : USize) (v : UInt64) : ByteArray

/-- Safe wrapper for setting UInt64 in ByteArray. -/
@[implemented_by ByteArray.usetUInt64Unsafe]
opaque ByteArray.usetUInt64 (a : ByteArray) (i : USize) (v : UInt64) : ByteArray := a

-- ============================================================================
-- Int64 FFI
-- ============================================================================

/-- Read an Int64 from ByteArray at byte offset `i`. -/
@[extern "scilean_byte_array_uget_int64"]
opaque ByteArray.ugetInt64 (a : @& ByteArray) (i : USize) : Int64

/-- Set an Int64 in ByteArray at byte offset `i`. Unsafe: no bounds checking. -/
@[extern "scilean_byte_array_uset_int64"]
unsafe opaque ByteArray.usetInt64Unsafe (a : ByteArray) (i : USize) (v : Int64) : ByteArray

/-- Safe wrapper for setting Int64 in ByteArray. -/
@[implemented_by ByteArray.usetInt64Unsafe]
opaque ByteArray.usetInt64 (a : ByteArray) (i : USize) (v : Int64) : ByteArray := a

-- ============================================================================
-- Utility Functions
-- ============================================================================

/-- Create a ByteArray filled with n copies of a UInt32 value. -/
def ByteArray.replicateUInt32 (n : Nat) (v : UInt32) : ByteArray := Id.run do
  let mut arr := ByteArray.empty
  for _ in [:n * 4] do
    arr := arr.push 0
  for i in [:n] do
    arr := arr.usetUInt32 (i * 4).toUSize v
  return arr

/-- Create a ByteArray filled with n copies of a UInt64 value. -/
def ByteArray.replicateUInt64 (n : Nat) (v : UInt64) : ByteArray := Id.run do
  let mut arr := ByteArray.empty
  for _ in [:n * 8] do
    arr := arr.push 0
  for i in [:n] do
    arr := arr.usetUInt64 (i * 8).toUSize v
  return arr
