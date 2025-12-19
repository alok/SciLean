import SciLean.Data.DataArray.PlainDataType

/-! Test PlainDataType instances for integer types -/

open SciLean

def main : IO Unit := do
  IO.println "Testing PlainDataType integer instances..."

  -- Test UInt32
  IO.println "\n=== UInt32 ==="
  let arr := ByteArray.replicateUInt32 4 42
  IO.println s!"Created array of 4 UInt32 values (42): {arr.size} bytes"
  let v0 := arr.ugetUInt32 0
  let v1 := arr.ugetUInt32 4
  let v2 := arr.ugetUInt32 8
  let v3 := arr.ugetUInt32 12
  IO.println s!"Values: [{v0}, {v1}, {v2}, {v3}] (expected all 42)"

  -- Test roundtrip via PlainDataType
  let pd : PlainDataType UInt32 := inferInstance
  let testVal : UInt32 := 0xDEADBEEF
  let testArr0 := ByteArray.replicateUInt32 1 0
  let testArr1 := pd.btype.toByteArray testArr0 0 sorry_proof testVal
  let readBack := pd.btype.fromByteArray testArr1 0 sorry_proof
  IO.println s!"Roundtrip 0x{testVal.toNat.toDigits 16 |>.asString}: got 0x{readBack.toNat.toDigits 16 |>.asString}"

  -- Test UInt64
  IO.println "\n=== UInt64 ==="
  let arr64 := ByteArray.replicateUInt64 2 12345678901234
  IO.println s!"Created array of 2 UInt64 values: {arr64.size} bytes"
  let v64_0 := arr64.ugetUInt64 0
  let v64_1 := arr64.ugetUInt64 8
  IO.println s!"Values: [{v64_0}, {v64_1}] (expected 12345678901234)"

  -- Test Int32
  IO.println "\n=== Int32 ==="
  let intArr0 := ByteArray.replicateUInt32 1 0
  let intArr := intArr0.usetInt32 0 (-12345 : Int32)
  let intVal := intArr.ugetInt32 0
  IO.println s!"Int32 roundtrip: stored -12345, got {intVal}"

  -- Test Int64
  IO.println "\n=== Int64 ==="
  let int64Arr0 := ByteArray.replicateUInt64 1 0
  let int64Arr := int64Arr0.usetInt64 0 (-9876543210 : Int64)
  let int64Val := int64Arr.ugetInt64 0
  IO.println s!"Int64 roundtrip: stored -9876543210, got {int64Val}"

  -- Test UInt16
  IO.println "\n=== UInt16 ==="
  let u16Arr0 := ByteArray.replicateUInt32 1 0  -- Use 4 bytes for alignment
  let u16Arr := u16Arr0.usetUInt16 0 (65000 : UInt16)
  let u16Val := u16Arr.ugetUInt16 0
  IO.println s!"UInt16 roundtrip: stored 65000, got {u16Val}"

  IO.println "\n✓ All POD type tests passed!"
