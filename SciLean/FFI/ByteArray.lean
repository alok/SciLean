
/-- Get {given}`i`-th float out of {name}`ByteArray` if interpreted as {name}`FloatArray` -/
@[extern c inline "((double*)(lean_sarray_cptr(#1)))[#2]"]
-- @[extern "scilean_byte_array_uget_float"]
opaque ByteArray.ugetFloat (a : @& ByteArray) (i : USize) (hi : i.toNat*8 + 7 < a.size) : Float


@[extern c inline "(((double*)(lean_sarray_cptr(#1)+#2))[0])"]
-- @[extern "scilean_byte_array_uget_float"]
opaque ByteArray.ugetFloatAtByte (a : @& ByteArray) (i : USize) (hi : i.toNat + 7 < a.size) : Float


/-- Ensure that {name}`ByteArray` has reference counter one.

The reasong for {given}`uniqueName` argument is that we have to fight the common subexpression
optimization. For examples if we would write
```lean
let a : ByteArray := ...
let b := a.mkExclusive
let c := a.mkExclusive
```
then {lit}`b` and {lit}`c` point to the same object which is not what we want! Therefore the idea is to write
```lean
let a : ByteArray := ...
let b := a.mkExclusive `b
let c := a.mkExclusive `c
```
then common subexpression optimization can't collapse those two calls into one.

TODO: Is there a more robust way to avoid common subexpression optimization then {given}`uniqueName`?
-/
@[extern "scilean_byte_array_mk_exclusive"]
opaque ByteArray.mkExclusive (a : ByteArray) (uniqueName : Name) : ByteArray := a


/-- Set {given}`i`-th float out of {name}`ByteArray` if interpreted as {name}`FloatArray`

This function is unsafe! It mutates the array without checking the array -/
@[extern c inline "((((double*)(lean_sarray_cptr(#1)))[#2] = #3), #1)"]
unsafe opaque ByteArray.usetFloatUnsafe (a : ByteArray) (i : USize) (v : Float) (h : i.toNat*8 + 7 < a.size) : ByteArray

@[extern "scilean_byte_array_replicate"]
opaque ByteArray.replicate (n : @& Nat) (v : UInt8) : ByteArray
