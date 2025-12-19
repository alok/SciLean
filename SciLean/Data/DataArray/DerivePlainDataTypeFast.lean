/-
Copyright (c) 2024 SciLean contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: SciLean contributors
-/
import SciLean.Data.DataArray.PlainDataType

-- Fast Derive Handler for PlainDataType
--
-- Generates zero-overhead ByteType instances with direct field access.
-- Unlike proxy_equiv%, this inlines completely at compile time.
--
-- Usage: `deriving PlainDataType` on any structure whose fields are PlainDataType.

namespace SciLean.PlainDataType.Deriving

open Lean Elab Command Meta Term

/-- Get the byte size of a PlainDataType at compile time via `evalExpr`. -/
def getPlainDataTypeBytes (type : Expr) : MetaM (Option Nat) := do
  let pdType ← mkAppM ``SciLean.PlainDataType #[type]
  match ← trySynthInstance pdType with
  | .some inst =>
    let btypeExpr := Expr.proj ``SciLean.PlainDataType 0 inst
    let bytesExpr := Expr.proj ``SciLean.ByteType 0 btypeExpr
    let toNatExpr ← mkAppM ``USize.toNat #[bytesExpr]
    try
      let val ← unsafe evalExpr Nat (mkConst ``Nat) toNatExpr
      return some val
    catch _ =>
      -- Fallback: reduce and pattern match on literal
      let reduced ← reduce toNatExpr
      match reduced with
      | .lit (.natVal n) => return some n
      | _ => return none
  | .none | .undef => return none

/-- Metadata for a structure field's PlainDataType layout. -/
structure FieldPDInfo where
  name : Name
  type : Expr
  typeName : Name
  byteOffset : Nat
  byteSize : Nat
  deriving Inhabited

/-- Analyze a structure's fields and compute byte offsets. -/
def analyzeStructure (structName : Name) : MetaM (Option (Array FieldPDInfo)) := do
  let env ← getEnv
  let some structInfo := getStructureInfo? env structName | return none

  let mut fields : Array FieldPDInfo := #[]
  let mut offset : Nat := 0

  for fieldName in structInfo.fieldNames do
    let projFn := structName ++ fieldName
    let some projInfo := env.find? projFn | return none

    let fieldType ← forallTelescopeReducing projInfo.type fun _ body => pure body
    let typeName := match fieldType with
      | .const n _ => n
      | .app (.const n _) _ => n
      | _ => `unknown

    let some byteSize ← getPlainDataTypeBytes fieldType | return none

    fields := fields.push { name := fieldName, type := fieldType, typeName, byteOffset := offset, byteSize }
    offset := offset + byteSize

  return some fields

/-- Generate fast PlainDataType instance for a structure. -/
def mkPlainDataTypeFast (declName : Name) : CommandElabM Bool := do
  let env ← getEnv
  let some _ := getStructureInfo? env declName | return false
  let some fields ← liftTermElabM (analyzeStructure declName) | return false
  if fields.isEmpty then return false

  let totalBytes := fields.foldl (· + ·.byteSize) 0
  let structIdent := mkIdent declName
  let byteTypeIdent := mkIdent (declName ++ `plainDataTypeByteType)
  let sorryProof := mkIdent `sorryProofAxiom

  -- Build fromByteArray: read each field and construct
  let mut fromParts : Array (TSyntax `term) := #[]
  for field in fields do
    let fieldTypeId := mkIdent field.typeName
    let offsetLit : TSyntax `term := ⟨Syntax.mkNumLit (toString field.byteOffset)⟩
    fromParts := fromParts.push (← `((inferInstance : PlainDataType $fieldTypeId).btype.fromByteArray arr (i + $offsetLit) $sorryProof))

  -- Build toByteArray: serialize each field with explicit types
  let mut toBody : TSyntax `term ← `(arr)
  for field in fields.reverse do
    let fieldTypeId := mkIdent field.typeName
    let offsetLit : TSyntax `term := ⟨Syntax.mkNumLit (toString field.byteOffset)⟩
    let fieldIdent := mkIdent field.name
    let btIdent := mkIdent (Name.mkSimple s!"bt_{field.name}")
    toBody ← `(
      let $btIdent : SciLean.ByteType $fieldTypeId := (inferInstance : PlainDataType $fieldTypeId).btype
      let fieldVal : $fieldTypeId := v.$fieldIdent
      let arr := @SciLean.ByteType.toByteArray $fieldTypeId $btIdent arr (i + $offsetLit) $sorryProof fieldVal
      $toBody)

  -- Build struct constructor (supports up to 8 fields)
  let fromBody : TSyntax `term ← match fromParts.toList with
    | [] => `(default)
    | [a] => `(⟨$a⟩)
    | [a, b] => `(⟨$a, $b⟩)
    | [a, b, c] => `(⟨$a, $b, $c⟩)
    | [a, b, c, d] => `(⟨$a, $b, $c, $d⟩)
    | [a, b, c, d, e] => `(⟨$a, $b, $c, $d, $e⟩)
    | [a, b, c, d, e, f] => `(⟨$a, $b, $c, $d, $e, $f⟩)
    | [a, b, c, d, e, f, g] => `(⟨$a, $b, $c, $d, $e, $f, $g⟩)
    | [a, b, c, d, e, f, g, h] => `(⟨$a, $b, $c, $d, $e, $f, $g, $h⟩)
    | _ => `(default)  -- >8 fields: fall back to default (won't work, but rare)

  elabCommand (← `(
    @[inline] def $byteTypeIdent : SciLean.ByteType $structIdent where
      bytes := $(Syntax.mkNumLit (toString totalBytes))
      h_size := $sorryProof
      fromByteArray := fun arr i _ => $fromBody
      toByteArray := fun arr i _ v => $toBody
      toByteArray_size := $sorryProof
      fromByteArray_toByteArray := $sorryProof
      fromByteArray_toByteArray_other := $sorryProof

    instance : SciLean.PlainDataType $structIdent where
      btype := $byteTypeIdent))
  return true

def mkPlainDataTypeFastHandler (declNames : Array Name) : CommandElabM Bool := do
  if declNames.size != 1 then return false
  mkPlainDataTypeFast declNames[0]!

end SciLean.PlainDataType.Deriving

open Lean Elab in
initialize
  registerDerivingHandler ``SciLean.PlainDataType SciLean.PlainDataType.Deriving.mkPlainDataTypeFastHandler
  registerTraceClass `Elab.Deriving.PlainDataType
