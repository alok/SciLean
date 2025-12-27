import Init.Prelude
import Lean.Elab.DocString
import Verso.Code.External

open Lean
open Lean.Doc
open scoped Lean.Doc.Syntax

private def onlyCode (xs : TSyntaxArray `inline) : DocM StrLit := do
  if h : xs.size = 1 then
    let stx := xs[0]
    match stx with
    | `(inline|code($s)) => pure s
    | _ => throwErrorAt stx "expected a code literal"
  else
    throwError "expected a single code literal"

/-- Dummy code block for non-Lean languages. -/
@[doc_code_block]
def nonLeanCode (s : StrLit) : DocM (Block ElabInline ElabBlock) := do
  pure (.code s.getString)

/-- Root-level alias for the lit role. -/
@[doc_role]
def lit (xs : TSyntaxArray `inline) : DocM (Inline ElabInline) := do
  let s ← onlyCode xs
  pure (.code s.getString)

/-- Alias for {name (full := _root_.lit)}`lit` when {lit}`Verso.Code.External` is open. -/
@[doc_role Verso.Code.External.lit]
def litExternal (xs : TSyntaxArray `inline) : DocM (Inline ElabInline) := do
  _root_.lit xs
