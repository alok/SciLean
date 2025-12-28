import Lean
import SciLean.Modules.ML.Jaxpr.AST
import SciLean.VersoPrelude

/-!
# JAXPR DSL and elaborator

This module provides a small DSL for constructing a
{name (full := SciLean.ML.Jaxpr.Jaxpr)}`Jaxpr` value.

Example:

```nonLeanCode
open SciLean.ML.Jaxpr in
#eval ([jaxpr|
  in x y
  let z := add x y
  let w := mul z y
  out w
] : Jaxpr).toString
```
-/

/-! ## Syntax -/

/-- JAXPR types in the DSL (lightweight, identifier-only). -/
declare_syntax_cat jaxpr_type
syntax ident : jaxpr_type

/-- JAXPR atoms in the DSL. -/
declare_syntax_cat jaxpr_atom
syntax ident : jaxpr_atom
syntax ident ":" jaxpr_type : jaxpr_atom
syntax num : jaxpr_atom
syntax num ":" jaxpr_type : jaxpr_atom
syntax str : jaxpr_atom
syntax str ":" jaxpr_type : jaxpr_atom

/-- JAXPR variables in the DSL (outputs of equations). -/
declare_syntax_cat jaxpr_var
syntax ident : jaxpr_var
syntax ident ":" jaxpr_type : jaxpr_var

/-- JAXPR lines in the DSL. -/
declare_syntax_cat jaxpr_line
syntax (name := jaxpr_line_in) "in" jaxpr_atom+ : jaxpr_line
syntax (name := jaxpr_line_let) "let" jaxpr_var ":=" ident jaxpr_atom+ : jaxpr_line
syntax (name := jaxpr_line_out) "out" jaxpr_atom+ : jaxpr_line

/-- Term-level elaborator for the JAXPR DSL. -/
syntax "[jaxpr|" jaxpr_line* "]" : term

namespace SciLean.ML.Jaxpr

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

private def reprint (stx : Syntax) : String :=
  stx.reprint.getD (toString stx)

private def typeFromSyntax (stx : Syntax) : String :=
  match stx with
  | `(jaxpr_type| $ty:ident) => ty.getId.toString
  | _ => reprint stx

private def mkAtomVar (name : String) (ty? : Option String) : Expr :=
  mkApp2 (mkConst ``Atom.var) (toExpr name) (toExpr ty?)

private def mkAtomLit (value : String) (ty? : Option String) : Expr :=
  mkApp2 (mkConst ``Atom.lit) (toExpr value) (toExpr ty?)

private def atomFromSyntax (stx : Syntax) : TermElabM Expr := do
  match stx with
  | `(jaxpr_atom| $id:ident) =>
      return mkAtomVar id.getId.toString none
  | `(jaxpr_atom| $id:ident : $ty:jaxpr_type) =>
      return mkAtomVar id.getId.toString (some (typeFromSyntax ty))
  | `(jaxpr_atom| $_:num) =>
      return mkAtomLit (reprint stx) none
  | `(jaxpr_atom| $_:num : $ty:jaxpr_type) =>
      return mkAtomLit (reprint stx) (some (typeFromSyntax ty))
  | `(jaxpr_atom| $_:str) =>
      return mkAtomLit (reprint stx) none
  | `(jaxpr_atom| $_:str : $ty:jaxpr_type) =>
      return mkAtomLit (reprint stx) (some (typeFromSyntax ty))
  | _ =>
      throwErrorAt stx "unsupported JAXPR atom"

private def varFromSyntax (stx : Syntax) : TermElabM Expr := do
  match stx with
  | `(jaxpr_var| $id:ident) =>
      return mkAtomVar id.getId.toString none
  | `(jaxpr_var| $id:ident : $ty:jaxpr_type) =>
      return mkAtomVar id.getId.toString (some (typeFromSyntax ty))
  | _ =>
      throwErrorAt stx "expected a JAXPR variable"

private def flattenArgs (args : Array Syntax) : Array Syntax :=
  if args.size == 1 && args[0]!.getKind == nullKind then
    args[0]!.getArgs
  else
    args

private def atomsFromArgs (args : Array Syntax) : TermElabM (List Expr) := do
  let flat := flattenArgs args
  flat.foldlM (init := ([] : List Expr)) fun acc atom => do
    let e ← atomFromSyntax atom
    return acc.concat e

elab_rules : term
  | `([jaxpr| $[$lines:jaxpr_line]* ]) => do
      let mut invars : List Expr := []
      let mut outvars : List Expr := []
      let mut eqns : List Expr := []
      for line in lines do
        let lineRaw := line.raw
        if lineRaw.isOfKind ``jaxpr_line_in then
          -- Structure: [in_token, null_node_with_atoms]
          let args := lineRaw.getArgs
          if args.size < 2 then
            throwErrorAt line "expected atoms after 'in'"
          let atomsExprs ← atomsFromArgs #[args[1]!]
          invars := invars ++ atomsExprs
        else if lineRaw.isOfKind ``jaxpr_line_out then
          -- Structure: [out_token, null_node_with_atoms]
          let args := lineRaw.getArgs
          if args.size < 2 then
            throwErrorAt line "expected atoms after 'out'"
          let atomsExprs ← atomsFromArgs #[args[1]!]
          outvars := outvars ++ atomsExprs
        else if lineRaw.isOfKind ``jaxpr_line_let then
          -- Structure: [let_token, jaxpr_var, :=_token, ident, null_node_with_atoms]
          let args := lineRaw.getArgs
          if args.size < 5 then
            throwErrorAt line "expected a JAXPR let line"
          let outStx := args[1]!     -- jaxpr_var (after "let")
          let primStx := args[3]!    -- ident (after ":=")
          let atomsNode := args[4]!  -- null node with atoms
          let outExpr ← varFromSyntax outStx
          let primExpr : Expr := toExpr primStx.getId.toString
          let argExprs ← atomsFromArgs #[atomsNode]
          let argsList ← mkListLit (mkConst ``Atom) argExprs
          let eqnExpr := mkApp3 (mkConst ``Eqn.mk) outExpr primExpr argsList
          eqns := eqns.concat eqnExpr
        else
          throwErrorAt line "unsupported JAXPR line"
      let inList ← mkListLit (mkConst ``Atom) invars
      let outList ← mkListLit (mkConst ``Atom) outvars
      let eqnList ← mkListLit (mkConst ``Eqn) eqns
      return mkApp3 (mkConst ``Jaxpr.mk) inList eqnList outList

end SciLean.ML.Jaxpr
