import SciLean.Modules.ML.Jaxpr.AST
import SciLean.Modules.ML.XLA.DotGeneral
import SciLean.Modules.ML.XLA.Transpose
import SciLean.Modules.ML.XLA.Concatenate
import SciLean.Modules.ML.XLA.Slice
import SciLean.Modules.ML.XLA.Pad
import SciLean.Modules.ML.XLA.Convolution
import SciLean.Modules.ML.XLA.Split
import SciLean.VersoPrelude

namespace SciLean.ML.Jaxpr

/-!
# Lowering JAXPR to XLA primitives

This module provides a tiny lowering pass that maps JAXPR primitives
onto existing XLA modules (StableHLO semantics).
-/

/-- XLA primitives we can lower to. -/
inductive XlaPrim where
  | dot_general
  | transpose
  | concatenate
  | slice
  | pad
  | convolution
  | split
  deriving Repr, BEq, Inhabited

/-- Resolve the target XLA namespace for a primitive. -/
@[inline]
def XlaPrim.moduleName : XlaPrim → Lean.Name
  | .dot_general => Lean.Name.str (Lean.Name.str Lean.Name.anonymous "SciLean") "DotGeneral"
  | .transpose => Lean.Name.str (Lean.Name.str Lean.Name.anonymous "SciLean") "Transpose"
  | .concatenate => Lean.Name.str (Lean.Name.str Lean.Name.anonymous "SciLean") "Concatenate"
  | .slice => Lean.Name.str (Lean.Name.str Lean.Name.anonymous "SciLean") "Slice"
  | .pad => Lean.Name.str (Lean.Name.str Lean.Name.anonymous "SciLean") "Pad"
  | .convolution => Lean.Name.str (Lean.Name.str Lean.Name.anonymous "SciLean") "Convolution"
  | .split => Lean.Name.str (Lean.Name.str Lean.Name.anonymous "SciLean") "Split"

/-- Metadata about a primitive. Arity {lit}`0` means variadic. -/
structure PrimInfo where
  name : String
  arity : Nat
  xla? : Option XlaPrim
  deriving Repr, Inhabited

/-- Registry of known primitives. -/
@[inline]
def primRegistry : List PrimInfo :=
  [
    ⟨"dot_general", 2, some .dot_general⟩,
    ⟨"transpose", 1, some .transpose⟩,
    ⟨"concatenate", 2, some .concatenate⟩,
    ⟨"slice", 1, some .slice⟩,
    ⟨"pad", 1, some .pad⟩,
    ⟨"convolution", 2, some .convolution⟩,
    ⟨"split", 1, some .split⟩,
    -- Common elementwise ops (not lowered yet)
    ⟨"add", 2, none⟩,
    ⟨"mul", 2, none⟩,
    ⟨"sub", 2, none⟩
  ]

/-- Lookup a primitive in the registry. -/
@[inline]
def primInfo? (name : String) : Option PrimInfo :=
  primRegistry.find? (fun p => p.name == name)

/-- Lowered XLA equation. -/
structure XlaEqn where
  outVar : Atom
  prim : XlaPrim
  args : List Atom
  deriving Repr, BEq, Inhabited

/-- Lowered XLA program. -/
structure XlaProgram where
  invars : List Atom
  eqns : List XlaEqn
  outvars : List Atom
  deriving Repr, BEq, Inhabited

private def ensureVar (a : Atom) : Except String Atom :=
  match a with
  | .var _ _ => .ok a
  | .lit _ _ => .error "equation output must be a variable"

/-- Lower a single equation if the primitive is supported. -/
def lowerEqn (e : Eqn) : Except String XlaEqn := do
  let outVar ← ensureVar e.out
  let some info := primInfo? e.prim
    | throw s!"unknown primitive: {e.prim}"
  if info.arity != 0 && e.args.length != info.arity then
    throw s!"arity mismatch for {e.prim}: expected {info.arity}, got {e.args.length}"
  let some prim := info.xla?
    | throw s!"primitive not lowered yet: {e.prim}"
  return ⟨outVar, prim, e.args⟩

/-- Lower a full JAXPR to XLA primitives (best-effort, fails on unknown ops). -/
def lower (j : Jaxpr) : Except String XlaProgram := do
  let mut eqns : List XlaEqn := []
  for e in j.eqns do
    eqns := eqns.concat (← lowerEqn e)
  return ⟨j.invars, eqns, j.outvars⟩

end SciLean.ML.Jaxpr
