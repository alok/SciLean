import Lean
import SciLean.VersoPrelude

namespace SciLean.ML.Jaxpr

/-!
# JAXPR AST

Lightweight data structures for a JAXPR-like intermediate representation.

- {lit}`Atom` represents a variable or a literal (stored as a {lit}`String`).
- {lit}`Eqn` represents one primitive application.
- {lit}`Jaxpr` is a list of equations with explicit inputs and outputs.
-/

/-- A JAXPR atom: variable or literal, optionally typed. -/
inductive Atom where
  | var (name : String) (ty? : Option String := none)
  | lit (value : String) (ty? : Option String := none)
  deriving Repr, BEq, Inhabited

/-- A JAXPR equation: {lit}`out = prim args...`. -/
structure Eqn where
  out : Atom
  prim : String
  args : List Atom
  deriving Repr, BEq, Inhabited

/-- A JAXPR program: inputs, equations, outputs. -/
structure Jaxpr where
  invars : List Atom
  eqns : List Eqn
  outvars : List Atom
  deriving Repr, BEq, Inhabited

@[inline]
private def Atom.withType (base : String) (ty? : Option String) : String :=
  match ty? with
  | none => base
  | some ty => s!"{base}:{ty}"

/-- Render an atom as text. -/
@[inline]
def Atom.toString : Atom → String
  | .var name ty? => Atom.withType name ty?
  | .lit value ty? => Atom.withType value ty?

instance : ToString Atom := ⟨Atom.toString⟩

/-- Render an equation as {lit}`out = prim args...`. -/
@[inline]
def Eqn.toString (e : Eqn) : String :=
  let out := e.out.toString
  let args := e.args.map (·.toString) |>.intersperse " " |>.foldl (· ++ ·) ""
  if args.isEmpty then
    s!"{out} = {e.prim}"
  else
    s!"{out} = {e.prim} {args}".trimAsciiEnd.toString

instance : ToString Eqn := ⟨Eqn.toString⟩

/-- Render a full JAXPR as lines. -/
@[inline]
def Jaxpr.toString (j : Jaxpr) : String :=
  let inVars := j.invars.map (·.toString) |>.intersperse " " |>.foldl (· ++ ·) ""
  let outVars := j.outvars.map (·.toString) |>.intersperse " " |>.foldl (· ++ ·) ""
  let eqnLines := j.eqns.map (·.toString) |>.intersperse "\n" |>.foldl (· ++ ·) ""
  let header := s!"in {inVars}".trimAsciiEnd.toString
  let footer := s!"out {outVars}".trimAsciiEnd.toString
  if eqnLines.isEmpty then
    s!"{header}\n{footer}"
  else
    s!"{header}\n{eqnLines}\n{footer}"

instance : ToString Jaxpr := ⟨Jaxpr.toString⟩

end SciLean.ML.Jaxpr
