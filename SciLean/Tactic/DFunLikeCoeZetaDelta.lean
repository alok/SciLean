import SciLean.Meta.SimpAttr

import Mathlib.Data.FunLike.Basic -- this import does not seem to be enough
import Mathlib.Logic.Equiv.Defs
import SciLean.VersoPrelude

namespace Scilean

open Lean Meta in
/-- Zeta delta reduction for bundled morphisms/{name}`DFunLike.coe`.

Expressions of the form {lit}`DFunLike.coe fVar x` where {lit}`fVar` is a free variable with value {lit}`fVal` are
replaced with {lit}`DFunLike.coe fVal x`.

For examples, {lit}`let f : R →*+ R := Ring.id; ⇑f x` reduces to {lit}`⇑Ring.id x`.
-/
simproc_decl dfunlike_coe_zetaDelta (DFunLike.coe _ _) := fun e => do
  let x := e.appArg!
  let .fvar fId := e.appFn!.appArg! | return .continue
  let .some f ← fId.getValue? | return .continue
  let coe := e.appFn!.appFn!
  return .visit { expr := coe.app f |>.app x }

attribute [simp, simp_core_proc] dfunlike_coe_zetaDelta
