import Mathlib.Algebra.Group.Defs
import SciLean.Logic.Function.Constant
import SciLean.VersoPrelude

namespace SciLean.UnsafeAD

axiom unsafeNonzero {α} [Zero α] (a : α) : a ≠ 0
axiom unsafeIsConstant {α β} (f : α → β) : f.IsConstant

/-- This tactic is used to ignore common assumptions of differentiation rules when used as discharger
in {lit}`autodiff` tactic.

Most notably, it assumes that every number is non-zero thus you can freely differentiate through
division, norm, logarithm etc.

TODO: Add differentiation rules for if statement where the condition depends on the value and
      make this tactic compatible with this rule.
-/
macro "unsafeAD" : tactic =>
  `(tactic| (intros; simp only [not_false_eq_true, ne_eq, unsafeNonzero, unsafeIsConstant]))
