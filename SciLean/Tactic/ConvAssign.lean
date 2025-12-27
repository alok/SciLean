import SciLean.VersoPrelude
namespace SciLean.Tactic

/-- Assuming that the goal is a metavariable, {lit}`assign t` assigns term {lit}`t` to that metavariable. -/
macro (name := Conv.assign) "assign " t:term : conv =>
  `(conv| tactic => exact (rfl : (_ = $t)))
