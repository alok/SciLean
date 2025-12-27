import SciLean.Monad.TensorM
import SciLean.Monad.TensorMPure
import SciLean.VersoPrelude

namespace SciLean

/-!
# TensorMEval

Helpers to evaluate {lit}`TensorM` and {lit}`TensorMPure` programs via
{name}`MonadEval`. This makes {lit}`#eval` usable with layout-aware code.
-/

private def layoutErrorMessage (err : LayoutError) : String :=
  s!"TensorM failed: {reprStr err}"

namespace TensorM

def evalOrThrow (x : TensorM α) : IO α := do
  match (← runDefault x) with
  | .ok a => pure a
  | .error err => throw <| IO.userError (layoutErrorMessage err)

end TensorM

namespace TensorMPure

def evalIO (x : TensorMPure α) : IO α := do
  match runDefault x with
  | .ok a => pure a
  | .error err => throw <| IO.userError (layoutErrorMessage err)

end TensorMPure

instance : MonadEval TensorM IO where
  monadEval := TensorM.evalOrThrow

instance : MonadEval TensorMPure IO where
  monadEval := TensorMPure.evalIO

end SciLean
