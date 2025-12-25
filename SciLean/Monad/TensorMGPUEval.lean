import SciLean.Monad.TensorMGPU
import SciLean.VersoPrelude

namespace SciLean

/-!
# TensorMGPUEval

{name}`MonadEval` instance for {lit}`TensorMGPU`, using {name}`GPU.exec`
with default layout settings.
-/

private def layoutErrorMessage (err : LayoutError) : String :=
  s!"TensorMGPU failed: {reprStr err}"

namespace TensorMGPU

def evalOrThrow (x : TensorMGPU α) : IO α := do
  match (← runDefaultIO x) with
  | .ok a => pure a
  | .error err => throw <| IO.userError (layoutErrorMessage err)

end TensorMGPU

instance : MonadEval TensorMGPU IO where
  monadEval := TensorMGPU.evalOrThrow

end SciLean
