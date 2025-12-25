import Std.Tactic
import SciLean
import SciLean.Modules.ML.Jaxpr
import tensor_basic

/-!
Smoke tests for SciLean.

These are intentionally compilation-only (no `#eval`) so they work on macOS
even when module precompilation is disabled (to avoid building `Mathlib:shared`).
-/

open SciLean

-- A couple of basic "does it elaborate?" checks.
#check SciLean.Numerics.ODE.rungeKutta4
#check SciLean.Numerics.ODE.heunMethod
#check SciLean.Numerics.ODE.explicitMidpoint

-- Tensor types
#check Device
#check CpuTensor
#check GpuTensor

-- JAXPR DSL elaboration + round-trip toString
open SciLean.ML.Jaxpr in
def smokeJaxpr : Jaxpr := [jaxpr|
  in x:f32 y:f32
  let z:f32 := add x:f32 y:f32
  out z:f32
]

example : smokeJaxpr.toString = "in x:f32 y:f32\nz:f32 = add x:f32 y:f32\nout z:f32" := by
  native_decide
