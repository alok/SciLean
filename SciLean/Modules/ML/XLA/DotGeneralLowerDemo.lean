import SciLean.Modules.ML.XLA.DotGeneral
import SciLean.Modules.ML.Jaxpr.Elab
import SciLean.Modules.ML.Jaxpr.Lower
import SciLean.Util.SorryProof
import SciLean.VersoPrelude

namespace SciLean

namespace DotGeneral

open ML.Jaxpr

def demoJaxpr : Jaxpr :=
  [jaxpr|
    in lhs rhs
    let outv := dot_general lhs rhs
    out outv
  ]

def demoLower : Except String XlaProgram :=
  ML.Jaxpr.lower demoJaxpr

def demoLhsDims : Dims 2 := ⟨#[2, 3], by decide⟩
def demoRhsDims : Dims 2 := ⟨#[3, 4], by decide⟩
def demoArgs : Args demoLhsDims demoRhsDims :=
  { lhs_batching_dimensions := #[]
    rhs_batching_dimensions := #[]
    lhs_contracting_dimensions := #[1]
    rhs_contracting_dimensions := #[0] }

def demoOutDims : Dims 2 := ⟨#[2, 4], by decide⟩

def demoPreconditions : Preconditions demoArgs demoOutDims := by
  sorry_proof

end DotGeneral

end SciLean
