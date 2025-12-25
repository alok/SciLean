import SciLean.Modules.ML.Jaxpr

open SciLean.ML.Jaxpr

/-- Minimal JAXPR demo. -/
def demo : Jaxpr := [jaxpr|
  in x:f32 y:f32
  let z:f32 := add x:f32 y:f32
  let w:f32 := mul z:f32 y:f32
  out w:f32
]

#eval demo.toString
#eval (lower demo)
