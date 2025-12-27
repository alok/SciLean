import SciLean

open SciLean

set_default_scalar Float

/-- Simple matrix/vector sanity check using {lit}`Float^[Idx n]` arrays. -/
def main : IO Unit := do
  let v : Float^[Idx 3] := ⊞ (i : Idx 3) => (i.1.toNat + 1).toFloat
  let m : Float^[Idx 3, Idx 3] := ⊞ (i : Idx 3) (j : Idx 3) => if i == j then 1.0 else 0.0
  let w : Float^[Idx 3] := ⊞ (i : Idx 3) => ∑ᴵ (j : Idx 3), m[i,j] * v[j]

  IO.println s!"v = {v}"
  IO.println s!"m = {m}"
  IO.println s!"m*v = {w}"
