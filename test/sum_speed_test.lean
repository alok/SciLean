import SciLean

open SciLean

/-- Minimal loop benchmark for the {lit}`ForLoopTest` executable. -/
def sumNat (n : Nat) : Nat := Id.run do
  let mut acc := 0
  for i in [0:n] do
    acc := acc + i
  return acc

def main : IO Unit := do
  let n := 1000000
  let s := sumNat n
  IO.println s!"sum_speed_test: n={n}, sum={s}"
