import SciLean.Data.Tensor.Ops

open SciLean

set_default_scalar Float

/-- Minimal profile for {lit}`TensorOps` on CPU tensors. -/
def main : IO Unit := do
  let a : Float^[Idx 4] := ⊞ (i : Idx 4) => (i.1.toNat + 1).toFloat
  let b : Float^[Idx 4] := ⊞ (i : Idx 4) => (2 * i.1.toNat).toFloat

  let aT : CpuTensor Float (Idx 4) := a
  let bT : CpuTensor Float (Idx 4) := b

  let c := TensorOps.add aT bT
  let d := TensorOps.mul aT bT

  IO.println s!"TensorOps.add (cpu): {c.data}"
  IO.println s!"TensorOps.mul (cpu): {d.data}"
