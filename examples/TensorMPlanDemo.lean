import SciLean.Monad.TensorMPlan

open SciLean

private def fakeOp (transposed : Bool) : TensorMPlan Unit := do
  let caps ← TensorMPlan.getCaps
  let policy ← TensorMPlan.getPolicy
  if policy.preferViews && caps.acceptsTransposed && transposed then
    TensorMPlan.recordViewHit "fakeOp"
  else if policy.allowCopy then
    TensorMPlan.recordCopy "fakeOp" 128
  else
    throw (LayoutError.copyDisallowed "fakeOp")

#eval
  let caps : LayoutCaps := { acceptsTransposed := true }
  let policy : LayoutPolicy := { preferViews := true, allowCopy := true }
  let (res, log) := TensorMPlan.run caps policy {} (fakeOp true)
  s!"view: res={reprStr res} log={reprStr log}"

#eval
  let caps : LayoutCaps := { acceptsTransposed := false }
  let policy : LayoutPolicy := { preferViews := true, allowCopy := true }
  let (res, log) := TensorMPlan.run caps policy {} (fakeOp true)
  s!"copy: res={reprStr res} log={reprStr log}"

#eval
  let caps : LayoutCaps := { acceptsTransposed := false }
  let policy : LayoutPolicy := { preferViews := true, allowCopy := false }
  let (res, log) := TensorMPlan.run caps policy {} (fakeOp true)
  s!"error: res={reprStr res} log={reprStr log}"
