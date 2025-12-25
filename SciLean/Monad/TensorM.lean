import Init.Control.Reader
import Init.Control.State
import Init.Control.Except
import SciLean.VersoPrelude

namespace SciLean

/-!
# TensorM

{lit}`TensorM` is a small monad stack for layout-aware tensor execution.
It combines:
- {name}`ReaderT` for layout capabilities and policy knobs.
- {name}`StateT` for copy/view statistics.
- {name}`ExceptT` for structured layout errors.
- {name}`IO` for FFI and device operations.

This keeps layout policy pluggable and makes it easy to run the same code
with different backends or debugging settings.
-/

structure LayoutCaps where
  /-- Kernel accepts transposed views. -/
  acceptsTransposed : Bool := true
  /-- Kernel accepts arbitrary strides. -/
  acceptsStride : Bool := false
  /-- Kernel accepts non-zero buffer offsets. -/
  acceptsOffset : Bool := false
  deriving Repr, Inhabited

structure LayoutPolicy where
  /-- Prefer using views when possible. -/
  preferViews : Bool := true
  /-- Allow materializing a contiguous copy if needed. -/
  allowCopy : Bool := true
  deriving Repr, Inhabited

structure LayoutStats where
  /-- Number of materializations performed. -/
  copies : Nat := 0
  /-- Total bytes copied. -/
  bytesCopied : Nat := 0
  /-- Number of fast-path view hits. -/
  viewHits : Nat := 0
  deriving Repr, Inhabited

inductive LayoutError where
  | unsupportedLayout (op : String)
  | copyDisallowed (op : String)
  | kernelUnavailable (op : String)
  deriving Repr, Inhabited

/-- Layout-aware tensor monad transformer. -/
abbrev TensorMT (m : Type → Type) :=
  ReaderT LayoutCaps (ReaderT LayoutPolicy (StateT LayoutStats (ExceptT LayoutError m)))

/-- Layout-aware tensor monad specialized to {name}`IO`. -/
abbrev TensorM := TensorMT IO

namespace TensorMT

variable {m : Type → Type} [Monad m]

@[inline]
def getCaps : TensorMT m LayoutCaps :=
  fun caps => fun _policy => pure caps

@[inline]
def getPolicy : TensorMT m LayoutPolicy :=
  fun _caps => fun policy => pure policy

@[inline]
def withCaps (caps : LayoutCaps) (x : TensorMT m α) : TensorMT m α :=
  fun _ => fun policy => x caps policy

@[inline]
def withPolicy (policy : LayoutPolicy) (x : TensorMT m α) : TensorMT m α :=
  fun caps => fun _ => x caps policy

@[inline]
def recordViewHit : TensorMT m Unit := do
  modify fun s => { s with viewHits := s.viewHits + 1 }

@[inline]
def recordCopy (bytes : Nat) : TensorMT m Unit := do
  modify fun s => { s with copies := s.copies + 1, bytesCopied := s.bytesCopied + bytes }

@[inline]
def require (ok : Bool) (err : LayoutError) : TensorMT m Unit :=
  if ok then pure () else throw err

@[inline]
def withStats (x : TensorMT m α) : TensorMT m (α × LayoutStats) := do
  let a ← x
  let s ← get
  return (a, s)

@[inline]
def liftIO [MonadLift IO m] (x : IO α) : TensorMT m α :=
  liftM x

end TensorMT

namespace TensorM

def defaultCaps : LayoutCaps := { acceptsStride := true, acceptsOffset := true }
def defaultPolicy : LayoutPolicy := {}
def defaultStats : LayoutStats := {}

@[inline]
def run (caps : LayoutCaps) (policy : LayoutPolicy) (stats : LayoutStats) (x : TensorM α) :
    IO (Except LayoutError (α × LayoutStats)) :=
  ExceptT.run <| (StateT.run (ReaderT.run (ReaderT.run x caps) policy) stats)

@[inline]
def eval (caps : LayoutCaps) (policy : LayoutPolicy) (stats : LayoutStats) (x : TensorM α) :
    IO (Except LayoutError α) := do
  let r ← run caps policy stats x
  return r.map (·.1)

@[inline]
def exec (caps : LayoutCaps) (policy : LayoutPolicy) (stats : LayoutStats) (x : TensorM α) :
    IO (Except LayoutError LayoutStats) := do
  let r ← run caps policy stats x
  return r.map (·.2)

@[inline]
def runDefault (x : TensorM α) : IO (Except LayoutError α) :=
  eval defaultCaps defaultPolicy defaultStats x

@[inline] def getCaps : TensorM LayoutCaps := TensorMT.getCaps
@[inline] def getPolicy : TensorM LayoutPolicy := TensorMT.getPolicy
@[inline] def withCaps (caps : LayoutCaps) (x : TensorM α) : TensorM α := TensorMT.withCaps caps x
@[inline] def withPolicy (policy : LayoutPolicy) (x : TensorM α) : TensorM α := TensorMT.withPolicy policy x
@[inline] def recordViewHit : TensorM Unit := TensorMT.recordViewHit
@[inline] def recordCopy (bytes : Nat) : TensorM Unit := TensorMT.recordCopy bytes
@[inline] def require (ok : Bool) (err : LayoutError) : TensorM Unit := TensorMT.require ok err
@[inline] def withStats (x : TensorM α) : TensorM (α × LayoutStats) := TensorMT.withStats x
@[inline]
def liftIO (x : IO α) : TensorM α :=
  fun _caps _policy => StateT.lift (ExceptT.lift x)

end TensorM

end SciLean
