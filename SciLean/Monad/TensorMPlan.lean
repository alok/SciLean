import Mathlib.Control.Monad.Writer
import SciLean.Monad.TensorM
import SciLean.VersoPrelude

namespace SciLean

/-!
# TensorMPlan

This module defines a layout-planning variant of {name}`TensorMT` that
records layout events in a log while running without {name}`IO`.
-/

/-- Kinds of layout events emitted during planning. -/
inductive LayoutEventKind where
  | viewHit
  | copy
  | note
  deriving Repr, Inhabited

/-- A single layout event recorded during planning. -/
structure LayoutEvent where
  op : String
  kind : LayoutEventKind
  bytes : Nat := 0
  note? : Option String := none
  deriving Repr, Inhabited

/-- Layout event log collected during planning. -/
abbrev LayoutLog := List LayoutEvent

/-- Pure layout-planning monad that records a {name}`LayoutLog`. -/
abbrev TensorMPlan := TensorMT (WriterT LayoutLog Id)

namespace TensorMPlan

def defaultCaps : LayoutCaps := TensorM.defaultCaps
def defaultPolicy : LayoutPolicy := TensorM.defaultPolicy
def defaultStats : LayoutStats := TensorM.defaultStats

@[inline]
def run (caps : LayoutCaps) (policy : LayoutPolicy) (stats : LayoutStats) (x : TensorMPlan α) :
    Except LayoutError (α × LayoutStats) × LayoutLog :=
  WriterT.run <| ExceptT.run <|
    StateT.run (ReaderT.run (ReaderT.run x caps) policy) stats

@[inline]
def eval (caps : LayoutCaps) (policy : LayoutPolicy) (stats : LayoutStats) (x : TensorMPlan α) :
    Except LayoutError α × LayoutLog :=
  let (r, log) := run caps policy stats x
  (r.map (·.1), log)

@[inline]
def exec (caps : LayoutCaps) (policy : LayoutPolicy) (stats : LayoutStats) (x : TensorMPlan α) :
    Except LayoutError LayoutStats × LayoutLog :=
  let (r, log) := run caps policy stats x
  (r.map (·.2), log)

@[inline]
def runDefault (x : TensorMPlan α) : Except LayoutError α × LayoutLog :=
  eval defaultCaps defaultPolicy defaultStats x

@[inline] def getCaps : TensorMPlan LayoutCaps := TensorMT.getCaps
@[inline] def getPolicy : TensorMPlan LayoutPolicy := TensorMT.getPolicy
@[inline] def withCaps (caps : LayoutCaps) (x : TensorMPlan α) : TensorMPlan α := TensorMT.withCaps caps x
@[inline] def withPolicy (policy : LayoutPolicy) (x : TensorMPlan α) : TensorMPlan α := TensorMT.withPolicy policy x
@[inline] def require (ok : Bool) (err : LayoutError) : TensorMPlan Unit := TensorMT.require ok err
@[inline] def withStats (x : TensorMPlan α) : TensorMPlan (α × LayoutStats) := TensorMT.withStats x

@[inline]
def log (ev : LayoutEvent) : TensorMPlan Unit :=
  liftM (m := WriterT LayoutLog Id) ((PUnit.unit, [ev]) : WriterT LayoutLog Id PUnit)

@[inline]
def recordViewHit (op : String) : TensorMPlan Unit := do
  TensorMT.recordViewHit
  log { op := op, kind := .viewHit }

@[inline]
def recordCopy (op : String) (bytes : Nat) : TensorMPlan Unit := do
  TensorMT.recordCopy bytes
  log { op := op, kind := .copy, bytes := bytes }

@[inline]
def note (op : String) (msg : String) : TensorMPlan Unit :=
  log { op := op, kind := .note, note? := some msg }

end TensorMPlan

end SciLean
