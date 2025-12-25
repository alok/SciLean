import SciLean.Monad.TensorM

namespace SciLean

/-!
# TensorMPure

{lit}`TensorMPure` is a pure specialization of {name}`TensorMT` for layout planning
and testing without {name}`IO`. It preserves layout policies, statistics, and
errors while running in {name}`Id`.
-/

/-- Pure specialization of {name}`TensorMT` without {name}`IO`. -/
abbrev TensorMPure := TensorMT Id

namespace TensorMPure

def defaultCaps : LayoutCaps := TensorM.defaultCaps
def defaultPolicy : LayoutPolicy := TensorM.defaultPolicy
def defaultStats : LayoutStats := TensorM.defaultStats

@[inline]
def run (caps : LayoutCaps) (policy : LayoutPolicy) (stats : LayoutStats) (x : TensorMPure α) :
    Except LayoutError (α × LayoutStats) :=
  ExceptT.run <| (StateT.run (ReaderT.run (ReaderT.run x caps) policy) stats)

@[inline]
def eval (caps : LayoutCaps) (policy : LayoutPolicy) (stats : LayoutStats) (x : TensorMPure α) :
    Except LayoutError α :=
  (run caps policy stats x).map (·.1)

@[inline]
def exec (caps : LayoutCaps) (policy : LayoutPolicy) (stats : LayoutStats) (x : TensorMPure α) :
    Except LayoutError LayoutStats :=
  (run caps policy stats x).map (·.2)

@[inline]
def runDefault (x : TensorMPure α) : Except LayoutError α :=
  eval defaultCaps defaultPolicy defaultStats x

@[inline] def getCaps : TensorMPure LayoutCaps := TensorMT.getCaps
@[inline] def getPolicy : TensorMPure LayoutPolicy := TensorMT.getPolicy
@[inline] def withCaps (caps : LayoutCaps) (x : TensorMPure α) : TensorMPure α := TensorMT.withCaps caps x
@[inline] def withPolicy (policy : LayoutPolicy) (x : TensorMPure α) : TensorMPure α := TensorMT.withPolicy policy x
@[inline] def recordViewHit : TensorMPure Unit := TensorMT.recordViewHit
@[inline] def recordCopy (bytes : Nat) : TensorMPure Unit := TensorMT.recordCopy bytes
@[inline] def require (ok : Bool) (err : LayoutError) : TensorMPure Unit := TensorMT.require ok err
@[inline] def withStats (x : TensorMPure α) : TensorMPure (α × LayoutStats) := TensorMT.withStats x

end TensorMPure

end SciLean
