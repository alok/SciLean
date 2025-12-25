import SciLean.Monad.TensorM
import SciLean.Monad.GPU

namespace SciLean

/-!
# TensorMGPU

GPU-backed specialization of {name}`TensorMT` with helpers that run inside
the {name}`GPU` monad and optionally escape to {name}`IO` via {name}`GPU.exec`.
-/

abbrev TensorMGPU := TensorMT GPU

namespace TensorMGPU

def defaultCaps : LayoutCaps := TensorM.defaultCaps
def defaultPolicy : LayoutPolicy := TensorM.defaultPolicy
def defaultStats : LayoutStats := TensorM.defaultStats

@[inline]
def run (caps : LayoutCaps) (policy : LayoutPolicy) (stats : LayoutStats) (x : TensorMGPU α) :
    GPU (Except LayoutError (α × LayoutStats)) :=
  ExceptT.run <| (StateT.run (ReaderT.run (ReaderT.run x caps) policy) stats)

@[inline]
def eval (caps : LayoutCaps) (policy : LayoutPolicy) (stats : LayoutStats) (x : TensorMGPU α) :
    GPU (Except LayoutError α) := do
  let r ← run caps policy stats x
  return r.map (·.1)

@[inline]
def exec (caps : LayoutCaps) (policy : LayoutPolicy) (stats : LayoutStats) (x : TensorMGPU α) :
    GPU (Except LayoutError LayoutStats) := do
  let r ← run caps policy stats x
  return r.map (·.2)

@[inline]
def runDefault (x : TensorMGPU α) : GPU (Except LayoutError α) :=
  eval defaultCaps defaultPolicy defaultStats x

@[inline]
def runIO (caps : LayoutCaps) (policy : LayoutPolicy) (stats : LayoutStats) (x : TensorMGPU α) :
    IO (Except LayoutError (α × LayoutStats)) :=
  GPU.exec (run caps policy stats x)

@[inline]
def runDefaultIO (x : TensorMGPU α) : IO (Except LayoutError α) :=
  GPU.exec (runDefault x)

end TensorMGPU

end SciLean
