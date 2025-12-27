import Mathlib.Tactic.ProxyType
import SciLean.Data.DataArray.PlainDataType

namespace SciLean
namespace PlainDataType

-- Derive handler for PlainDataType using Mathlib's proxy_equiv

macro "derive_plaindatatype%" t:term : term =>
  `(term| PlainDataType.ofEquiv (proxy_equiv% $t))

open Lean Elab Command in
def mkPlainDataType (declName : Name) : CommandElabM Bool := do
  let indVal ← getConstInfoInduct declName
  let cmd ← liftTermElabM do
    let header ← Deriving.mkHeader `PlainDataType 0 indVal
    let binders' ← Deriving.mkInstImplicitBinders ``PlainDataType indVal header.argNames
    let instCmd ← `(command|
      instance $header.binders:bracketedBinder* $(binders'.map TSyntax.mk):bracketedBinder* :
          PlainDataType $header.targetType := derive_plaindatatype% _)
    return instCmd
  trace[Elab.Deriving.plaindatatype] "instance command:\n{cmd}"
  elabCommand cmd
  return true

open Lean Elab Command in
def mkPlainDataTypeInstanceHandler (declNames : Array Name) : CommandElabM Bool := do
  if declNames.size != 1 then
    return false
  let declName := declNames[0]!
  mkPlainDataType declName

open Lean Elab in
initialize
  registerDerivingHandler ``PlainDataType mkPlainDataTypeInstanceHandler
  registerTraceClass `Elab.Deriving.plaindatatype

end PlainDataType
end SciLean
