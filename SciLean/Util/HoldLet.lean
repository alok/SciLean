import Verso.Code.External
import SciLean.VersoPrelude

namespace SciLean

/-- {lit}``holdLet`` is just identity function with a special support from some tactics.

Tactics like {lit}``autodiff``, {lit}``lsimp``, {lit}``lfun_trans`` inline let bindings of functions.
For example, {lit}``let f := fun x => x*x; f 2`` would get simplified to {lit}``2*2`` even with
option {lit}``zeta:=false``. This reduction is important for reverse mode AD.

Function {lit}``holdLet`` is useful for preventing this reduction. Adding {lit}``holdLet``:
{lit}``let f := holdLet <| fun x => x*x; f 2`` will prevent {lit}``lsimp`` from removing the let.
-/
@[inline]
def holdLet {α : Type u} (a : α) := a
