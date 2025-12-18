namespace SciLean

/-- {name}`holdLet` is just identity function with a special support from some tactics.

Tactics like {name}`autodiff`, {name}`lsimp`, {name}`lfun_trans` inline let bindings of functions, for examples
```lean
let f := fun x => x*x
f 2
```
would get simplified to {lit}`2*2` even with option {lit}`zeta:=false`. This reduction is important for
reverse mode AD.

Function {name}`holdLet` is useful for preventing this reduction, so adding {name}`holdLet`
```lean
let f := holdLet <| fun x => x*x
f 2
```
will prevent {name}`lsimp` to remove the let.  -/
@[inline]
def holdLet {α : Type u} (a : α) := a
