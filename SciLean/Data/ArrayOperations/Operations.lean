import SciLean.Data.ArrayOperations.Basic
import SciLean.Data.IndexType
import SciLean.Data.IndexType.Basic
import SciLean.Data.IndexType.Fold
import SciLean.VersoPrelude

namespace SciLean

namespace ArrayOps

variable {X I Y : Type*} {nI} [IndexType I nI] [Fold I]
  [GetElem' X I Y]
  [SetElem' X I Y]

/--
Maps elements of {lit}`xs` by {lit}`f` with data accessor {lit}`g`.

For {lit}`i`, {lit}`(mapIdxMonoAcc f g xs)[i] = f i (g i) xs[i]`.

This is a low level function that provides additional argument {lit}`g` which is used as data accessor
inside of {lit}`f`. For example, instead of writing

{lit}`def add (x y : X) := mapIdxMono (fun i xi => xi + y[i]) x`

you should write

{lit}`def add (x y : X) := mapIdxMonoAcc (fun i yi xi => xi + yi) (fun i => y[i]) x`

This way reverse mode AD can produce better code.

An example of higher arity function:

{lit}`def mulAdd (x y z : X) := mapIdxMonoAcc (fun i (xi,yi) zi => xi*yi + zi) (fun i => (x[i],y[i])) z`
-/
@[inline, specialize, macro_inline]
def mapIdxMonoAcc (f : I → Z → Y → Y) (g : I → Z) (xs : X) : X :=
  IndexType.fold (init:=xs) .full (fun (i : I) xs  =>
    let xi := xs[i]
    let yi := g i
    let xi' := f i (g i) xi
    setElem xs i xi' .intro)


/--
Maps elements of {lit}`xs` by {lit}`f`.

For {lit}`i`, {lit}`(mapIdxMono f xs)[i] = f i xs[i]`.

Note: Consider using {name}`mapIdxMonoAcc` if {lit}`f` is accessing element of another array,
like {lit}`f := fun i xi => xi + y[i]`. Reverse mode AD is able to produce better gradients for
{name}`mapIdxMonoAcc`.
-/
@[reducible, inline, specialize, macro_inline]
def mapIdxMono (f : I → Y → Y) (xs : X) : X :=
  mapIdxMonoAcc (fun i _ y => f i y) (fun _ => ()) xs


/--
Maps elements of {lit}`xs` by {lit}`f`.

For {lit}`i`, {lit}`(mapMono f xs)[i] = f xs[i]`.
-/
@[reducible, inline, specialize, macro_inline]
def mapMono [DefaultIndex X I] (f : Y → Y) (xs : X) : X :=
  mapIdxMono (fun _ : I => f) xs
