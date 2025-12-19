import SciLean.Data.ArrayType.Basic
import SciLean.Data.ListN
import SciLean.Data.ArrayLike
import SciLean.Lean.Meta.Basic
import Batteries.Lean.Expr

import Mathlib.LinearAlgebra.Matrix.Notation

import Qq


namespace SciLean

open Lean Parser
open TSyntax.Compat



-- Element access notation -------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------

initialize registerTraceClass `getElem_notation


/-- Turn an array of terms in into a tuple. -/
private def mkTuple (xs : Array (TSyntax `term)) : MacroM (TSyntax `term) :=
  `(term| ($(xs[0]!), $(xs[1:]),*))

/--
The syntax {lit}`x[i,j,k]` gets the element of {lit}`x` indexed by {lit}`(i, j, k)`.

Note that the product is right associated, so {lit}`x[i,j,k]`, {lit}`x[i,(j,k)]`,
and {lit}`x[(i,j,k)]` result in the same expression.
-/
macro:max (name:=indexedGet) (priority:=high+1) x:term noWs "[" i:term ", " is:term,* "]" : term => do
  let idx ← mkTuple (#[i] ++ is.getElems)
  `($x[$idx])



open Lean Elab Term Meta in
@[inherit_doc indexedGet]
elab:max (priority:=high+2) x:term noWs "[" is:term,* "]" : term => do
  try
    let rank := is.getElems.size
    let x ← elabTermAndSynthesize x none
    let X ← inferType x
    let Idx ← mkFreshTypeMVar
    let Elem ← mkFreshTypeMVar
    let Valid ← mkFreshExprMVar none
    let cls := (mkAppN (← mkConstWithFreshMVarLevels ``DefaultIndexOfRank) #[X, mkNatLit rank, Idx])
    let _ ← synthInstance cls
    trace[getElem_notation] "Default index type {Idx} of rank {rank} for {X}"
    let getElemCls := mkAppN (← mkConstWithFreshMVarLevels ``GetElem) #[X, Idx, Elem, Valid]
    let inst ← synthInstance getElemCls
    let i ← elabTerm (← liftMacroM (mkTuple is.getElems)) Idx
    return ← mkAppOptM ``getElem #[X,Idx,Elem,Valid,inst,x,i,Expr.const ``True.intro []]
  catch e =>
    let i ← liftMacroM (mkTuple is.getElems)
    trace[getElem_notation] "Failed to infer default index type with error:\n{e.toMessageData}"
    return ← elabTerm (← `(getElem $x $i (by get_elem_tactic))) none



-- /-- Element index can either be an index or a range. -/
-- syntax elemIndex := (term)? (":" (term)?)?


-- todo: merge with `indexedGet`
--       right now I could not figure out how to correctly write down the corresponding macro_rules
-- @[inherit_doc indexedGet]
-- syntax:max (name:=indexedGetRanges) (priority:=high) term noWs "[" elemIndex "," elemIndex,* "]" : term

macro (priority:=high) x:ident noWs "[" ids:term,* "]" " := " xi:term : doElem => do
  let i ← mkTuple ids.getElems
  `(doElem| $x:ident := setElem $x $i $xi (by get_elem_tactic))

macro (priority:=high) x:ident noWs "[" ids:term,* "]" " ← " xi:term : doElem => do
  let i ← mkTuple ids.getElems
  `(doElem| $x:ident := setElem $x $i (← $xi) (by get_elem_tactic))

macro x:ident noWs "[" ids:term,* "]" " += " xi:term : doElem => do
  let i ← mkTuple ids.getElems
  `(doElem| $x:ident := let xi := $x[$i]; setElem $x $i (xi + $xi) (by get_elem_tactic))

macro x:ident noWs "[" ids:term,* "]" " -= " xi:term : doElem => do
  let i ← mkTuple ids.getElems
  `(doElem| $x:ident := let xi := $x[$i]; setElem $x $i (xi - $xi) (by get_elem_tactic))

macro x:ident noWs "[" ids:term,* "]" " *= " xi:term : doElem => do
  let i ← mkTuple ids.getElems
  `(doElem| $x:ident := let xi := $x[$i]; setElem $x $i (xi * $xi) (by get_elem_tactic))

macro x:ident noWs "[" ids:term,* "]" " /= " xi:term : doElem => do
  let i ← mkTuple ids.getElems
  `(doElem| $x:ident := let xi := $x[$i]; setElem $x $i (xi / $xi) (by get_elem_tactic))

macro x:ident noWs "[" ids:term,* "]" " •= " xi:term : doElem => do
  let i ← mkTuple ids.getElems
  `(doElem| $x:ident := let xi := $x[$i]; setElem $x $i ($xi • xi) (by get_elem_tactic))


@[app_unexpander ArrayType.get] def unexpandArrayTypeGet : Lean.PrettyPrinter.Unexpander
  | `($(_) $x ($i, $is,*)) => `($x[$i:term,$[$is:term],*])
  | `($(_) $x $i) => `($x[$i:term])
  | _ => throw ()


----------------------------------------------------------------------------------------------------


-- Notation: ⊞ i => f i --
--------------------------

-- open Lean.TSyntax.Compat in
-- macro "⊞ " x:term " => " b:term:51 : term => `(introElemNotation fun $x => $b)
-- macro "⊞ " x:term " : " X:term " => " b:term:51 : term => `(introElemNotation fun ($x : $X) => $b)

initialize registerTraceClass `ofFn_notation

open Term  in
/-- Notation {lit}`⊞ i => f i` creates an array with element {lit}`f i`. It is a syntax for {name}`ofFn`.

For example:
```
variable (f : Idx n → Float) (g : Idx m → Idx n → Float)

#check ⊞ i => f i             -- Float^[n]
#check ⊞ i j => g i j         -- Float^[m,n]
#check ⊞ i => ⊞ j => g i j    -- Float^[n]^[m]
#check ⊞ i j k => g i j + f k -- Float^[m,n,n]
```
The preferd way of creating a matrix is {lit}`⊞ i j => g i j` not {lit}`⊞ (i,j) => g i j`. Both versions
work but the former notaion allows us to writh {lit}`⊞ i (j : Idx 5) => f i` instead of
{lit}`⊞ ((i,j) : _ ×Idx 4) => f i`. This is also the reson for the somewhat unexpected fact that
{lit}`⊞ i => ⊞ j => g i j ≠ ⊞ i j => g i j`.

The resulting type does not always have to be {lit}`Float^[n]` (notation for {lit}`DataArrayN Float (Idx n)`).
When we explicitely force the resulting type we can obtain different array type.
For this to work there has to be instance {lit}`OfFn coll idx elem` for {lit}`f : idx → elem`.

For example:
```
#check (⊞ i => f i : Vector Float 2) -- Vector Float 2
```

Examples of notation expansion
```
variable (f : Idx n → Float) (g : Idx m → Idx n → Float)

(⊞ i => f i)          ==> ofFn (coll:=Float^[n]) f
(⊞ i => f i : Vector Float 2) ==> ofFn (coll:=Vector Float 2) f
(⊞ i j => g i j)      ==> ofFn (coll:=Float^[m,n]) ↿g
(⊞ i => ⊞ j => g i j) ==> ofFn (coll:=Float^[n]^[m]) (fun i => ofFn (coll:=Float^[n]) (g i))
```
the operation {lit}`↿` uncurries a function i.e. {lit}`↿g = fun (i,j) => g i j`
  -/
syntax (name:=ofFnNotation) "⊞ " funBinder* " => " term:51 : term

open Term Function Elab Term Meta in
@[term_elab ofFnNotation]
def ofFnElab : TermElab := fun stx expectedType? =>
  match stx with
  | `(⊞ $xs* => $b) => do
    let fn ← elabTermAndSynthesize (← `(fun $xs* => $b)) none
    let arity := (← inferType fn).getNumHeadForalls
    -- uncurry if necessary
    let fn ←
      if arity = 1 then
        pure fn
      else
        -- mkUncurryFun arity fn
        mkAppM ``HasUncurry.uncurry #[fn]

    let .some (Idx, Elem) := (← inferType fn).arrow?
      | throwError "⊞ _ => _: expectes function type {← ppExpr (← inferType fn)}"

    let Coll := expectedType?.getD (←mkFreshTypeMVar)
    try do
      let cls := mkAppN (← mkConstWithFreshMVarLevels ``DefaultCollection) #[Coll, Idx, Elem]
      let _ ← synthInstance cls
    catch _ =>
      -- todo: do some tracing
      pure ()

    mkAppOptM ``ofFn #[Coll, Idx, Elem, none, fn]
  | _ => throwUnsupportedSyntax



-- Notation: ⊞[1,2,3] --
------------------------

/-- A notation for creating a {lit}`DataArray` from a list of elements. -/
syntax (name := ofFnExplicitNotation) (priority:=high)
  "⊞[" ppRealGroup(sepBy1(ppGroup(term,+,?), ";", "; ", allowTrailingSep)) "]" : term


open Term Function Elab Term Meta in
@[term_elab ofFnExplicitNotation]
def ofFnExplicitElab : TermElab := fun stx expectedType? =>
  match stx with
  | `(⊞[$[$[$rows],*];*]) => do
    let m := rows.size
    let n := if h : 0 < m then rows[0].size else 0
    let elems := rows.flatten

    -- check dimenstions
    for row in rows do
      if row.size ≠ n then
        throwErrorAt (mkNullNode row) s!"\
          Rows must be of equal length; this row has {row.size} items, \
          the previous rows have {n}"

    let n := Syntax.mkNumLit (toString n)
    let m := Syntax.mkNumLit (toString rows.size)

    if rows.size = 1 then
      let fn ←
        elabTerm (←`(⊞ (i : Idx $n) => ![$elems,*] i.toFin)) expectedType?

      return fn
    else
      let fn ←
        elabTerm (←`(⊞ (i : Idx $m) (j : Idx $n) => !![$[$[$rows],*];*] i.toFin j.toFin))
          expectedType?

      return fn
  | _ => throwUnsupportedSyntax


@[app_unexpander ofFn]
def unexpandArrayTypeOfFnNotation : Lean.PrettyPrinter.Unexpander
  | `($(_) $f) =>
    match f with
    | `(fun $_ => ![$xs,*] $_) =>
      `(⊞[$xs,*])
    | `(fun $x => $b) =>
      `(⊞ $x:term => $b)
    | `(↿fun $_ $_ => !![$[$[$xs],*];*] $_ $_) =>
      `(⊞[$[$[$xs],*];*])
    | `(↿fun $xs* => $b) =>
      `(⊞ $xs* => $b)
    | _ => throw ()
  | _  => throw ()


-- Notation: Float ^ Idx n --
-----------------------------

namespace ArrayType.PowerNotation

-- Notation: Float^[10,20] --
-----------------------------

declare_syntax_cat dimSpec
syntax term : dimSpec
syntax (priority:=high) "[" dimSpec,+ "]" : dimSpec
syntax "[" term ":" term "]": dimSpec

/-- {lit}`x^[y]` is either array type or iterated function depending on the type of {lit}`x`.

**iterated function** {lit}`f^[n]` call {lit}`f` {lit}`n`-times e.g. {lit}`f^[3] = f∘f∘f`

**type product** {lit}`Float^[n]` array of {lit}`n` elemts with values in {lit}`Float`

The array notation is quite flexible and allows you to create arrays arrayType with various types.
Examples where {lit}`n` {lit}`m` {lit}`k` {lit}`l` are {lit}`USize`, {lit}`a` {lit}`b` are {lit}`Int64`
and {lit}`ι` {lit}`κ` are {lit}`Type` with {lit}`Index` instances:
- {lit}`Float^[n]` index type: {lit}`Idx n` i.e. numbers {lit}`0,...,n-1`
- {lit}`Float^[n,m]` index type: {lit}`Idx n × Idx m` i.e. paris {lit}`(0,0),(0,1),...,(1,0),(1,1),...,(n-1,m-1)`
- {lit}`Float^[[-a:b]]` index type: {lit}`Idx' (-a) b` i.e. closed interval from {lit}`-a` to {lit}`b` ({lit}`={-a, -a+1, ..., b-1, b}`)
- {lit}`Float^[k,l,m]` index type: {lit}`Idx k × Idx l × Idx m` - type product is right associated by default
- {lit}`Float^[[k,l],m]` index type: {lit}`(Idx k × Idx l) × Idx m` - left associated product requires explicit brackets
- {lit}`Float^[ι]` index type {lit}`ι` - generic type with {lit}`Index` {lit}`ι` instances
- {lit}`Float^[ι,[10,[-a:b]],κ]` index type: {lit}`ι × (Idx 10 × Idx' (-a) b) × κ` - mix of all above
-/
syntax (name:=typeIntPower) (priority:=high) term "^[" dimSpec,* "]" : term

open Lean Meta Elab Term Qq
partial def expand' (l : List (TSyntax `dimSpec)) : TermElabM Expr :=
  match l with
  | []  => default
  | [t] =>
    match t with
    | `(dimSpec| $n:term) => do
      try
        let n ← elabTerm n q(Nat)
        return ← mkAppM ``Idx #[n]
      catch _ =>
        return ← elabTerm n none
    | `(dimSpec| [$n:term : $m:term]) => do elabTerm (← `(Idx2 ($n : Int) ($m : Int))) q(Type)
    | `(dimSpec| [$ds:dimSpec,*]) => expand' ds.getElems.toList
    | _ => throwError "unexpected type power syntax"
  | t :: l' =>  do
    let a ← expand' [t]
    let b ← expand' l'
    mkAppM ``Prod #[a,b]


open Lean Meta Elab Term
elab_rules (kind:=typeIntPower) : term
| `($x:term ^[$ns,*]) => do
  let X ← elabTerm x none

  if ¬(← isType (← inferType X)) then
    if ns.getElems.size != 1 then
      throwUnsupportedSyntax
    else
      match ns.getElems[0]! with
      | `(dimSpec| $n:term) =>
        let f ← elabTerm x none
        let n ← elabTerm n q(Nat)
        return ← mkAppM ``Nat.iterate #[f,n]
      | _ => throwUnsupportedSyntax

  let I ← expand' ns.getElems.toList
  -- let C ← mkFreshTypeMVar
  -- let inst ← synthInstance <| mkAppN (← mkConstWithFreshMVarLevels ``ArrayTypeNotation) #[C,Y,X]
  -- let C ← whnfR (← instantiateMVars C)
  -- let result ← instantiateMVars <| ← mkAppOptM ``arrayTypeCont #[Y,X,C,inst]
  return ← mkAppOptM `SciLean.DataArrayN #[X,none,I,none,none] --result.getRevArg! 1

end ArrayType.PowerNotation
