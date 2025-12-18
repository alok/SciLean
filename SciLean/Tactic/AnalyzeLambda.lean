import Lean
import Qq

import SciLean.Lean.Meta.Basic
import SciLean.Data.ArraySet

set_option linter.unusedVariables false

open Lean Meta

namespace SciLean


inductive HeadFunInfo
  | const (constName : Name) (arity : Nat)
  | fvar (id : FVarId) (arity : Nat)
  | bvar (i : Nat) (arity : Nat)


def HeadFunInfo.arity (info : HeadFunInfo) : Nat :=
  match info with
  | .const _ n => n
  | .fvar _ n => n
  | .bvar _ n => n

def HeadFunInfo.ctorName (info : HeadFunInfo) : Name :=
  match info with
  | .const _ _ => ``const
  | .fvar _ _ => ``fvar
  | .bvar _ _ => ``bvar

def HeadFunInfo.isFVar (info : HeadFunInfo) (id : FVarId) : Bool :=
  match info with
  | .const _ _ => false
  | .fvar id' _ => id == id'
  | .bvar _ _ => false


inductive MainArgCase where
  /-- there are no main arguments -/
  | noMainArg
  /-- Main arguments are just {given}`x` i.e. {lit}`x = (a'₁, ..., a'ₖ)` where {lit}`a' = mainIds.map (fun i => aᵢ)` are main arguments -/
  | trivialUncurried
  /-- Main arguments are just functions of {given}`x` and do not depend on {given}`yⱼ`

  This allows to write the lambda function as composition
  ```
  fun x y₀ ... yₙ₋₁ => f a₀ ... aₘ₋₁
  =
  f' ∘ g'
  =
  (fun (a'₁, ..., a'ₖ) y₀ ... yₙ₋₁ => f a₀ ... aₘ₋₁) ∘ (fun x => (a'₁, ..., a'ₖ))
  ```
  where the function {given}`f'` is in {name}`MainArgCase.trivialUncurried` case -/
  | nonTrivialNoTrailing
  /-- Main arguments depend on {given}`x` and {given}`yⱼ` -/
  | nonTrivialWithTrailing
deriving DecidableEq, Repr

inductive TrailingArgCase where
  /-- there are no trailing arguments -/
  | noTrailingArg

  /-- Trailing arguments are exactly equal to {given}`yⱼ`
  i.e. {lit}`yⱼ = a''ⱼ` where {lit}`a'' := trailingArgs.map (fun i => aᵢ)` -/
  | trivial

  /-- Trailing arguments are just {given}`y₀` i.e. {lit}`n=1` and {lit}`y₀ = (a''₁, ..., a''ₖ)`
  where {lit}`a'' := trailingIds.map (fun i => aᵢ)`

  It is guaranteed that {lit}`k>1`, when {lit}`k=1` then we are in {name}`TrailingArgCase.trivial` case -/
  | trivialUncurried

  /-- Trailing arguments are non trivial function of {given}`yⱼ`

  In this case we usually want to find inverse map {given}`h` mapping {given}`a''` to {given}`yⱼ`
  ```
  fun x y₀ ... yₙ₋₁ => f a₀ ... aₘ₋₁
  =
  (·∘h) ∘ f'
  =
  (·∘h) ∘ (fun x a''₁ ... a''ₖ => f a₀ ... aₘ₋₁
  ```
  where the function {given}`f'` is now in {name}`TrailingArgCase.trivial` case
  (constructing such {given}`f'` is a bit tricky as it potentially requires to also
  use {given}`h` to replace {given}`yⱼ` with {given}`a''` in main arguments)
   -/
  | nonTrivial
deriving DecidableEq, Repr


/-- Info about lambda function {lit}`fun x y₀ ... yₙ₋₁ => f a₀ ... aₘ₋₁`
-/
structure LambdaInfo where
  -- /-- the lambda function itself -/
  -- fn : Expr
  /-- number of lambda binders -/
  arity : Nat -- n+1
  /-- number of function arguments in the body -/
  argNum : Nat -- m
  /-- info on the head function {given}`f` -/
  headFunInfo : HeadFunInfo
  /-- Set of argument indices {given}`i` saying that {given}`aᵢ` depends on {given}`x`, they might depend {given}`yⱼ` too -/
  mainIds : ArraySet Nat
  /-- Set of argument indices {given}`i` saying that {given}`aᵢ` depends on at least one of {given}`yⱼ` but not on {given}`x` -/
  trailingIds : ArraySet Nat
  -- /-- Set of argument indices `i` saying that `aᵢ` does not depend of `x` or `yⱼ`
  -- This is a complement of `mainIds ∪ trailinIds` -/
  -- unusedIds : ArraySet Nat
  mainArgCase : MainArgCase
  trailingArgCase : TrailingArgCase


/-- Analyze head function {given}`f` of lambda {lit}`fun x₁ ... xₙ => f ...` where {lit}`xs = #[x₁, ..., xₙ]`

Returns {name}`HeadFunInfo.bvar` if the head function is fvar and one of {given}`xs`
-/
private def analyzeHeadFun (fn : Expr) (xs : Array Expr) : MetaM HeadFunInfo := do
  match fn with
  | .const name _ =>
    pure (.const name (← getConstArity name))
  | .fvar id =>
    let arity := (← inferType fn).getNumHeadForalls
    if let .some i := xs.findIdx? (fun x => x.fvarId! == id) then
      pure (.bvar i arity)
    else
      pure (.fvar id arity)
  | _ => throwError s!"invalid head function {← ppExpr fn}"

/--
Decompose function as {lit}`fun x i₁ ... iₙ => f (g x) (h i₁ ... iₙ)`

{lit}`f = fun y₁ ... yₘ i₁ ... iₙ => f' y₁ ... yₘ`
-/
partial def analyzeLambda (e : Expr) : MetaM LambdaInfo := do
  lambdaTelescope e fun xs body => do

    if xs.size = 0 then
      throwError "lambda expected in analyzeLambda"

    -- if `body` is a projection turn it into application of projection function
    let body := (← revertStructureProj body).headBeta


    let fn := body.getAppFn'
    let args := body.getAppArgs

    let fnInfo ← analyzeHeadFun fn xs

    let x := xs[0]!
    let xId := x.fvarId!
    -- let xName ← xId.getUserName
    let ys := xs[1:].toArray

    let mut as'  : Array Expr := #[]
    let mut as'' : Array Expr := #[]

    let mut mainIds : Array Nat := #[]
    let mut trailingIds : Array Nat := #[]

    let mut mainCase : MainArgCase := .noMainArg
    let mut trailingCase : TrailingArgCase := .noTrailingArg

    for arg in args, i in [0:args.size] do
      let ys' := ys.filter (fun y => arg.containsFVar y.fvarId!)

      if arg.containsFVar xId then
        mainIds := mainIds.push i
        as' := as'.push arg
        if ys'.size ≠ 0 then
          mainCase := .nonTrivialWithTrailing
      else if ys'.size ≠ 0 then
        trailingIds := trailingIds.push i
        as'' := as''.push arg


    -- determine main arg case
    let a' ← mkProdElem as'
    if as'.size ≠ 0 && mainCase ≠ .nonTrivialWithTrailing then
      if (← isDefEq x a') then
        mainCase := .trivialUncurried
      else
        mainCase := .nonTrivialNoTrailing

    -- determine trailing arg case
    if as''.size ≠ 0 then
      trailingCase := .nonTrivial

      if ys.size = as''.size then
        if ← (Array.range ys.size).allM (fun i => isDefEq ys[i]! as''[i]!) then
          trailingCase := .trivial

      if ys.size = 1 && as''.size > 1 then
        let a'' ← mkProdElem as''
        if ← isDefEq ys[0]! a'' then
          trailingCase := .trivialUncurried


    return {
      arity := xs.size
      argNum := args.size
      headFunInfo := fnInfo
      mainIds := mainIds.toArraySet
      trailingIds := trailingIds.toArraySet
      mainArgCase := mainCase
      trailingArgCase := trailingCase
    }

open Qq

-- #exit
-- def LambdaInfo.print (info : LambdaInfo) : IO Unit := do
--   IO.println s!"arity: {info.arity}"
--   IO.println s!"argNum: {info.argNum}"
--   IO.println s!"headFunction ctor: {info.headFunInfo.ctorName}"
--   IO.println s!"headFunction arity: {info.headFunInfo.arity}"
--   IO.println s!"main ids: {info.mainIds}"
--   IO.println s!"trailing ids: {info.trailingIds}"
--   IO.println s!"main arg case: {repr info.mainArgCase}"
--   IO.println s!"trailing arg case: {repr info.trailingArgCase}"


-- #eval show MetaM Unit from do
--   withLocalDeclQ `f .default q(Float → Float → Float) fun f => do
--     let e := q(fun (x,y) (z,h) => $f h ($f z ($f x y)))
--     IO.println (← ppExpr e)
--     let e ← lambdaTelescope e fun xs b => do mkLambdaFVars xs (← whnf b)
--     IO.println (← ppExpr e)


-- #eval show MetaM Unit from do
--   withLocalDeclQ `x .default q(Float×Float) fun x => do
--     let e ← mkLambdaFVars #[x] (Expr.proj ``Prod 0 x)
--     let info ← analyzeLambda e
--     info.print

-- #eval show MetaM Unit from do
--   let e := q(fun (x : Float × Float) => x.1)
--   let info ← analyzeLambda e
--   info.print

-- #eval show MetaM Unit from do
--   let e := q(fun (x : (Fin 10 → Float) × (Fin 5 → Float)) => x.1)
--   let info ← analyzeLambda (← etaExpand' e)
--   info.print

-- #eval show MetaM Unit from do
--   let e := q(fun (x : (Fin 10 → Float) × (Fin 5 → Float)) i => x.1 i)
--   let info ← analyzeLambda e
--   info.print


-- #eval show MetaM Unit from do
--   let e := q(fun (x : (Fin 10 × Fin 20 → Float) × (Fin 5 → Float)) i j => x.1 (i,j))
--   let info ← analyzeLambda e
--   info.print


-- #eval show MetaM Unit from do
--   let e := q(fun (A : (Fin 10 → Fin 5 → Float)) (ij : Fin 10 × Fin 5) => A ij.1 ij.2)
--   let info ← analyzeLambda e
--   info.print

-- #eval show MetaM Unit from do
--   let e := q(fun (A : (Fin 10 → Fin 5 → Float)) i j => A i j)
--   let info ← analyzeLambda e
--   info.print

--  #eval show MetaM Unit from do
--   let e := q(fun (A : (Fin 10 → Fin 5 → Float)) j i => A i j)
--   let info ← analyzeLambda e
--   info.print


-- #eval show MetaM Unit from do
--   let e := q(fun (x : (Fin 10 → Float) × (Fin 5 → Float)) (i : Fin 10 × Fin 5) => x.1 i.1)
--   let info ← analyzeLambda e
--   info.print

-- #eval show MetaM Unit from do

--   withLocalDeclQ `i₂ .default q(Fin 2) fun i₂ => do
--   withLocalDeclQ `i₃ .default q(Fin 3) fun i₃ => do
--   withLocalDeclQ `i₅ .default q(Fin 5) fun i₅ => do
--   let e := q(fun (A : (Fin 1 → Fin 2 → Fin 3 → Fin 4 → Fin 5 → Fin 6 → Float)) i₁ i₄ i₆ => A i₁ $i₂ $i₃ i₄ $i₅ i₆)
--   IO.println (← ppExpr e.eta)
--   let _ ← analyzeLambda e


-- #eval show MetaM Unit from do

--   withLocalDeclQ `i₂ .default q(Fin 2) fun i₂ => do
--   withLocalDeclQ `i₃ .default q(Fin 3) fun i₃ => do
--   withLocalDeclQ `i₅ .default q(Fin 5) fun i₅ => do
--   let e := q(fun (A : (Fin 1 → Fin 2 → Fin 3 → Fin 4 → Fin 5 → Fin 6 → Float)) (i : Fin 1 × Fin 4 × Fin 6) => A i.1 $i₂ $i₃ i.2.1 $i₅ i.2.2)

--   let _ ← analyzeLambda e


-- #eval show MetaM Unit from do

--   withLocalDeclQ `i₂ .default q(Fin 2) fun i₂ => do
--   withLocalDeclQ `i₃ .default q(Fin 3) fun i₃ => do
--   withLocalDeclQ `i₅ .default q(Fin 5) fun i₅ => do
--   let e := q(fun (A : (Fin 1 → Fin 2 → Fin 3 → Fin 4 → Fin 5 → Fin 6 → Float)) (i : Fin 4 × Fin 1 × Fin 6) => A i.2.1 $i₂ $i₃ i.1 $i₅ i.2.2)

--   let _ ← analyzeLambda e
