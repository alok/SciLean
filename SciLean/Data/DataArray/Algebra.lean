import SciLean.Data.DataArray.RnEquiv
import SciLean.Data.ArrayOperations.Algebra
import SciLean.Algebra.VectorOptimize.Basic
import SciLean.VersoPrelude

/-! Algebraic structure in {lit}`X^[I]`

This file automatically pulls algebraic structure of {lit}`R^[n]` onto {lit}`X^[I]` anytime {lit}`X` has
an instance of {lit}`HasRnEquiv X m R`.

TODO: There should be a class that the structure of.
-/

namespace SciLean

open scoped ArrayType


namespace DataArrayN

variable
  (X I Y : Type*)
  {nI} [IndexType I nI]
  [PlainDataType Y]
  [PlainDataType X]
  [DataArrayEquiv X I Y]
  {J nJ} [IndexType J nJ] -- uncurry index
  {K nK} [IndexType K nK] -- this will be the canonical index to get to the data
  {R} [RealScalar R] [PlainDataType R] [BLAS (DataArray R) R R]


-- Derive operations and algebraic structures in `X^[I]`
instance instNormedAddCommGroup [HasRnEquiv X K R] :
    NormedAddCommGroup (X^[I]) := NormedAddCommGroup.ofRnEquiv (X^[I])

instance instAdjointSpace [HasRnEquiv X K R] :
    AdjointSpace R (X^[I]) := AdjointSpace.ofRnEquiv (X^[I])

instance instCompleteSpace [HasRnEquiv X K R] :
    CompleteSpace (X^[I]) := sorry_proof


-- short circuit instances
instance [HasRnEquiv X K R] : Add (X^[I]) := by infer_instance
instance [HasRnEquiv X K R] : Sub (X^[I]) := by infer_instance
instance [HasRnEquiv X K R] : Neg (X^[I]) := by infer_instance
instance [HasRnEquiv X K R] : SMul ℕ (X^[I]) := by infer_instance
instance [HasRnEquiv X K R] : SMul ℤ (X^[I]) := by infer_instance
instance [HasRnEquiv X K R] : SMul R (X^[I]) := by infer_instance
instance [HasRnEquiv X K R] : MulAction ℕ (X^[I]) := by infer_instance
instance [HasRnEquiv X K R] : MulAction ℤ (X^[I]) := by infer_instance
instance [HasRnEquiv X K R] : MulAction R (X^[I]) := by infer_instance
instance [HasRnEquiv X K R] : Inner R (X^[I]) := by infer_instance
instance [HasRnEquiv X K R] : AddCommGroup (X^[I]) := by infer_instance
instance [HasRnEquiv X K R] : Module R (X^[I]) := by infer_instance
instance [HasRnEquiv X K R] : TopologicalSpace (X^[I]) := by infer_instance


-- TODO: change definitino of `AdjointSpace` to require `SMul (Rᵐᵒᵖ) X` and
--       complex conjugation fow which I use `Star X` right now but it is not a good choice
--       as it would clash with conjugate-transpose for matrices
instance [HasRnEquiv X K R] : SMul (Rᵐᵒᵖ) (X^[I]) := ⟨fun r x => r.1•x⟩
instance [HasRnEquiv X K R] : Star (X^[I]) := ⟨fun x => x⟩


instance [HasRnEquiv X K R] : Axpby R (X^[I]) where
  axpby a x b y :=
    let data := BLAS.LevelOneDataExt.axpby nI a (toRn x).1 0 1 b (toRn y).1 0 1
    fromRn ⟨data, sorry_proof⟩
  axpby_spec := sorry_proof

----------------------------------------------------------------------------------------------------

-- example : Add (R^[I]) := by sorry_proof
-- example : Add (R^[I,J]) := by sorry_proof
-- example : Add (R^[J]^[I]) := by sorry_proof
-- example : Add (R^[I]^[J]) := by sorry_proof
-- example : Add (R^[I]^[J]^[I]) := by sorry_proof
-- example : Add (R^[I]^[J]^[I]) := by sorry_proof

-- example : SMul R (R^[I]) := by sorry_proof
-- example : SMul R (R^[I,J]) := by sorry_proof
-- example : SMul R (R^[J]^[I]) := by sorry_proof
-- example : SMul R (R^[I]^[J]) := by sorry_proof
-- example : SMul R (R^[I]^[J]^[I]) := by sorry_proof
-- example : SMul R (R^[I]^[J]^[I]) := by sorry_proof

----------------------------------------------------------------------------------------------------


-- IsZeroGetElem instances
instance instIsZeroGetElemInductive [HasRnEquiv X K R] :
    IsZeroGetElem (X^[J]^[I]) I where
  getElem_zero := by sorry_proof

instance  instIsZeroGetElemBase  : IsZeroGetElem (R^[I]) I := by sorry_proof

instance instIsZeroGetElemUncurry {L nL} [IndexType L nL]
    [HasRnEquiv X K R]
    {Y} [PlainDataType Y] [Zero Y]
    [DataArrayEquiv (X^[L]) J Y] [GetElem' (X^[L]) J Y]  [IsZeroGetElem (X^[L]) J] :
    IsZeroGetElem (X^[L]^[I]) (I × J) where
  getElem_zero := by intro ⟨i,j⟩; simp[getElem_curry]

instance instIsZeroGetElemUncurryBase : IsZeroGetElem (R^[I]) (I × Unit) where
  getElem_zero := by sorry_proof

instance (priority:=1000) instIsZeroGetElemCurryBase
    {X : Type*} [PlainDataType X] [Zero X]
    {I J : Type*} {nI} [IndexType I nI] {nJ} [IndexType J nJ]
    [Fold.{_,0} I] [DecidableEq I] [Fold.{_,0} J] [DecidableEq J] :
    IsZeroGetElem (X^[J]^[I]) (I × J) where
  getElem_zero := by intro _; sorry_proof

instance instIsZeroGetElemRn [HasRnEquiv X K R] : IsZeroGetElem (X^[I]) (Idx (nI*nK)) where
  getElem_zero := sorry_proof

-- example : IsZeroGetElem (R^[I]) I := by sorry_proof
-- example : IsZeroGetElem (R^[I,J]) (I×J) := by sorry_proof

-- set_option trace.Meta.synthInstance true in
-- example : IsZeroGetElem (R^[J]^[I]) (I×J) := by sorry_proof
-- example : IsZeroGetElem (R^[I]^[J]) J := by sorry_proof
-- example : IsZeroGetElem (R^[I]^[J]^[I]) (I) := by sorry_proof
-- example : IsZeroGetElem (R^[I]^[J]^[I]) (I×J) := by sorry_proof
-- example : IsZeroGetElem (R^[I]^[J]^[I]) (Idx (nI*(nJ*(nI*1)))) := by sorry_proof

----------------------------------------------------------------------------------------------------

-- IsAddGetElem instances
instance instIsAddGetElemInductive [HasRnEquiv X K R] :
    IsAddGetElem (X^[J]^[I]) I where
  getElem_add := by intro x x'; sorry_proof

instance instIsAddGetElemBase : IsAddGetElem (R^[I]) I := by sorry_proof

-- this has incorrect assumptions
instance instIsAddGetElemUncurry {L nL} [IndexType L nL]
    [HasRnEquiv X K R]
    {Y} [PlainDataType Y] [Add Y]
    [DataArrayEquiv (X^[L]) J Y] [GetElem' (X^[L]) J Y]  [IsAddGetElem (X^[L]) J] :
    IsAddGetElem (X^[L]^[I]) (I × J) where
  getElem_add := sorry_proof

instance instIsAddGetElemUncurryBase : IsAddGetElem (R^[I]) (I × Unit) where
  getElem_add := by sorry_proof

instance (priority:=1000) instIsAddGetElemCurryBase
    {X : Type*} [PlainDataType X] [Add X]
    {I J : Type*} {nI} [IndexType I nI] {nJ} [IndexType J nJ]
    [Fold.{_,0} I] [DecidableEq I] [Fold.{_,0} J] [DecidableEq J] :
    IsAddGetElem (X^[J]^[I]) (I × J) where
  getElem_add := by intro _ _ _; sorry_proof

instance instIsAddGetElemRn [HasRnEquiv X K R] : IsAddGetElem (X^[I]) (Idx (nI*nK)) where
  getElem_add := sorry_proof

-- example : IsAddGetElem (R^[I]) I := by sorry_proof
-- example : IsAddGetElem (R^[I,J]) (I×J) := by sorry_proof
-- example : IsAddGetElem (R^[J]^[I]) (I×J) := by sorry_proof
-- example : IsAddGetElem (R^[I]^[J]) J := by sorry_proof
-- example : IsAddGetElem (R^[I]^[J]^[I]) (I) := by sorry_proof
-- example : IsAddGetElem (R^[I]^[J]^[I]) (I×J) := by sorry_proof
-- example : IsAddGetElem (R^[I]^[J]^[I]) (I×J×I) := by sorry_proof
-- example : IsAddGetElem (R^[I]^[J]^[I]) (Idx (nI*(nJ*(nI*1)))) := by sorry_proof

----------------------------------------------------------------------------------------------------


-- IsNegGetElem instances
instance instIsNegGetElemInductive [HasRnEquiv X K R] :
    IsNegGetElem (X^[J]^[I]) I where
  getElem_neg := by sorry_proof

instance instIsNegGetElemBase : IsNegGetElem (R^[I]) I := by sorry_proof

instance instIsNegGetElemUncurry {L nL} [IndexType L nL]
    [HasRnEquiv X K R]
    {Y} [PlainDataType Y] [Neg Y]
    [DataArrayEquiv (X^[L]) J Y] [GetElem' (X^[L]) J Y]  [IsNegGetElem (X^[L]) J] :
    IsNegGetElem (X^[L]^[I]) (I × J) where
  getElem_neg := by intro ⟨i,j⟩; simp[getElem_curry]

instance instIsNegGetElemUncurryBase : IsNegGetElem (R^[I]) (I × Unit) where
  getElem_neg := by sorry_proof

instance (priority:=1000) instIsNegGetElemCurryBase
    {X : Type*} [PlainDataType X] [Neg X]
    {I J : Type*} {nI} [IndexType I nI] {nJ} [IndexType J nJ]
    [Fold.{_,0} I] [DecidableEq I] [Fold.{_,0} J] [DecidableEq J] :
    IsNegGetElem (X^[J]^[I]) (I × J) where
  getElem_neg := by intro _ _; sorry_proof

instance instIsNegGetElemRn [HasRnEquiv X K R] : IsNegGetElem (X^[I]) (Idx (nI*nK)) where
  getElem_neg := sorry_proof


-- example : IsNegGetElem (R^[I]) I := by sorry_proof
-- example : IsNegGetElem (R^[I,J]) (I×J) := by sorry_proof
-- example : IsNegGetElem (R^[J]^[I]) (I×J) := by sorry_proof
-- example : IsNegGetElem (R^[I]^[J]) J := by sorry_proof
-- example : IsNegGetElem (R^[I]^[J]^[I]) (I) := by sorry_proof
-- example : IsNegGetElem (R^[I]^[J]^[I]) (I×J) := by sorry_proof
-- example : IsNegGetElem (R^[I]^[J]^[I]) (Idx (nI*(nJ*(nI*1)))) := by sorry_proof
----------------------------------------------------------------------------------------------------


-- IsSMulGetElem instances
instance instIsSMulGetElemInductive [HasRnEquiv X K R] :
    IsSMulGetElem R (X^[J]^[I]) I where
  getElem_smul := by sorry_proof

instance instIsSMulGetElemBase : IsSMulGetElem R (R^[I]) I := by sorry_proof

-- this has incorrect assumptions
instance instIsSMulGetElemUncurry {L nL} [IndexType L nL]
    [HasRnEquiv X K R]
    {Y} [PlainDataType Y] [SMul R Y]
    [DataArrayEquiv (X^[L]) J Y] [GetElem' (X^[L]) J Y]  [IsSMulGetElem R (X^[L]) J] :
    IsSMulGetElem R (X^[L]^[I]) (I × J) where
  getElem_smul := sorry_proof

instance instIsSMulGetElemUncurryBase : IsSMulGetElem R (R^[I]) (I × Unit) where
  getElem_smul := by sorry_proof

instance (priority:=1000) instIsSMulGetElemCurryBase
    {𝕜 X : Type*} [PlainDataType X] [SMul 𝕜 X]
    {I J : Type*} {nI} [IndexType I nI] {nJ} [IndexType J nJ]
    [Fold.{_,0} I] [DecidableEq I] [Fold.{_,0} J] [DecidableEq J]
    [SMul 𝕜 (X^[J]^[I])] :
    IsSMulGetElem 𝕜 (X^[J]^[I]) (I × J) where
  getElem_smul := by intro _ _ _; sorry_proof

instance instIsSMulGetElemRn [HasRnEquiv X K R] : IsSMulGetElem R (X^[I]) (Idx (nI*nK)) where
  getElem_smul := sorry_proof


-- example : IsSMulGetElem R (R^[I]) I := by sorry_proof
-- example : IsSMulGetElem R (R^[I,J]) (I×J) := by sorry_proof
-- example : IsSMulGetElem R (R^[J]^[I]) (I×J) := by sorry_proof
-- example : IsSMulGetElem R (R^[I]^[J]) J := by sorry_proof
-- example : IsSMulGetElem R (R^[I]^[J]^[I]) (I) := by sorry_proof
-- example : IsSMulGetElem R (R^[I]^[J]^[I]) (I×J) := by sorry_proof
-- example : IsSMulGetElem R (R^[I]^[J]^[I]) (Idx (nI*(nJ*(nI*1)))) := by sorry_proof
-- example : IsSMulGetElem R (R^[I]^[J]^[I]) (Idx (nI*(nJ*(nI*1)))) := by sorry_proof

----------------------------------------------------------------------------------------------------



-- IsInnerGetElem instances
instance instIsInnerGetElemInductive [HasRnEquiv X K R] :
    IsInnerGetElem R (X^[J]^[I]) I where
  inner_eq_sum_getElem := by sorry_proof

instance instIsInnerGetElemBase : IsInnerGetElem R (R^[I]) I := by sorry_proof

-- this has incorrect assumptions
instance instIsInnerGetElemUncurry {L nL} [IndexType L nL]
    [HasRnEquiv X K R]
    {Y} [PlainDataType Y] [Inner R Y]
    [DataArrayEquiv (X^[L]) J Y] [GetElem' (X^[L]) J Y]  [IsInnerGetElem R (X^[L]) J] :
    IsInnerGetElem R (X^[L]^[I]) (I × J) where
  inner_eq_sum_getElem := sorry_proof

instance instIsInnerGetElemUncurryBase : IsInnerGetElem R (R^[I]) (I × Unit) where
  inner_eq_sum_getElem := by sorry_proof

instance (priority:=1000) instIsInnerGetElemCurryBase
    {𝕜 X : Type*} [AddCommMonoid 𝕜] [PlainDataType X] [Inner 𝕜 X]
    {I J : Type*} {nI} [IndexType I nI] {nJ} [IndexType J nJ]
    [Fold.{_,0} I] [DecidableEq I] [Fold.{_,0} J] [DecidableEq J]
    [Inner 𝕜 (X^[J]^[I])] :
    IsInnerGetElem 𝕜 (X^[J]^[I]) (I × J) where
  inner_eq_sum_getElem := by intro _ _; sorry_proof

instance instIsInnerGetElemRn [HasRnEquiv X K R] : IsInnerGetElem R (X^[I]) (Idx (nI*nK)) where
  inner_eq_sum_getElem := sorry_proof


-- example : IsInnerGetElem R (R^[I]) I := by sorry_proof
-- example : IsInnerGetElem R (R^[I,J]) (I×J) := by sorry_proof
-- example : IsInnerGetElem R (R^[J]^[I]) (I×J) := by sorry_proof
-- example : IsInnerGetElem R (R^[I]^[J]) J := by sorry_proof
-- example : IsInnerGetElem R (R^[I]^[J]^[I]) (I) := by sorry_proof
-- example : IsInnerGetElem R (R^[I]^[J]^[I]) (I×J) := by sorry_proof
-- example : IsInnerGetElem R (R^[I]^[J]^[I]) (Idx (nI*(nJ*(nI*1)))) := by sorry_proof

----------------------------------------------------------------------------------------------------


-- IsModuleGetElem instances
instance instIsModuleGetElemInductive [HasRnEquiv X K R] :
    IsModuleGetElem R (X^[J]^[I]) I where

instance instIsModuleGetElemBase : IsModuleGetElem R (R^[I]) I where

-- this has incorrect assumptions
instance instIsModuleGetElemUncurry {L nL} [IndexType L nL]
    [HasRnEquiv X K R]
    {Y} [PlainDataType Y] [AddCommGroup Y] [Module R Y]
    [DataArrayEquiv (X^[L]) J Y] [GetElem' (X^[L]) J Y]  [IsModuleGetElem R (X^[L]) J] :
    IsModuleGetElem R (X^[L]^[I]) (I × J) where

instance instIsModuleGetElemUncurryBase : IsModuleGetElem R (R^[I]) (I × Unit) where

set_option synthInstance.checkSynthOrder false in
instance (priority:=1000) instIsModuleGetElemCurryBase
    {𝕜 : outParam Type*} {X : Type*} [Ring 𝕜] [PlainDataType X] [AddCommGroup X] [Module 𝕜 X]
    {I J : Type*} {nI} [IndexType I nI] {nJ} [IndexType J nJ]
    [Fold.{_,0} I] [DecidableEq I] [Fold.{_,0} J] [DecidableEq J]
    [AddCommGroup (X^[J]^[I])] [Module 𝕜 (X^[J]^[I])] :
    IsModuleGetElem 𝕜 (X^[J]^[I]) (I × J) where
  getElem_zero := by intro _; sorry_proof
  getElem_add := by intro _ _ _; sorry_proof
  getElem_neg := by intro _ _; sorry_proof
  getElem_smul := by intro _ _ _; sorry_proof

instance instIsModuleGetElemRn [HasRnEquiv X K R] : IsModuleGetElem R (X^[I]) (Idx (nI*nK)) where


-- example : IsModuleGetElem R (R^[I]) I := by sorry_proof
-- example : IsModuleGetElem R (R^[I,J]) (I×J) := by sorry_proof
-- example : IsModuleGetElem R (R^[J]^[I]) (I×J) := by sorry_proof
-- example : IsModuleGetElem R (R^[I]^[J]) J := by sorry_proof
-- example : IsModuleGetElem R (R^[I]^[J]^[I]) (I) := by sorry_proof
-- example : IsModuleGetElem R (R^[I]^[J]^[I]) (I×J) := by sorry_proof
-- example : IsModuleGetElem R (R^[I]^[J]^[I]) (Idx (nI*(nJ*(nI*1)))) := by sorry_proof

----------------------------------------------------------------------------------------------------



-- IsContinuousGetElem instances
instance instIsContinuousGetElemInductive [HasRnEquiv X K R] :
    IsContinuousGetElem (X^[J]^[I]) I where
  continuous_getElem := sorry_proof

instance instIsContinuousGetElemBase : IsContinuousGetElem (R^[I]) I := by sorry_proof

-- this has incorrect assumptions
instance instIsContinuousGetElemUncurry {L nL} [IndexType L nL]
    [HasRnEquiv X K R]
    {Y} [PlainDataType Y] [TopologicalSpace Y]
    [DataArrayEquiv (X^[L]) J Y] [GetElem' (X^[L]) J Y]  [IsContinuousGetElem (X^[L]) J] :
    IsContinuousGetElem (X^[L]^[I]) (I × J) where
  continuous_getElem := by intro ⟨i,j⟩; simp[getElem_curry]; fun_prop

instance instIsContinuousGetElemUncurryBase : IsContinuousGetElem (R^[I]) (I × Unit) where
  continuous_getElem := sorry_proof

instance (priority:=1000) instIsContinuousGetElemCurryBase
    {X : Type*} [TopologicalSpace X] [PlainDataType X]
    {I J : Type*} {nI} [IndexType I nI] {nJ} [IndexType J nJ]
    [Fold.{_,0} I] [DecidableEq I] [Fold.{_,0} J] [DecidableEq J]
    [TopologicalSpace (X^[J]^[I])] :
    IsContinuousGetElem (X^[J]^[I]) (I × J) where
  continuous_getElem := by intro _; sorry_proof

instance instContinuousGetElemRn [HasRnEquiv X K R] : IsContinuousGetElem (X^[I]) (Idx (nI*nK)) where
  continuous_getElem := sorry_proof


-- example : IsContinuousGetElem (R^[I]) I := by sorry_proof
-- example : IsContinuousGetElem (R^[I,J]) (I×J) := by sorry_proof
-- example : IsContinuousGetElem (R^[J]^[I]) (I×J) := by sorry_proof
-- example : IsContinuousGetElem (R^[I]^[J]) J := by sorry_proof
-- example : IsContinuousGetElem (R^[I]^[J]^[I]) (I) := by sorry_proof
-- example : IsContinuousGetElem (R^[I]^[J]^[I]) (I×J) := by sorry_proof
-- example : IsContinuousGetElem (R^[I]^[J]^[I]) (Idx (nI*(nJ*(nI*1)))) := by sorry_proof

----------------------------------------------------------------------------------------------------
