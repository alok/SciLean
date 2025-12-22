import SciLean.Algebra.TensorProduct.Basic

namespace SciLean


/--
Class providing matrix transposition

This takes {given}`A : Y ⊗ X` and produces {lean}`Aᵀ : X ⊗ Y`
 -/
class TensorProductTranspose
  (R Y X YX XY : Type*) [RCLike R]
  [NormedAddCommGroup Z] [AdjointSpace R Z] [NormedAddCommGroup Y] [AdjointSpace R Y] [NormedAddCommGroup X] [AdjointSpace R X]
  [AddCommGroup YX] [Module R YX] [AddCommGroup XY] [Module R XY]
  [TensorProductType R Y X YX] [TensorProductType R X Y XY]
  where

    /-- Matrix transposition/conjugation

        conjTranspose A = Aᵀ  or  Aᴴ
    -/
    conjTranspose (A : YX) : XY
