/-
Copyright (c) 2024 SciLean contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: SciLean contributors
-/
import SciLean.Data.IndexType.Basic
import SciLean.Util.SorryProof

namespace SciLean

/-- Shape metadata for index types, used to construct tensor layouts. -/
class IndexTypeShape (ι : Type*) (n : outParam Nat) [IndexType ι n] where
  /-- Shape as an {name}`Array` of dimension sizes. -/
  shape : Array Nat
  /-- Total number of elements implied by shape equals {lean}`n`. -/
  h_numel : shape.foldl (· * ·) 1 = n

namespace IndexTypeShape

/-- Total number of elements implied by shape. -/
@[inline]
def numel {ι : Type*} {n : Nat} [IndexType ι n] [IndexTypeShape ι n] : Nat :=
  (IndexTypeShape.shape (ι:=ι)).foldl (· * ·) 1

end IndexTypeShape

/-! ## Instances -/

instance {n : Nat} : IndexTypeShape (Idx n) n where
  shape := #[n]
  h_numel := by sorry_proof

instance {n : Nat} : IndexTypeShape (Fin n) n where
  shape := #[n]
  h_numel := by sorry_proof

instance {a b : Int} : IndexTypeShape (Idx2 a b) (b - a + 1).toNat where
  shape := #[(b - a + 1).toNat]
  h_numel := by sorry_proof

instance {I J : Type*} {m n : Nat}
    [IndexType I m] [IndexType J n]
    [IndexTypeShape I m] [IndexTypeShape J n] :
    IndexTypeShape (I × J) (m * n) where
  shape :=
    Array.mk ((IndexTypeShape.shape (ι:=I)).toList ++ (IndexTypeShape.shape (ι:=J)).toList)
  h_numel := by sorry_proof

instance : IndexTypeShape Unit 1 where
  shape := #[1]
  h_numel := by sorry_proof

end SciLean
