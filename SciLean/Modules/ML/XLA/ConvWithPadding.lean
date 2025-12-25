import SciLean.Modules.ML.XLA.XlaTensorIndex
import SciLean.Data.DataArray
import SciLean.Data.DataArray.Algebra
import SciLean.Data.FinProd
import SciLean.Tactic.InferVar
import SciLean.Analysis.Normed.IsContinuousLinearMap
import SciLean.Analysis.Scalar.FloatAsReal
import SciLean.Data.ArrayType

namespace SciLean


variable {R} [RealScalar R] [PlainDataType R] [BLAS (DataArray R) R R]


theorem ArrayType.lt_elemwise {Cont Idx Elem} [ArrayType Cont Idx Elem] [LT Elem] {x y : Cont} :
   (∀ i, ArrayType.get x i < ArrayType.get y i) → x < y := id

theorem ArrayType.le_elemwise {Cont Idx Elem} [ArrayType Cont Idx Elem] [LE Elem] {x y : Cont} :
   (∀ i, ArrayType.get x i ≤ ArrayType.get y i) → x ≤ y := id


macro "tensor_index_bounds" i:term : tactic =>
  `(tactic|
     (constructor
      · apply ArrayType.le_elemwise; intro d; simp; have := ($i).2 d; omega;
      · apply ArrayType.lt_elemwise; intro d; simp; have := ($i).2 d; omega))

instance {r} {dim : Dims r} : GetElem (DataArrayN R (XlaTensorIndex dim)) (Vector ℤ r) R (fun _ i => 0 ≤ i ∧ i < dim) where
  getElem x i h := x[⟨i, by sorry_proof⟩]

@[fun_prop]
theorem getElem_clm {r} {dim : Dims r} (i : Vector ℤ r) (h : 0 ≤ i ∧ i < dim) :
    IsContinuousLinearMap R (fun x : DataArrayN R (XlaTensorIndex dim) => x[i]'h) := by
  sorry_proof

macro "reduce_dim" : tactic => `(tactic|
  first | simp_all | ((try simp); infer_var) | (ext i; simp; ring))


def convWithPadding {spDims kerDims : Dims r}
    (x : R^[XlaTensorIndex spDims]) (y : R^[XlaTensorIndex kerDims]) (low high : Vector ℤ r)
    {outDim : Dims r} (_houtDim : outDim = (spDims - kerDims + low + high + 1):= by reduce_dim) :
    R^[XlaTensorIndex outDim] :=
  ⊞ (i : XlaTensorIndex outDim) =>
    ∑ (j : XlaTensorIndex kerDims),
      let i' := i.1 + j.1 - low
      if h : 0 ≤ i' ∧ i' < spDims then
        x[i'] * y[j]
      else
        0



@[fun_trans]
theorem convWithPadding.arg_y.adjoint_rule {r} {spDims kerDims : Dims r}
    (x : R^[XlaTensorIndex spDims]) (l h : Vector ℤ r)
    {outDim : Dims r} (houtDim : outDim = (spDims - kerDims + l + h + 1)) :
    (adjoint R (fun (y : R^[XlaTensorIndex kerDims]) => convWithPadding x y l h houtDim))
    =
    fun z => convWithPadding x z l h (by sorry_proof) := by
  sorry_proof


def rev {r} {dim : Dims r} (_x : R^[XlaTensorIndex dim]) :
    R^[XlaTensorIndex dim] := by
  classical
  exact 0


@[fun_trans]
theorem convWithPadding.arg_x.adjoint_rule {r} {spDims kerDims : Dims r}
    (y : R^[XlaTensorIndex kerDims]) (l h : Vector ℤ r)
    {outDim : Dims r} (houtDim : outDim = (spDims - kerDims + l + h + 1)) :
    (adjoint R (fun (x : R^[XlaTensorIndex spDims]) => convWithPadding x y l h houtDim))
    =
    fun z : R^[XlaTensorIndex outDim] =>
      convWithPadding z (rev y) (kerDims-l-1) (kerDims-h-1) (by sorry_proof) := by
  sorry_proof
