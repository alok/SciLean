import SciLean.Data.FinProd
import SciLean.Data.ListN
import SciLean.Data.DataArray
import SciLean.Data.IndexType.Fold
import Mathlib.Tactic.Ring
import SciLean.Tactic.CompiledTactics
import SciLean.Util.SorryProof

namespace SciLean

/-- Dimensions of a rank {given}`rank` tensor.

A rank {lean}`rank` tensor has dimensions {lit}`#[d₁,...,dᵣ]`.
Each dimension has indices in the range {lit}`0,...,dᵢ-1`. -/
abbrev Dims (rank : ℕ) := Vector ℤ rank

instance {α n} : GetElem (Vector α n) (Fin n) α (fun _ _ => True) where
  getElem xs i _ := xs.get i

/-- Dimensions of a rank {given}`rank` tensor.

A rank {lean}`rank` tensor has dimensions {lit}`#[(a₁,b₁),...,(aᵣ,bᵣ)]`.
Each dimension has indices in the range {lit}`-aᵢ,...,bᵢ`. -/
abbrev DimsI (rank : ℕ) := Vector (ℤ×ℤ) rank

/-- Padding of a rank {given}`rank` tensor.

Tensor dimensions of rank {lean}`rank` are padded to yield
  - {lit}`-lᵢ,...,dᵢ+hᵢ-1`
  - {lit}`aᵢ-lᵢ,...,bᵢ+hᵢ` -/
abbrev Padding (rank : ℕ) := Vector (ℤ×ℤ) rank

/-- Alias for rank-indexed vectors. -/
abbrev ArrayN (α : Type*) (r : Nat) := Vector α r

instance {r} (n : Nat) : OfNat (Dims r) n :=
  ⟨.ofFn fun _ => (OfNat.ofNat n : ℤ)⟩


/-- Tensor index of a rank {given}`r` tensor with dimensions {given}`dims`.

Written {lit}`#[d₁,...,dᵣ]` for a tensor of rank {lean}`r` and dimensions {lean}`dims`. -/
@[ext]
structure XlaTensorIndex {r} (dims : Dims r) where
  val : Vector ℤ r
  bounds : ∀ (i : Fin r), 0 ≤ val[i] ∧ val[i] < dims[i]


/-- Tensor index of a rank {given}`r` tensor with dimensions {given}`dims`.

Written {lit}`#[(a₁,b₁),...,(aᵣ,bᵣ)]` for a tensor of rank {lean}`r` and dimensions {lean}`dims`. -/
@[ext]
structure XlaTensorIndexI {r} (dims : DimsI r) where
  val : Vector ℤ r
  bounds : ∀ (i : Fin r), dims[i].1 ≤ val[i] ∧ val[i] ≤ dims[i].2

instance {r} {dim : Dims r} : GetElem (XlaTensorIndex dim) (Fin r) ℤ (fun _ _ => True) where
  getElem x i _ := x.val.get i

instance {r} {dim : DimsI r} : GetElem (XlaTensorIndexI dim) (Fin r) ℤ (fun _ _ => True) where
  getElem x i _ := x.val.get i

instance {r} {dim : Dims r} : DecidableEq (XlaTensorIndex dim) :=
  fun x y =>
    if h : x.val = y.val then
      .isTrue (by cases x; simp_all)
    else
      .isFalse (by by_contra h; simp_all)

instance {r} {dim : DimsI r} : DecidableEq (XlaTensorIndexI dim) :=
  fun x y =>
    if h : x.val = y.val then
      .isTrue (by cases x; simp_all)
    else
      .isFalse (by by_contra h; simp_all)

@[simp]
theorem XlaTensorIndex.get_le {r} {dim : Dims r} (i : XlaTensorIndex dim) :
    ∀ (d : Fin r), 0 ≤ i[d] ∧ i[d] < dim[d] := i.2

def XlaTensorIndex.bounds' {r} {dims : Dims r} (i : XlaTensorIndex dims) :
    ∀ d : Fin r, (0:ℤ) ≤ i[d] ∧ i[d] < dims[d] := by
  intro d
  exact i.get_le d

@[simp]
theorem XlaTensorIndexI.get_le {r} {dim : DimsI r} (i : XlaTensorIndexI dim) :
    ∀ (d : Fin r), dim[d].1 ≤ i[d] ∧ i[d] ≤ dim[d].2 := i.2


instance {r} {dims : Dims r}  : IndexType (XlaTensorIndex dims) r := sorry
instance {r} {dims : DimsI r} : IndexType (XlaTensorIndexI dims) r := sorry

@[inline]
instance {m : Type _ → Type _} {r} {dims : Dims r} : FoldM (XlaTensorIndex dims) m := by
  classical
  refine { forIn := ?_ }
  intro β _ _r init _f
  exact pure init

@[inline]
instance {m : Type _ → Type _} {r} {dims : DimsI r} : FoldM (XlaTensorIndexI dims) m := by
  classical
  refine { forIn := ?_ }
  intro β _ _r init _f
  exact pure init

def convDim {r} (spDims kerDim : Dims r) (pad : Padding r) : Dims r :=
  .ofFn fun i =>
    let n : Int := spDims[i]
    let m : Int := kerDim[i]
    let lh := pad[i]
    n - m + lh.1 + lh.2

def convDimI {r} (spDims kerDim : DimsI r) (pad : Padding r) : DimsI r :=
  .ofFn fun i =>
    let ab := spDims[i]
    let cd := kerDim[i]
    let lh := pad[i]
    (ab.1 - lh.1 - cd.1, ab.2 + lh.2 - cd.2)


/-- Padding for reverse pass given kernel dimensions -/
def Padding.rev (kerDim : Dims r) (pad : Padding r) : Padding r :=
  .ofFn fun i => (kerDim[i] - pad[i].1, kerDim[i] - pad[i].2)

/-- Padding for reverse pass given kernel dimensions -/
def Padding.revI (pad : Padding r) (kerDim : DimsI r) : Padding r :=
  .ofFn fun i =>
    let m := kerDim[i].2 - kerDim[i].1
    (m - pad[i].1, m - pad[i].2)

def DimsI.rev (kerDim : DimsI r) : DimsI r :=
  .ofFn fun i => (-kerDim[i].2, -kerDim[i].1)

@[simp]
theorem DimsI.rev_rev (dims : DimsI r) : dims.rev.rev = dims := by
  sorry_proof

def XlaTensorIndexI.rev {dims : DimsI r} (i : XlaTensorIndexI dims)
    {outDims} (_houtDims : outDims = dims.rev := by (try simp_all); (try infer_var)) :
    XlaTensorIndexI outDims :=
{
  val := .ofFn fun d => -i[d]
  bounds := by
    sorry_proof
}


def DimsI.pad (spDim : DimsI r) (pad : Padding r) : DimsI r :=
  .ofFn fun i => (spDim[i].1 - pad[i].1, spDim[i].2 + pad[i].2)


def Dims.pad (spDim : Dims r) (pad : Padding r) : DimsI r :=
  .ofFn fun i => (0- pad[i].1, spDim[i] + pad[i].2)


@[simp]
theorem convDim_swap {r} (spDims kerDim : Dims r) (pad : Padding r) :
    convDim spDims (convDim spDims kerDim pad) pad
    =
    kerDim := by
  sorry_proof

@[simp]
theorem convDim_swap' {r} (spDims kerDim : Dims r) (pad : Padding r) :
    convDim (convDim spDims kerDim pad) kerDim (pad.rev kerDim)
    =
    spDims := by
  sorry_proof

@[simp]
theorem convDimI_swap {r} (spDims kerDim : DimsI r) (pad : Padding r) :
    convDimI spDims (convDimI spDims kerDim pad) pad
    =
    kerDim := by
  sorry_proof

@[simp]
theorem convDimI_swap' {r} (spDims kerDim : DimsI r) (pad : Padding r) :
    convDimI (convDimI spDims kerDim pad) kerDim.rev (pad.revI kerDim)
    =
    spDims := by
  sorry_proof

/--
Index operation during convolution.

Given output tensor index {given}`i` and kernel index {given}`j`, return the index used to
access the input tensor with dimensions {given}`spDim`. The result depends on
{lean}`i`, {lean}`j`, and {lean}`spDim`. -/
def XlaTensorIndex.convMap {kerDim}
    (spDim : Dims r) (pad : Padding r) (i : XlaTensorIndex (convDim spDim kerDim pad))
    (j : XlaTensorIndex kerDim) : XlaTensorIndexI (spDim.pad pad) :=
{
  val := .ofFn (fun a => i[a] + j[a] - pad[a].1)
  bounds := by
    sorry_proof
}

/-- Index operation during convolution.

Given output tensor index {given}`i` and kernel index {given}`j`, return the index used to
access the input tensor with dimensions {given}`spDim`. The result depends on
{lean}`i`, {lean}`j`, and {lean}`spDim`. -/
def XlaTensorIndexI.convMap {kerDim}
    (spDim : DimsI r) (pad : Padding r) (i : XlaTensorIndexI (convDimI spDim kerDim pad))
    (j : XlaTensorIndexI kerDim) : XlaTensorIndexI (spDim.pad pad) :=
{
  val := .ofFn (fun a => i[a] + j[a])
  bounds := by
    sorry_proof
}


def XlaTensorIndexI.InRange {r} {dims : DimsI r} (i : XlaTensorIndexI dims) (dims' : Dims r) : Prop :=
  ∀ (d : Fin r), (0 ≤ i[d]) ∧ (i[d] < dims'[d])

instance {r} {dims : DimsI r} (i : XlaTensorIndexI dims) (dims' : Dims r) :
    Decidable (i.InRange dims') := by unfold XlaTensorIndexI.InRange; infer_instance

def XlaTensorIndexI.cast {r} {dims : DimsI r}
    (i : XlaTensorIndexI dims) {dims' : Dims r} (h : i.InRange dims') : XlaTensorIndex dims' where
  val := i.val
  bounds := h


def XlaTensorIndexI.InRangeI {r} {dims : DimsI r} (i : XlaTensorIndexI dims) (dims' : DimsI r) : Prop :=
  ∀ (d : Fin r), (dims'[d].1 ≤ i[d]) ∧ (i[d] ≤ dims'[d].2)

instance {r} {dims : DimsI r} (i : XlaTensorIndexI dims) (dims' : DimsI r) :
    Decidable (i.InRangeI dims') := by unfold XlaTensorIndexI.InRangeI; infer_instance

def XlaTensorIndexI.castI {r} {dims : DimsI r}
    (i : XlaTensorIndexI dims) {dims' : DimsI r} (h : i.InRangeI dims') : XlaTensorIndexI dims' where
  val := i.val
  bounds := h


variable {R} [RealScalar R] [PlainDataType R]

def DataArrayN.get' {r} {dims : Dims r} {dims' : DimsI r}
    (x : R^[XlaTensorIndex dims]) (i : XlaTensorIndexI dims') : R :=

  if h : i.InRange dims then
    x[i.cast h]
  else
    0

def DataArrayN.get'' {r} {dims dims' : DimsI r}
    (x : R^[XlaTensorIndexI dims]) (i : XlaTensorIndexI dims') : R :=

  if h : i.InRangeI dims then
    x[i.castI h]
  else
    0



def XlaTensorIndexI.convMap' {kerDim : DimsI r}
    (spDim : DimsI r) (pad : Padding r) {outDim : DimsI r} (i : XlaTensorIndexI outDim)
    (j : XlaTensorIndexI kerDim)
    (_houtDim : outDim = convDimI spDim kerDim pad := by (try simp_all); (try infer_var))
     : XlaTensorIndexI (spDim.pad pad) :=
{
  val := .ofFn (fun a => i[a] + j[a])
  bounds := by
    sorry_proof
}


-- open XlaTensorIndexI in
-- @[simp]
-- theorem convMapI_swap'
-- {kerDim : DimsI r}
--     (spDim : DimsI r) (pad : Padding r) {outDim : DimsI r} (i : XlaTensorIndexI outDim)
--     (j : XlaTensorIndexI kerDim)
--     (houtDim : outDim = convDimI spDim kerDim pad := by (try simp_all); (try infer_var))
--     (k : XlaTensorIndexI (spDim.pad pad)) :
--   sorry = convMap' (convDimI spDims kerDim pad) (pad.revI kerDim) (k.pad pad) (j.rev)
--        (by ext d <;> simp_all[DimsI.pad]) := sorry


-- {r} (spDims kerDim : DimsI r) (pad : Padding r) :
--     convDimI (convDimI spDims kerDim pad) kerDim.rev (pad.revI kerDim)
--     =
--     spDims := by
--   ext <;> (simp[convDimI,DimsI.rev,Padding.revI]; ring)

def _root_.Vector.removeIds {n α} (a : Vector α n) (ids : Finset (Fin n))
    {m} (_h : m = n - ids.card := by (try simp); (try infer_var)) : Vector α m :=
  ⟨
    (let d := ids.map ⟨fun i => i.1, by intro i; aesop⟩
     (a.toArray.mapIdx (fun i v => (v, i))).filterMap (fun (d', i) => if i ∉ d then .some d' else none)),
    by
      sorry_proof
  ⟩


abbrev Dims.removeDims {r} (dims : Dims r) (d : Finset (Fin r))
    {s} (h : s = r - d.card := by (try simp); (try infer_var)) : Dims s := dims.removeIds d h


/-- Type used to index dimensions of a rank {lean}`r+2` tensor with {given}`r` spatial
dimensions. The spatial rank is {lean}`r`, with batch/feature dimensions. -/
inductive DimId (r : Nat) where
  | batchInputDim
  | featureOutputDim
  | spatialDim (i : Fin r)
deriving DecidableEq

instance {r} : IndexType (DimId r) (r + 2) := by
  classical
  sorry


/-- This equivalence determines which dimensions of a {lean}`r+2` tensor are spatial, batch and feature. -/
def DimMap (r : ℕ) := DimId r ≃ Fin (r+2)

def DimMap.batchDimId {r} (map : DimMap r) : Fin (r+2) := map.toFun .batchInputDim
def DimMap.featureDimId {r} (map : DimMap r) : Fin (r+2) := map.toFun .featureOutputDim
instance (r) : CoeFun (DimMap r) (fun _ => Fin r → (Fin (r+2))) :=
  ⟨fun f i => f.toFun (.spatialDim i)⟩

/-- Takes dimensions spec of a {lean}`r+2` rank tensor and returns the spatial dimensions. -/
def DimMap.spatialDims {r} (map : DimMap r) (dims : Dims (r+2)) : Dims r :=
  .ofFn fun i => dims[map.toFun (.spatialDim i)]

def DimMap.batchDim {r} (map : DimMap r) (dims : Dims (r+2)) : ℤ :=
  dims[map.toFun (.batchInputDim)]

def DimMap.featureDim {r} (map : DimMap r) (dims : Dims (r+2)) : ℤ :=
  dims[map.toFun (.featureOutputDim)]


open Set in
def DimMap.indexMap {r} (map : DimMap r) (dims : Dims (r+2))
   {dims'} (_hdims' : dims' = map.spatialDims dims := by (try simp); (try infer_var))
   {b} (_hb : b = map.batchDim dims := by (try simp); (try infer_var))
   {f} (_hf : f = map.featureDim dims := by (try simp); (try infer_var)) :
   XlaTensorIndex dims
   ≃
   Ico 0 b × Ico 0 f × XlaTensorIndex dims' :=
by
  classical
  sorry



section ArraNMissing

instance : HMul (Vector ℤ n) (Vector ℕ+ n) (Vector ℤ n) := ⟨fun x y => .ofFn fun i => x[i] * y[i]⟩
instance : HDiv (Vector ℤ n) (Vector ℕ+ n) (Vector ℤ n) := ⟨fun x y => .ofFn fun i => x[i] / y[i]⟩
instance : HMod (Vector ℤ n) (Vector ℕ+ n) (Vector ℤ n) := ⟨fun x y => .ofFn fun i => x[i] % y[i]⟩
instance [Mul α] : HMul (Vector α r) (Vector α r) (Vector α r) := ⟨fun x y => .ofFn fun i => x[i] * y[i]⟩
-- instance : HAdd (Vector ℤ n) (Vector ℕ+ n) (Vector ℤ n) := ⟨fun x y => x.mapIdx fun i xi => xi + y[i]⟩
-- instance : HSub (Vector ℤ n) (Vector ℕ+ n) (Vector ℤ n) := ⟨fun x y => x.mapIdx fun i xi => xi - y[i]⟩

instance [Max α] : Max (Vector α r) := ⟨fun x y => .ofFn fun i => x[i] ⊔ y[i]⟩
instance [Mod α] : Mod (Vector α r) := ⟨fun x y => .ofFn fun i => x[i] % y[i]⟩
instance [Div α] : Div (Vector α r) := ⟨fun x y => .ofFn fun i => x[i] / y[i]⟩

def _root_.Vector.toNat [CoeHTCT α Nat] (x : Vector α n) : Vector ℕ n := .ofFn fun i => x[i]
def _root_.Vector.toInt [CoeHTCT α Int] (x : Vector α n) : Vector ℤ n := .ofFn fun i => x[i]


@[simp]
theorem Vector.hmul_apply (x : Vector ℤ n) (y : Vector ℕ+ n) (i : Fin n) :
    (x * y)[i] = x[i] * y[i] := by
  sorry_proof

@[simp]
theorem Vector.hdiv_apply (x : Vector ℤ n) (y : Vector ℕ+ n) (i : Fin n) :
    (x / y)[i] = x[i] / y[i] := by
  sorry_proof

@[simp]
theorem Vector.hmod_apply (x : Vector ℤ n) (y : Vector ℕ+ n) (i : Fin n) :
    (x % y)[i] = x[i] % y[i] := by
  sorry_proof

@[simp]
def Dims.rank {r} (_dims : Dims r) : ℕ := r
@[simp]
def _root_.SciLean.Vector.size {n α} (_a : Vector α n) : ℕ := n


end ArraNMissing
