import SciLean.Data.DataArray.DataArray
import SciLean.Data.IndexType
import SciLean.Data.IndexType.Fold
import SciLean.VersoPrelude

namespace SciLean

open scoped ArrayType

structure SparseMatrixCOO (R : Type*) [PlainDataType R] (I J : Type*)
    {nI} [IndexType I nI] {nJ} [IndexType J nJ] where
  rows : DataArray UInt32
  cols : DataArray UInt32
  vals : DataArray R
  h_rows : rows.size = vals.size
  h_cols : cols.size = vals.size

namespace SparseMatrixCOO

variable
  {R : Type*} [PlainDataType R] [Inhabited R] {I J : Type*}
  {nI} [IndexType I nI] {nJ} [IndexType J nJ]
  [DecidableEq UInt32]

structure Builder (R : Type*) [PlainDataType R] (I J : Type*)
    {nI} [IndexType I nI] {nJ} [IndexType J nJ] where
  rows : Array UInt32
  cols : Array UInt32
  vals : Array R

namespace Builder
@[inline]
def empty : Builder R I J :=
  ⟨#[], #[], #[]⟩

@[inline]
def add (b : Builder R I J) (i : I) (j : J) (v : R) : Builder R I J :=
  let r : UInt32 := UInt32.ofNat (toIdx i).1.toNat
  let c : UInt32 := UInt32.ofNat (toIdx j).1.toNat
  ⟨b.rows.push r, b.cols.push c, b.vals.push v⟩

@[inline]
def build (b : Builder R I J) : SparseMatrixCOO R I J :=
  ⟨DataArray.ofArray b.rows, DataArray.ofArray b.cols, DataArray.ofArray b.vals,
    sorry_proof, sorry_proof⟩

end Builder

@[inline]
def empty : SparseMatrixCOO R I J :=
  (Builder.empty (R:=R) (I:=I) (J:=J)).build

@[inline]
def nnz (A : SparseMatrixCOO R I J) : Nat :=
  A.vals.size

@[inline]
def add (A : SparseMatrixCOO R I J) (i : I) (j : J) (v : R) : SparseMatrixCOO R I J :=
  let r : UInt32 := UInt32.ofNat (toIdx i).1.toNat
  let c : UInt32 := UInt32.ofNat (toIdx j).1.toNat
  ⟨A.rows.push r, A.cols.push c, A.vals.push v, sorry_proof, sorry_proof⟩

@[inline]
def get [Zero R] [Add R] (A : SparseMatrixCOO R I J) (i : I) (j : J) : R := Id.run do
  let r : UInt32 := UInt32.ofNat (toIdx i).1.toNat
  let c : UInt32 := UInt32.ofNat (toIdx j).1.toNat
  let mut sum : R := Zero.zero
  for idx in fullRange (Idx A.vals.size) do
    let ri := A.rows.get (idx.cast A.h_rows.symm)
    let ci := A.cols.get (idx.cast A.h_cols.symm)
    if ri = r ∧ ci = c then
      sum := sum + A.vals.get idx
  return sum

end SparseMatrixCOO


structure SparseMatrixCSR (R : Type*) [PlainDataType R] (I J : Type*)
    {nI} [IndexType I nI] {nJ} [IndexType J nJ] where
  rowPtr : DataArray UInt32
  colInd : DataArray UInt32
  vals : DataArray R
  h_rowPtr : rowPtr.size = nI + 1
  h_colInd : colInd.size = vals.size

namespace SparseMatrixCSR

variable
  {R : Type*} [PlainDataType R] {I J : Type*}
  {nI} [IndexType I nI] {nJ} [IndexType J nJ]

partial def sumRowCol [Zero R] [Add R]
    (A : SparseMatrixCSR R I J) (col : UInt32) (start stop : Nat) : R :=
  let rec loop (k : Nat) (sum : R) : R :=
    if k < stop then
      let idx : Idx A.vals.size := ⟨k.toUSize, sorry_proof⟩
      let sum :=
        if A.colInd.get (idx.cast A.h_colInd.symm) = col then
          sum + A.vals.get idx
        else
          sum
      loop (k+1) sum
    else
      sum
  loop start Zero.zero

partial def sumRow [Zero R] [Add R] [Mul R]
    (A : SparseMatrixCSR R I J) (x : R^[J]) (start stop : Nat) : R :=
  let rec loop (k : Nat) (sum : R) : R :=
    if k < stop then
      let idx : Idx A.vals.size := ⟨k.toUSize, sorry_proof⟩
      let col := A.colInd.get (idx.cast A.h_colInd.symm)
      let j : J := IndexType.fromIdx ⟨col.toUSize, sorry_proof⟩
      let xj : R := DataArrayN.get x j
      loop (k+1) (sum + A.vals.get idx * xj)
    else
      sum
  loop start Zero.zero

@[inline]
def nnz (A : SparseMatrixCSR R I J) : Nat :=
  A.vals.size

@[inline]
def get [Zero R] [Add R] (A : SparseMatrixCSR R I J) (i : I) (j : J) : R := Id.run do
  let row : Nat := (toIdx i).1.toNat
  let col : UInt32 := UInt32.ofNat (toIdx j).1.toNat
  let start := (A.rowPtr.get ⟨row.toUSize, sorry_proof⟩).toNat
  let stop := (A.rowPtr.get ⟨(row+1).toUSize, sorry_proof⟩).toNat
  return sumRowCol A col start stop

@[inline]
def mulVec [Zero R] [Add R] [Mul R] [Fold.{_,0} I] (A : SparseMatrixCSR R I J) (x : R^[J]) : R^[I] := Id.run do
  let mut y : R^[I] := (Zero.zero : R^[I])
  for i in fullRange I do
    let row : Nat := (toIdx i).1.toNat
    let start := (A.rowPtr.get ⟨row.toUSize, sorry_proof⟩).toNat
    let stop := (A.rowPtr.get ⟨(row+1).toUSize, sorry_proof⟩).toNat
    let rowSum : R := sumRow A x start stop
    y := DataArrayN.set y i rowSum
  return y

end SparseMatrixCSR

namespace SparseMatrixCOO

variable
  {R : Type*} [PlainDataType R] [Inhabited R] {I J : Type*}
  {nI} [IndexType I nI] {nJ} [IndexType J nJ]

@[inline]
def toCSR (A : SparseMatrixCOO R I J) : SparseMatrixCSR R I J := Id.run do
  let nnz := A.vals.size

  let mut rowCounts : Array Nat := .mkEmpty nI
  for _ in [0:nI] do
    rowCounts := Array.push rowCounts 0
  for k in [0:nnz] do
    let r := (A.rows.get ⟨k.toUSize, sorry_proof⟩).toNat
    rowCounts := Array.set! rowCounts r (rowCounts[r]'sorry_proof + 1)

  let mut rowPtrNat : Array Nat := .mkEmpty (nI + 1)
  for _ in [0:nI+1] do
    rowPtrNat := Array.push rowPtrNat 0
  let mut acc := 0
  for i in [0:nI] do
    rowPtrNat := Array.set! rowPtrNat i acc
    acc := acc + rowCounts[i]'sorry_proof
  rowPtrNat := Array.set! rowPtrNat nI acc

  let offsets0 := rowPtrNat
  let defaultVal : R := default
  let colIndArr0 : Array UInt32 := Array.replicate nnz 0
  let valArr0 : Array R := Array.replicate nnz defaultVal

  let rec fill (k : Nat) (colIndArr : Array UInt32) (valArr : Array R) (offsets : Array Nat) :
      Array UInt32 × Array R × Array Nat :=
    match k with
    | 0 => (colIndArr, valArr, offsets)
    | k+1 =>
      let idx := nnz - (k+1)
      let r := (A.rows.get ⟨idx.toUSize, sorry_proof⟩).toNat
      let pos := offsets[r]'sorry_proof
      let colIndArr := Array.set! colIndArr pos (A.cols.get ⟨idx.toUSize, sorry_proof⟩)
      let valArr := Array.set! valArr pos (A.vals.get ⟨idx.toUSize, sorry_proof⟩)
      let offsets := Array.set! offsets r (pos + 1)
      fill k colIndArr valArr offsets

  let (colIndArr, valArr, _offsets) := fill nnz colIndArr0 valArr0 offsets0

  let rowPtrArr : Array UInt32 := rowPtrNat.map (fun x => UInt32.ofNat x)

  return {
    rowPtr := DataArray.ofArray rowPtrArr
    colInd := DataArray.ofArray colIndArr
    vals := DataArray.ofArray valArr
    h_rowPtr := sorry_proof
    h_colInd := sorry_proof
  }

end SparseMatrixCOO

end SciLean
