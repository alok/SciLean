import SciLean.Data.SparseMatrix.Basic
import SciLean.Data.Tensor.GpuTensor
import SciLean.Data.Tensor.Layout
import SciLean.FFI.Metal
import SciLean.VersoPrelude

namespace SciLean

/-- GPU-resident CSR sparse matrix (Float32 only). -/
structure GpuSparseMatrixCSR (I J : Type)
    {nI} [IndexType I nI] {nJ} [IndexType J nJ] where
  rowPtr : Metal.GpuBuffer
  colInd : Metal.GpuBuffer
  vals : Metal.GpuBuffer

namespace SparseMatrixCSR

variable {I J : Type} {nI} [IndexType I nI] {nJ} [IndexType J nJ]

/-- Upload CPU CSR to GPU buffers (Float32 values). -/
@[inline]
def toGpu (A : SparseMatrixCSR Float32 I J) : IO (GpuSparseMatrixCSR I J) := do
  let rowPtr ← Metal.GpuBuffer.fromByteArray A.rowPtr.byteData
  let colInd ← Metal.GpuBuffer.fromByteArray A.colInd.byteData
  let vals ← Metal.GpuBuffer.fromByteArray A.vals.byteData
  return ⟨rowPtr, colInd, vals⟩

end SparseMatrixCSR

namespace GpuSparseMatrixCSR

variable {I J : Type} {nI} [IndexType I nI] {nJ} [IndexType J nJ]

/-- Sparse SpMV on GPU: {lit}`y = A * x` for Float32. -/
@[inline]
def mulVec (A : GpuSparseMatrixCSR I J) (x : GpuTensor Float32 J) : IO (GpuTensor Float32 I) := do
  let yBuf ← Metal.GpuBuffer.csrSpmv A.rowPtr A.colInd A.vals x.data.buffer nI.toUSize
  let layout := TensorLayout.contiguous #[nI]
  return ⟨⟨yBuf, layout⟩⟩

end GpuSparseMatrixCSR

end SciLean
