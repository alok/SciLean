/-
Legacy interop helpers for migrating away from {name}`GpuBufferN`.
These are not imported by default; use explicitly during incremental porting.
-/
import SciLean.Data.Tensor.Basic
import SciLean.Data.Tensor.GpuBufferN
import SciLean.Data.Tensor.Basic
import SciLean.Data.IndexType.Shape

namespace SciLean

set_option linter.deprecated false

namespace LegacyInterop

variable {α : Type*} [PlainDataType α]
variable {ι : Type*} {n : ℕ} [IndexType ι n] [IndexTypeShape ι n]

@[deprecated (since := "v4.26") "Use StridedGpuTensor / GpuTensor directly; this is a temporary bridge."]
/-- Convert legacy contiguous {name}`GpuBufferN` to strided {name}`GpuTensor`. -/
def ofGpuBufferN (buf : GpuBufferN α ι) : GpuTensor α ι :=
  GpuTensor.fromContiguousBuffer (ι:=ι) buf.buffer (IndexTypeShape.shape (ι:=ι))

@[deprecated (since := "v4.26") "Use StridedGpuTensor / GpuTensor directly; this is a temporary bridge."]
/-- Convert strided {name}`GpuTensor` to legacy contiguous {name}`GpuBufferN` (copies if needed). -/
def toGpuBufferN (t : GpuTensor α ι) : IO (GpuBufferN α ι) := do
  let tContig ← GpuTensor.ensureContiguous t
  return ⟨tContig.data.buffer⟩

end LegacyInterop

end SciLean
