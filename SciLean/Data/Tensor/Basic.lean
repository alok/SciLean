/-
Copyright (c) 2024 SciLean contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: SciLean contributors
-/
import SciLean.FFI.Metal.GpuBufferView
import SciLean.Data.DataArray
import SciLean.Data.IndexType.Shape

namespace SciLean

/-!
Unified Tensor Type with Device as Constructor

Type-honest tensor where device is part of the constructor, not a separate type.
Device ID enables multi-GPU, pattern matching enables device dispatch.
-/

/-- Unified tensor type with device as constructor.
    Device ID enables multi-GPU support with compile-time knowledge.
    Shape is tracked in the type via IndexType. -/
inductive Tensor (α : Type) [PlainDataType α] (ι : Type) {n : outParam ℕ} [IndexType ι n] : Type where
  /-- CPU tensor stored in DataArrayN. -/
  | cpu (data : DataArrayN α ι)
  /-- Metal GPU tensor with device ID and layout-aware buffer view. -/
  | metal (deviceId : Nat := 0) (data : GpuBufferView α)
  -- | cuda (deviceId : Nat := 0) (data : CudaBufferView α)  -- future

namespace Tensor

variable {α : Type} [PlainDataType α]
variable {ι : Type} {n : ℕ} [IndexType ι n]

/-! ## Device Queries -/

/-- Which device this tensor is on. -/
inductive DeviceKind where
  | cpu
  | metal (id : Nat)
  -- | cuda (id : Nat)  -- future
  deriving DecidableEq, BEq, Repr, Inhabited

instance : ToString DeviceKind where
  toString
    | .cpu => "cpu"
    | .metal id => s!"metal:{id}"

/-- Get the device of a tensor. -/
def device (t : Tensor α ι) : DeviceKind :=
  match t with
  | .cpu _ => .cpu
  | .metal id _ => .metal id

/-- Check if tensor is on CPU. -/
def isCpu (t : Tensor α ι) : Bool :=
  match t with
  | .cpu _ => true
  | .metal _ _ => false

/-- Check if tensor is on Metal GPU. -/
def isMetal (t : Tensor α ι) : Bool :=
  match t with
  | .cpu _ => false
  | .metal _ _ => true

/-- Get Metal device ID (or 0 if CPU). -/
def metalDeviceId (t : Tensor α ι) : Nat :=
  match t with
  | .cpu _ => 0
  | .metal id _ => id

/-! ## Size and Shape -/

/-- Number of elements (from type). -/
def size (_ : Tensor α ι) : ℕ := n

/-- Size as USize. -/
def usize (_ : Tensor α ι) : USize := n.toUSize

/-! ## Device Matching -/

/-- Check if two tensors are on the same device. -/
def sameDevice (a b : Tensor α ι) : Bool :=
  match a, b with
  | .cpu _, .cpu _ => true
  | .metal id _, .metal id' _ => id == id'
  | _, _ => false

/-- Ensure two tensors are on the same device, panic otherwise. -/
def assertSameDevice (a b : Tensor α ι) (op : String := "operation") : Unit :=
  if !sameDevice a b then
    panic! s!"{op}: device mismatch ({a.device} vs {b.device})"
  else ()

/-! ## Construction -/

/-- Create CPU tensor from DataArrayN. -/
def ofCpu (data : DataArrayN α ι) : Tensor α ι := .cpu data

/-- Create Metal tensor from GpuBufferView. -/
def ofMetal (data : GpuBufferView α) (deviceId : Nat := 0) : Tensor α ι :=
  .metal deviceId data

/-- Create Metal tensor from contiguous GpuBuffer with shape. -/
def ofMetalBuffer (buffer : Metal.GpuBuffer) (shape : Array Nat) (deviceId : Nat := 0) :
    Tensor α ι :=
  .metal deviceId (GpuBufferView.fromContiguous buffer shape)

/-- Create Metal tensor from contiguous GpuBuffer using IndexTypeShape. -/
def ofMetalContiguous [IndexTypeShape ι n] (buffer : Metal.GpuBuffer) (deviceId : Nat := 0) :
    Tensor α ι :=
  .metal deviceId (GpuBufferView.fromContiguous buffer (IndexTypeShape.shape (ι:=ι)))

/-! ## Data Access -/

/-- Get CPU data (returns none if on GPU). -/
def cpuData? (t : Tensor α ι) : Option (DataArrayN α ι) :=
  match t with
  | .cpu data => some data
  | .metal _ _ => none

/-- Get Metal buffer view (returns none if on CPU). -/
def metalData? (t : Tensor α ι) : Option (GpuBufferView α) :=
  match t with
  | .cpu _ => none
  | .metal _ data => some data

/-- Get underlying Metal.GpuBuffer (returns none if on CPU). -/
def metalBuffer? (t : Tensor α ι) : Option Metal.GpuBuffer :=
  match t with
  | .cpu _ => none
  | .metal _ data => some data.buffer

/-- Get CPU data or throw error. -/
def cpuData! (t : Tensor α ι) : IO (DataArrayN α ι) :=
  match t with
  | .cpu data => pure data
  | .metal _ _ => throw (.userError "cpuData!: tensor is on Metal GPU")

/-- Get Metal buffer view or throw error. -/
def metalData! (t : Tensor α ι) : IO (GpuBufferView α) :=
  match t with
  | .cpu _ => throw (.userError "metalData!: tensor is on CPU")
  | .metal _ data => pure data

/-- Get underlying Metal.GpuBuffer or throw error. -/
def metalBuffer! (t : Tensor α ι) : IO Metal.GpuBuffer :=
  match t with
  | .cpu _ => throw (.userError "metalBuffer!: tensor is on CPU")
  | .metal _ data => pure data.buffer

/-! ## Device Transfers -/

/-- Transfer tensor to CPU (no-op if already on CPU). -/
def toCpu (t : Tensor α ι) : IO (Tensor α ι) :=
  match t with
  | .cpu data => pure (.cpu data)
  | .metal _ data => do
      -- Ensure contiguous, then download
      let contiguousData ← data.contiguous
      let bytes ← Metal.GpuBuffer.toByteArray contiguousData.buffer
      let arr : DataArray α := ⟨bytes, sorry_proof⟩
      let arrN : DataArrayN α ι := ⟨arr, sorry_proof⟩
      pure (.cpu arrN)

/-- Transfer tensor to Metal GPU (no-op if already on same device). -/
def toMetal [IndexTypeShape ι n] (t : Tensor α ι) (deviceId : Nat := 0) : IO (Tensor α ι) :=
  match t with
  | .cpu data => do
      let gpuBuf ← Metal.GpuBuffer.fromByteArray data.data.byteData
      pure (.metal deviceId (GpuBufferView.fromContiguous gpuBuf (IndexTypeShape.shape (ι:=ι))))
  | .metal id data =>
      if id == deviceId then pure (.metal id data)
      else do
        -- Transfer between GPUs: download then upload
        -- TODO: Use peer-to-peer transfer if available
        let contiguousData ← data.contiguous
        let bytes ← Metal.GpuBuffer.toByteArray contiguousData.buffer
        let gpuBuf ← Metal.GpuBuffer.fromByteArray bytes
        pure (.metal deviceId (GpuBufferView.fromContiguous gpuBuf (IndexTypeShape.shape (ι:=ι))))

/-- Transfer tensor to the same device as another tensor. -/
def toDevice [IndexTypeShape ι n] (t : Tensor α ι) (target : Tensor α ι) : IO (Tensor α ι) :=
  match target with
  | .cpu _ => t.toCpu
  | .metal id _ => t.toMetal id

/-! ## Layout Operations (Metal only) -/

variable {m k p : ℕ}

/-- Get layout rank (Metal only, returns 0 for CPU). -/
def rank (t : Tensor α ι) : Nat :=
  match t with
  | .cpu _ => 0  -- CPU doesn't track layout
  | .metal _ data => data.rank

/-- Check if contiguous (CPU always true, Metal checks layout). -/
def isContiguous (t : Tensor α ι) : Bool :=
  match t with
  | .cpu _ => true
  | .metal _ data => data.isContiguous

/-- Check if transposed (CPU always false, Metal checks layout). -/
def isTransposed (t : Tensor α ι) : Bool :=
  match t with
  | .cpu _ => false
  | .metal _ data => data.isTransposed

/-- Transpose last two dimensions (O(1) for Metal, copies for CPU).
    Only valid for 2D tensors. -/
def transpose (t : Tensor Float (Idx m × Idx k)) : Tensor Float (Idx k × Idx m) :=
  match t with
  | .cpu data =>
      -- CPU needs actual transpose (could optimize with layout tracking)
      let transposed : DataArrayN Float (Idx k × Idx m) :=
        ⊞ (j, i) => data[(i, j)]
      .cpu transposed
  | .metal id data =>
      -- Metal: O(1) stride manipulation
      .metal id data.transpose

/-- Make a contiguous copy if needed. -/
def contiguous (t : Tensor α ι) : IO (Tensor α ι) :=
  match t with
  | .cpu data => pure (.cpu data)
  | .metal id data => do
      let newData ← data.contiguous
      pure (.metal id newData)

/-- Ensure contiguous layout (copy if needed). -/
def ensureContiguous (t : Tensor α ι) : IO (Tensor α ι) :=
  match t with
  | .cpu data => pure (.cpu data)
  | .metal id data =>
      if data.isContiguous && data.layout.offset == 0 then
        pure (.metal id data)
      else do
        let newData ← data.contiguous
        pure (.metal id newData)

/-! ## Element-wise Operations -/

/-- Element-wise addition. Both tensors must be on same device. -/
def add [IndexTypeShape ι n] (a b : Tensor Float ι) : IO (Tensor Float ι) := do
  let _ := assertSameDevice a b "add"
  match a, b with
  | .cpu da, .cpu db =>
      let result : DataArrayN Float ι := ⊞ i => da[i] + db[i]
      pure (.cpu result)
  | .metal id bufA, .metal _ bufB => do
      let bufA ← bufA.contiguous
      let bufB ← bufB.contiguous
      let total := IndexTypeShape.numel (ι:=ι)
      let result ← Metal.GpuBuffer.add bufA.buffer bufB.buffer total.toUSize
      pure (.metal id (GpuBufferView.fromContiguous result (IndexTypeShape.shape (ι:=ι))))
  | _, _ => throw (.userError "add: device mismatch")

/-- Element-wise multiplication. Both tensors must be on same device. -/
def mul [IndexTypeShape ι n] (a b : Tensor Float ι) : IO (Tensor Float ι) := do
  let _ := assertSameDevice a b "mul"
  match a, b with
  | .cpu da, .cpu db =>
      let result : DataArrayN Float ι := ⊞ i => da[i] * db[i]
      pure (.cpu result)
  | .metal id bufA, .metal _ bufB => do
      let bufA ← bufA.contiguous
      let bufB ← bufB.contiguous
      let total := IndexTypeShape.numel (ι:=ι)
      let result ← Metal.GpuBuffer.mul bufA.buffer bufB.buffer total.toUSize
      pure (.metal id (GpuBufferView.fromContiguous result (IndexTypeShape.shape (ι:=ι))))
  | _, _ => throw (.userError "mul: device mismatch")

/-- Element-wise subtraction. Both tensors must be on same device. -/
def sub [IndexTypeShape ι n] (a b : Tensor Float ι) : IO (Tensor Float ι) := do
  let _ := assertSameDevice a b "sub"
  match a, b with
  | .cpu da, .cpu db =>
      let result : DataArrayN Float ι := ⊞ i => da[i] - db[i]
      pure (.cpu result)
  | .metal id bufA, .metal _ bufB => do
      let bufA ← bufA.contiguous
      let bufB ← bufB.contiguous
      let total := IndexTypeShape.numel (ι:=ι)
      let result ← Metal.GpuBuffer.sub bufA.buffer bufB.buffer total.toUSize
      pure (.metal id (GpuBufferView.fromContiguous result (IndexTypeShape.shape (ι:=ι))))
  | _, _ => throw (.userError "sub: device mismatch")

/-- Scalar multiplication. -/
def scale [IndexTypeShape ι n] (alpha : Float) (a : Tensor Float ι) : IO (Tensor Float ι) :=
  match a with
  | .cpu da =>
      let result : DataArrayN Float ι := ⊞ i => alpha * da[i]
      pure (.cpu result)
  | .metal id buf => do
      let buf ← buf.contiguous
      let total := IndexTypeShape.numel (ι:=ι)
      let result ← Metal.GpuBuffer.scale total.toUSize alpha buf.buffer
      pure (.metal id (GpuBufferView.fromContiguous result (IndexTypeShape.shape (ι:=ι))))

/-- Element-wise negation. -/
def neg [IndexTypeShape ι n] (a : Tensor Float ι) : IO (Tensor Float ι) :=
  scale (-1.0) a

/-- ReLU activation. -/
def relu [IndexTypeShape ι n] (a : Tensor Float ι) : IO (Tensor Float ι) :=
  match a with
  | .cpu da =>
      let result : DataArrayN Float ι := ⊞ i => if da[i] > 0 then da[i] else 0
      pure (.cpu result)
  | .metal id buf => do
      let buf ← buf.contiguous
      let total := IndexTypeShape.numel (ι:=ι)
      let result ← Metal.GpuBuffer.relu buf.buffer total.toUSize
      pure (.metal id (GpuBufferView.fromContiguous result (IndexTypeShape.shape (ι:=ι))))

/-! ## Reduction Operations -/

/-- Sum all elements. -/
def sum [IndexTypeShape ι n] [FoldM ι Id] (a : Tensor Float ι) : IO Float :=
  match a with
  | .cpu da =>
      -- Use IndexType.fold for pure summation
      pure (IndexType.fold (IndexType.Range.full (I:=ι)) 0.0 (fun i s => s + da[i]))
  | .metal _ buf => do
      let buf ← buf.contiguous
      let total := IndexTypeShape.numel (ι:=ι)
      Metal.GpuBuffer.sum buf.buffer total.toUSize

/-! ## Matrix Operations -/

/-- Matrix multiplication. Tensors must be on same device.
    A is m x k, B is k x p, result is m x p. -/
def gemm [FoldM (Idx k) Id] (A : Tensor Float (Idx m × Idx k)) (B : Tensor Float (Idx k × Idx p)) :
    IO (Tensor Float (Idx m × Idx p)) :=
  match A, B with
  | .cpu dataA, .cpu dataB =>
      -- Naive CPU GEMM using fold for inner product
      let result : DataArrayN Float (Idx m × Idx p) := ⊞ (i, j) =>
        IndexType.fold (IndexType.Range.full (I:=Idx k)) 0.0 (fun l s => s + dataA[(i, l)] * dataB[(l, j)])
      pure (.cpu result)
  | .metal id bufA, .metal _ bufB => do
      -- Use layout-aware GEMM
      let layout_A := bufA.layout
      let layout_B := bufB.layout
      let aRowStride := layout_A.rowStride
      let bRowStride := layout_B.rowStride
      let aTransposed := layout_A.isTransposed
      let bTransposed := layout_B.isTransposed
      let result ← Metal.GpuBuffer.gemmLayout
        bufA.buffer bufB.buffer
        m.toUSize k.toUSize p.toUSize
        aRowStride.toUSize bRowStride.toUSize
        layout_A.offset.toUSize layout_B.offset.toUSize
        aTransposed bTransposed
      pure (.metal id (GpuBufferView.fromContiguous result #[m, p]))
  | _, _ => throw (.userError "gemm: device mismatch")

end Tensor

/-! ## Backwards-Compatible CpuTensor (wraps DataArrayN) -/

/-- CPU tensor wrapping DataArrayN with device tracking.
    This maintains backwards compatibility with existing code. -/
structure CpuTensor (α : Type) [PlainDataType α] (ι : Type) {n : outParam ℕ} [IndexType ι n] : Type where
  /-- The underlying CPU array. -/
  data : DataArrayN α ι

namespace CpuTensor

variable {α : Type} [PlainDataType α]
variable {ι : Type} {n : ℕ} [IndexType ι n]

/-! ## Construction -/

/-- Create from DataArrayN. -/
@[inline]
def ofDataArrayN (arr : DataArrayN α ι) : CpuTensor α ι := ⟨arr⟩

/-- Extract DataArrayN. -/
@[inline]
def toDataArrayN (t : CpuTensor α ι) : DataArrayN α ι := t.data

/-! ## Zero-Cost Coercions -/

instance : Coe (DataArrayN α ι) (CpuTensor α ι) where
  coe := ofDataArrayN

instance : Coe (CpuTensor α ι) (DataArrayN α ι) where
  coe := toDataArrayN

/-! ## Basic Properties -/

/-- Number of elements. -/
def size (_ : CpuTensor α ι) : ℕ := n

/-- Size as USize. -/
def usize (_ : CpuTensor α ι) : USize := n.toUSize

/-! ## Element Access -/

/-- Get element at index. -/
@[inline]
def get (t : CpuTensor α ι) (i : ι) : α := t.data.get i

/-- Set element at index. -/
@[inline]
def set (t : CpuTensor α ι) (i : ι) (v : α) : CpuTensor α ι :=
  ⟨t.data.set i v⟩

instance : GetElem (CpuTensor α ι) ι α (fun _ _ => True) where
  getElem t i _ := t.get i

instance : SetElem (CpuTensor α ι) ι α (fun _ _ => True) where
  setElem t i v _ := t.set i v
  setElem_valid := sorry_proof

/-! ## Float Operations -/

/-- Element-wise addition -/
def add (a b : CpuTensor Float ι) : CpuTensor Float ι :=
  ⟨⊞ i => a.data[i] + b.data[i]⟩

/-- Element-wise multiplication -/
def mul (a b : CpuTensor Float ι) : CpuTensor Float ι :=
  ⟨⊞ i => a.data[i] * b.data[i]⟩

/-- Element-wise negation -/
def neg (a : CpuTensor Float ι) : CpuTensor Float ι :=
  ⟨⊞ i => -a.data[i]⟩

/-- Scalar multiplication -/
def smul (s : Float) (a : CpuTensor Float ι) : CpuTensor Float ι :=
  ⟨⊞ i => s * a.data[i]⟩

/-- ReLU activation (clips negative values to 0). -/
def relu (a : CpuTensor Float ι) : CpuTensor Float ι :=
  ⟨⊞ i => if a.data[i] > 0 then a.data[i] else 0⟩

/-! ## Algebra Instances -/

instance : Add (CpuTensor Float ι) where
  add := CpuTensor.add

instance : Neg (CpuTensor Float ι) where
  neg := CpuTensor.neg

instance : Sub (CpuTensor Float ι) where
  sub a b := a + (-b)

instance : SMul Float (CpuTensor Float ι) where
  smul := CpuTensor.smul

instance : HMul Float (CpuTensor Float ι) (CpuTensor Float ι) where
  hMul := CpuTensor.smul

end CpuTensor

/-! ## Convenience Aliases -/

/-- CPU tensor alias for cleaner types (subtype of unified Tensor). -/
abbrev CpuTensor' (α : Type) [PlainDataType α] (ι : Type) {n : outParam ℕ} [IndexType ι n] :=
  { t : Tensor α ι // t.isCpu }

/-- Metal tensor alias for cleaner types (subtype of unified Tensor). -/
abbrev MetalTensor (α : Type) [PlainDataType α] (ι : Type) {n : outParam ℕ} [IndexType ι n] :=
  { t : Tensor α ι // t.isMetal }

/-! ## Device Enum (backwards compatibility) -/

/-- Compute device for tensor operations (old API). -/
inductive Device where
  | cpu    -- CPU with DataArrayN storage
  | metal  -- GPU via Apple Metal
  | cuda   -- GPU via NVIDIA CUDA (future)
  deriving DecidableEq, BEq, Repr, Inhabited

namespace Device

instance : ToString Device where
  toString
    | .cpu => "cpu"
    | .metal => "metal"
    | .cuda => "cuda"

end Device

end SciLean
