# SciLean Development Guidelines

## Project Philosophy
**Verification and sorry-free is NOT the point of this repo!** This is a scientific computing library focused on practical functionality. Technical debt via `sorry_proof` is acceptable. Priorities:
1. Keep up with Lean 4 releases (currently v4.26)
2. BLAS benchmarks and performance
3. Gradient/autodiff tests
4. Better ML support (see lean-mlir for inspiration)

## Backend Architecture (Future)
SciLean uses dependent types (`Float^[784]`, `Float^[128, 784]`) wrapping compute backends:

```
┌──────────────────────────────────────────────┐
│  SciLean Dependent Types + Autodiff          │
│  DataArrayN, gradients, fun_trans            │
└──────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│  Backend Typeclass (TensorBackend)           │
│  gemm, axpy, softmax, conv2d, ...            │
└──────────────────────────────────────────────┘
       │            │            │
       ▼            ▼            ▼
   ┌───────┐   ┌────────┐   ┌────────┐
   │ BLAS  │   │ Metal  │   │ CUDA   │
   │ (CPU) │   │ (MPS)  │   │(future)│
   └───────┘   └────────┘   └────────┘
```

**Current state:**
- LeanBLAS: BLAS Level 1-3 bindings (CPU) ✅
- Metal: Started in `SciLean/FFI/Metal.lean` ✅
- CUDA: Not started

**Related projects:**
- [TensorLib](https://github.com/leanprover/TensorLib): Verified tensors, .npy file support (steal this!)
- [c-libtorch](https://github.com/lighttransport/c-libtorch): C bindings to PyTorch (minimal, needs work)
- [tch-rs](https://github.com/LaurentMazare/tch-rs): Rust libtorch bindings (reference for API design)
- [Hasktorch](https://github.com/hasktorch/hasktorch): Haskell libtorch bindings

## Local Dependencies
- **LeanBLAS** (`../LeanBLAS`): Vendored locally for active development. Expect frequent changes to BLAS bindings, Level 1-3 operations, and FFI layer. Keep in sync with SciLean's mathlib version.
- **LeanPlot** (`../LeanPlot`): Local visualization library

## Build Commands
- Build entire project: `lake build`
- Run tests: `lake test`
- Test a specific file: `lake build Test.file_name` (e.g., `lake build Test.calculus.revFDeriv_test`)
- Run an example: `lake build ExampleName && .lake/build/bin/ExampleName` (e.g., `lake build HarmonicOscillator && .lake/build/bin/HarmonicOscillator`)

## Code Style Guidelines
- **Indentation**: 2 spaces
- **Naming**: CamelCase for types/classes, snake_case for variables, Unicode for mathematical symbols
- **Imports**: Organized at top by dependency, open primary namespace
- **Types**: Use typeclasses for mathematical abstractions, explicit type constraints where needed
- **Documentation**: `/--` blocks with markdown, mathematical notation where appropriate
- **Attributes**: Use `@[simp]`, `@[fun_trans]`, `@[data_synth]` for optimization rules
- **Error handling**: Use dependent types and type constraints over runtime checks

## Conventions
- Proofs use the `by` syntax with tactic blocks
- Mathematical properties follow the theorem naming pattern `operation.arg_name.property_rule`
- Make heavy use of metaprogramming for tactics and automation
- Clear distinction between forward and reverse mode differentiation in naming
- Add existing imports as comments when disabling them

## Verso Documentation
The project uses `doc.verso` for documentation. When writing doc comments:
- Use `{name}` role for declared names that can be resolved
- Use `{lean}` role for Lean expressions/syntax
- Use `{given}` role to declare variables in scope (e.g., `{given}`\`n\` then use `{lean}`\`n\`)
- Use `{lit}` role only for literal text that shouldn't be resolved

Build docs: `cd verso-docs && lake build && .lake/build/bin/generate-docs`

## Lean 4 Tips
- **Float infinity**: Lean 4 stdlib doesn't have `Float.inf`. Define as:
  ```lean
  def Float.inf : Float := 1.0 / 0.0
  def Float.negInf : Float := -1.0 / 0.0
  ```
  These are proper IEEE 754 infinity values for min/max tracking.

  ---

  use lean-lsp-mcp hover on nested src code after writing it to ENSURE its in
  the right namespace. like `Float.inf` may need to be `_root_.Float.inf`.

## PlainDataType Derive Handler

Fast derive handler for POD (Plain Old Data) serialization:

```lean
structure Vec3 where
  x : Float
  y : Float
  z : Float
  deriving PlainDataType  -- Zero overhead!
```

**Location:** `SciLean/Data/DataArray/DerivePlainDataTypeFast.lean`

**Key features:**
- Generates `@[inline]` ByteType with direct field access
- Compile-time byte offset computation via `evalExpr`
- Supports nested structures up to 8 fields
- Verified: derived instances match raw tuple performance

## Strided Tensor System ✅

PyTorch-style strided tensors with O(1) view operations. Enables efficient GEMM backward without data copies.

### Core Types

```lean
-- N-D layout with strides (Array Nat for FFI compatibility)
structure TensorLayout where
  shape : Array Nat      -- e.g., [batch, channels, height, width]
  strides : Array Nat    -- bytes between elements per dimension
  offset : Nat := 0

-- GPU buffer + layout metadata
structure StridedGpuBuffer (α : Type*) [PlainDataType α] where
  buffer : Metal.GpuBuffer
  layout : TensorLayout

-- Type-safe wrapper (shape in type via IndexType)
structure StridedGpuTensor (α : Type*) [PlainDataType α] (ι : Type*)
    {n : outParam ℕ} [IndexType ι n] where
  data : StridedGpuBuffer α
```

### O(1) View Operations

```lean
-- Transpose: swap strides, no data movement
def TensorLayout.transpose (l : TensorLayout) : TensorLayout :=
  let n := l.rank
  let newShape := l.shape.set! (n-2) l.shape[n-1]! |>.set! (n-1) l.shape[n-2]!
  let newStrides := l.strides.set! (n-2) l.strides[n-1]! |>.set! (n-1) l.strides[n-2]!
  ⟨newShape, newStrides, l.offset⟩

-- Permute, slice, squeeze, unsqueeze - all O(1)
```

### GEMM Backward (The Goal)

```lean
-- For C = A @ B:
-- dA = dC @ B^T   (O(1) transpose view!)
-- dB = A^T @ dC   (O(1) transpose view!)

@[data_synth]
theorem StridedGpuTensor.gemm.arg_AB.HasRevFDerivM_rule :
    HasRevFDerivM Float
      (fun (AB : StridedGpuTensor Float (Idx m × Idx k) ×
                 StridedGpuTensor Float (Idx k × Idx n)) =>
        StridedGpuTensor.gemm AB.1 AB.2)
      (fun AB => do
        let C ← StridedGpuTensor.gemm AB.1 AB.2
        pure (C, fun dC => do
          let B_T := AB.2.transpose  -- O(1), no copy
          let A_T := AB.1.transpose  -- O(1), no copy
          let dA ← StridedGpuTensor.gemm dC B_T
          let dB ← StridedGpuTensor.gemm A_T dC
          pure (dA, dB))) := by trivial
```

### Metal FFI

MPS supports strided access via `rowBytes` parameter:
```objc
MPSMatrixDescriptor* desc = [MPSMatrixDescriptor
    matrixDescriptorWithRows: m
    columns: k
    rowBytes: row_stride * sizeof(float)  // non-contiguous stride
    dataType: MPSDataTypeFloat32];
```

### Implementation Status

| File | Status |
|------|--------|
| `SciLean/Data/Tensor/Layout.lean` | ✅ TensorLayout with O(1) ops |
| `SciLean/FFI/Metal/StridedBuffer.lean` | ✅ StridedGpuBuffer |
| `SciLean/Data/Tensor/StridedGpuTensor.lean` | ✅ Type-safe wrapper |
| `Metal/metal_backend.mm` | ✅ copyStrided, gemmTN, gemmNT |
| `SciLean/FFI/Metal.lean` | ✅ FFI bindings |
| `SciLean/AD/TensorRevFDeriv.lean` | ✅ GEMM backward rule |
| `test/gpu_strided_tensor.lean` | ✅ All tests passing |

### Migration (TODO)

- Port existing ops to use StridedGpuTensor
- Remove legacy GpuTensor/GpuBufferN
- All tensors carry layout metadata from creation

## GPU Observability / Tracing

Hook into Lean's compiler tracer for GPU operation visibility.

### Trace Classes
```lean
-- Register in SciLean/FFI/Metal/Trace.lean
initialize registerTraceClass `GPU.metal
initialize registerTraceClass `GPU.metal.alloc
initialize registerTraceClass `GPU.metal.transfer
initialize registerTraceClass `GPU.metal.compute
initialize registerTraceClass `GPU.metal.batch
```

### Usage Pattern
```lean
-- Wrap GPU ops with withTraceNode for hierarchical traces
def tracedGemm (A B : GpuBuffer) (m k n : USize) : IO GpuBuffer := do
  withTraceNode `GPU.metal.compute
    (fun r => return s!"[{ExceptToEmoji.toEmoji r}] GEMM {m}x{k}x{n}") do
    Metal.GpuBuffer.gemm A B m k n
```

### Enable Tracing
```lean
-- In Lean files
set_option trace.GPU.metal true
set_option trace.GPU.metal.compute true

-- From command line
lake env lean -D trace.GPU.metal=true MyFile.lean
```

### Timing Integration
Use `Aesop.time` for nano-precision timing, accumulate in `Batteries.RBMap`:
```lean
def timeGPU (key : String) (op : IO α) : IO α := do
  let (result, nanos) ← Aesop.time op
  trace[GPU.metal.timing] "{key}: {nanos.print}"
  return result
```

---

Wrap math in docstrings in backticks or else you'll get weird parse errors.
