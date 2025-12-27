/-
Copyright (c) 2024 SciLean contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: SciLean contributors
-/
import SciLean.Data.Tensor.Basic

open SciLean

/-!
# Unified Tensor Type Tests

Tests for the new unified `Tensor` type with device as constructor.
This type provides compile-time device tracking with device ID support.
-/

/-! ## Type Elaboration Tests -/

-- DeviceKind type
#check Tensor.DeviceKind.cpu
#check Tensor.DeviceKind.metal 0
#check (Tensor.DeviceKind.cpu : Tensor.DeviceKind)

-- Tensor constructors
#check Tensor.cpu
#check Tensor.metal

-- Tensor type
#check Tensor Float (Idx 3)
#check Tensor Float (Idx 10 × Idx 5)

/-! ## CPU Tensor Construction -/

-- Create CPU tensor from DataArrayN
def cpuTensor3 : Tensor Float (Idx 3) :=
  .cpu ⊞[1.0, 2.0, 3.0]

-- ofCpu convenience
def cpuTensor3' : Tensor Float (Idx 3) :=
  Tensor.ofCpu ⊞[1.0, 2.0, 3.0]

/-! ## Device Query Tests -/

-- Device checks
#check (cpuTensor3.device : Tensor.DeviceKind)
#check cpuTensor3.isCpu
#check cpuTensor3.isMetal

-- Compile-time property verification
example : cpuTensor3.isCpu = true := by native_decide
example : cpuTensor3.isMetal = false := by native_decide

/-! ## Size and Shape -/

#check cpuTensor3.size
#check cpuTensor3.usize

example : cpuTensor3.size = 3 := by native_decide

/-! ## Data Access -/

-- Option-based access
#check (cpuTensor3.cpuData? : Option (Float^[Idx 3]))
#check (cpuTensor3.metalData? : Option (GpuBufferView Float))

-- CPU data retrieval works
example : cpuTensor3.cpuData?.isSome = true := by native_decide
example : cpuTensor3.metalData?.isNone = true := by native_decide

/-! ## Same Device Checks -/

def cpuTensor3b : Tensor Float (Idx 3) := .cpu ⊞[4.0, 5.0, 6.0]

example : cpuTensor3.sameDevice cpuTensor3b = true := by native_decide

/-! ## Operation Type Checking -/

variable (a b : Tensor Float (Idx 3))

-- Element-wise operations (in IO)
#check (Tensor.add a b : IO (Tensor Float (Idx 3)))
#check (Tensor.mul a b : IO (Tensor Float (Idx 3)))
#check (Tensor.sub a b : IO (Tensor Float (Idx 3)))
#check (Tensor.scale 2.0 a : IO (Tensor Float (Idx 3)))
#check (Tensor.neg a : IO (Tensor Float (Idx 3)))
#check (Tensor.relu a : IO (Tensor Float (Idx 3)))

-- Reduction
#check (Tensor.sum a : IO Float)

/-! ## Matrix Operations -/

variable (A : Tensor Float (Idx 4 × Idx 3))
variable (B : Tensor Float (Idx 3 × Idx 5))

#check (Tensor.gemm A B : IO (Tensor Float (Idx 4 × Idx 5)))

/-! ## Transfer Operations -/

#check (a.toCpu : IO (Tensor Float (Idx 3)))
#check (a.toMetal 0 : IO (Tensor Float (Idx 3)))

/-! ## Layout Operations (2D tensors) -/

variable (M : Tensor Float (Idx 4 × Idx 3))

#check (M.transpose : Tensor Float (Idx 3 × Idx 4))
#check M.isContiguous
#check M.isTransposed

/-! ## CPU Operations Verification -/

-- Create test tensors
def testA : Tensor Float (Idx 3) := .cpu ⊞[1.0, 2.0, 3.0]
def testB : Tensor Float (Idx 3) := .cpu ⊞[4.0, 5.0, 6.0]

-- Verify element access via cpuData?
example : testA.cpuData?.map (·[(0 : Idx 3)]) = some 1.0 := by native_decide
example : testA.cpuData?.map (·[(1 : Idx 3)]) = some 2.0 := by native_decide
example : testA.cpuData?.map (·[(2 : Idx 3)]) = some 3.0 := by native_decide

/-! ## Transpose Test (CPU) -/

def matrix2x3 : Tensor Float (Idx 2 × Idx 3) :=
  .cpu ⊞[(0,0) => 1.0, (0,1) => 2.0, (0,2) => 3.0,
         (1,0) => 4.0, (1,1) => 5.0, (1,2) => 6.0]

-- Transpose is now 3x2
def matrix3x2 : Tensor Float (Idx 3 × Idx 2) := matrix2x3.transpose

-- Check shapes
example : matrix2x3.size = 6 := by native_decide
example : matrix3x2.size = 6 := by native_decide

-- Verify transpose correctness for CPU
-- After transpose: (0,0) gets (0,0), (0,1) gets (1,0), etc.
-- Original: row 0 = [1,2,3], row 1 = [4,5,6]
-- Transposed: row 0 = [1,4], row 1 = [2,5], row 2 = [3,6]
example : matrix3x2.cpuData?.map (·[((0 : Idx 3), (0 : Idx 2))]) = some 1.0 := by native_decide
example : matrix3x2.cpuData?.map (·[((0 : Idx 3), (1 : Idx 2))]) = some 4.0 := by native_decide
example : matrix3x2.cpuData?.map (·[((1 : Idx 3), (0 : Idx 2))]) = some 2.0 := by native_decide
example : matrix3x2.cpuData?.map (·[((1 : Idx 3), (1 : Idx 2))]) = some 5.0 := by native_decide

/-! ## Subtype Aliases -/

#check CpuTensor' Float (Idx 3)
#check MetalTensor Float (Idx 3)
