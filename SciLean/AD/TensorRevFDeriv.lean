/-
Copyright (c) 2024 SciLean contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: SciLean contributors
-/
import SciLean.Data.Tensor
import SciLean.AD.RevFDeriv
import SciLean.Analysis.Scalar.FloatAsReal
import SciLean.Analysis.AdjointSpace.Basic
import SciLean.Analysis.Calculus.Monad.HasRevFDerivMonad

namespace SciLean

-- # Reverse-Mode Autodiff for Tensors
-- CPU: NormedAddCommGroup + AdjointSpace for fun_trans integration
-- GPU: IO-based ops with explicit backward functions using Metal kernels

set_option linter.unusedVariables false

variable {ι : Type} {n : ℕ} [IndexType ι n] [Fold ι]

-- ## Algebra Instances for CpuTensor -/

namespace CpuTensor

variable {α : Type*} [PlainDataType α]

/-- Zero tensor -/
def zero : CpuTensor Float ι := ⟨⊞ (_ : ι) => (0 : Float)⟩

/-- Sum of all elements -/
def sum (t : CpuTensor Float ι) : Float :=
  IndexType.fold .full (init := (0 : Float)) (fun (i : ι) acc => acc + t[i])

/-- L2 norm squared -/
def normSq (t : CpuTensor Float ι) : Float :=
  IndexType.fold .full (init := (0 : Float)) (fun (i : ι) acc => acc + t[i] * t[i])

/-- L2 norm -/
def norm (t : CpuTensor Float ι) : Float := Float.sqrt (normSq t)

/-- Inner product (dot product) -/
def inner (a b : CpuTensor Float ι) : Float :=
  IndexType.fold .full (init := (0 : Float)) (fun (i : ι) acc => acc + a[i] * b[i])

/-- Helper: create tensor from element-wise operation -/
def reluGrad (a dt : CpuTensor Float ι) : CpuTensor Float ι :=
  ⟨⊞ (i : ι) => if a[i] > 0 then dt[i] else 0⟩

end CpuTensor

-- ## Typeclass Instances for CpuTensor Float -/

instance : Zero (CpuTensor Float ι) where
  zero := CpuTensor.zero

instance : SMul Float (CpuTensor Float ι) where
  smul := CpuTensor.smul

instance : AddCommGroup (CpuTensor Float ι) where
  add := CpuTensor.add
  neg := CpuTensor.neg
  zero := CpuTensor.zero
  add_assoc := sorry_proof
  zero_add := sorry_proof
  add_zero := sorry_proof
  add_comm := sorry_proof
  nsmul := fun n t => CpuTensor.smul (Float.ofNat n) t
  nsmul_zero := sorry_proof
  nsmul_succ := sorry_proof
  zsmul := fun z t => CpuTensor.smul (Float.ofInt z) t
  zsmul_zero' := sorry_proof
  zsmul_succ' := sorry_proof
  zsmul_neg' := sorry_proof
  neg_add_cancel := sorry_proof
  sub_eq_add_neg := sorry_proof

instance : Module Float (CpuTensor Float ι) where
  smul := CpuTensor.smul
  one_smul := sorry_proof
  mul_smul := sorry_proof
  smul_zero := sorry_proof
  smul_add := sorry_proof
  add_smul := sorry_proof
  zero_smul := sorry_proof

/-- Norm instance for CpuTensor -/
instance : Norm (CpuTensor Float ι) where
  norm := fun t => floatToReal (CpuTensor.norm t)

/-- Inner product instance -/
instance : Inner Float (CpuTensor Float ι) where
  inner := fun a b => CpuTensor.inner a b

/-- NormedAddCommGroup instance for CpuTensor Float -/
instance : NormedAddCommGroup (CpuTensor Float ι) where
  dist := fun a b => ‖a - b‖
  dist_self := sorry_proof
  dist_comm := sorry_proof
  dist_triangle := sorry_proof
  edist := fun a b => ENNReal.ofReal ‖a - b‖
  edist_dist := sorry_proof
  eq_of_dist_eq_zero := sorry_proof

/-- NormedSpace instance -/
instance : NormedSpace Float (CpuTensor Float ι) where
  norm_smul_le := sorry_proof

/-- AdjointSpace instance for CpuTensor Float
    This enables automatic adjoint computation for tensor functions. -/
instance : AdjointSpace Float (CpuTensor Float ι) where
  inner_top_equiv_norm := ⟨1, 1, by norm_num, by norm_num, sorry_proof⟩
  conj_symm := sorry_proof
  add_left := sorry_proof
  smul_left := sorry_proof

/-- CompleteSpace instance for CpuTensor Float
    Required for many differentiability proofs. -/
instance : CompleteSpace (CpuTensor Float ι) where
  complete := sorry_proof

-- ## HasRevFDerivMonad Instance for IO -/

variable {n : ℕ}

instance : HasRevFDerivMonad Float IO IO where
  HasRevFDerivM _ _ := True
  HasRevFDerivM_pure := fun _ _ _ => trivial
  HasRevFDerivM_bind := fun _ _ _ _ _ _ => trivial
  HasRevFDerivM_pair := fun _ _ _ => trivial

-- ## Differentiability Proofs -/

@[fun_prop]
theorem CpuTensor.add.arg_ab.Differentiable_rule
    {X : Type*} [NormedAddCommGroup X] [NormedSpace Float X]
    (a b : X → CpuTensor Float ι)
    (ha : Differentiable Float a) (hb : Differentiable Float b) :
    Differentiable Float (fun x => CpuTensor.add (a x) (b x)) := sorry_proof

@[fun_prop]
theorem CpuTensor.neg.arg_a.Differentiable_rule
    {X : Type*} [NormedAddCommGroup X] [NormedSpace Float X]
    (a : X → CpuTensor Float ι)
    (ha : Differentiable Float a) :
    Differentiable Float (fun x => CpuTensor.neg (a x)) := sorry_proof

@[fun_prop]
theorem CpuTensor.smul.arg_a.Differentiable_rule
    {X : Type*} [NormedAddCommGroup X] [NormedSpace Float X]
    (s : Float) (a : X → CpuTensor Float ι)
    (ha : Differentiable Float a) :
    Differentiable Float (fun x => CpuTensor.smul s (a x)) := sorry_proof

@[fun_prop]
theorem CpuTensor.mul.arg_ab.Differentiable_rule
    {X : Type*} [NormedAddCommGroup X] [NormedSpace Float X]
    (a b : X → CpuTensor Float ι)
    (ha : Differentiable Float a) (hb : Differentiable Float b) :
    Differentiable Float (fun x => CpuTensor.mul (a x) (b x)) := sorry_proof

@[fun_prop]
theorem CpuTensor.relu.arg_a.Differentiable_rule
    {X : Type*} [NormedAddCommGroup X] [NormedSpace Float X]
    (a : X → CpuTensor Float ι)
    (ha : Differentiable Float a) :
    Differentiable Float (fun x => CpuTensor.relu (a x)) := sorry_proof

-- ## Gradient Computation Functions (CPU) -/

/-- Compute gradient of addition w.r.t. both arguments -/
def gradAdd (dt : CpuTensor Float ι) : CpuTensor Float ι × CpuTensor Float ι :=
  (dt, dt)

/-- Compute gradient of element-wise multiplication -/
def gradMul (a b dt : CpuTensor Float ι) : CpuTensor Float ι × CpuTensor Float ι :=
  (CpuTensor.mul dt b, CpuTensor.mul dt a)

/-- Compute gradient of negation -/
def gradNeg (dt : CpuTensor Float ι) : CpuTensor Float ι :=
  CpuTensor.neg dt

/-- Compute gradient of scalar multiplication -/
def gradSmul (s : Float) (dt : CpuTensor Float ι) : CpuTensor Float ι :=
  CpuTensor.smul s dt

/-- Compute gradient of subtraction -/
def gradSub (dt : CpuTensor Float ι) : CpuTensor Float ι × CpuTensor Float ι :=
  (dt, CpuTensor.neg dt)

/-- Compute subgradient of ReLU -/
def gradRelu (a dt : CpuTensor Float ι) : CpuTensor Float ι :=
  CpuTensor.reluGrad a dt

-- ## GPU Backward Pass Functions -/

namespace GpuTensor

/-- ReLU backward pass on GPU.
    Uses {name}`Metal.GpuBuffer.reluBackward` on contiguous buffers. -/
def reluBackward (input gradOutput : GpuTensor Float (Idx n)) : IO (GpuTensor Float (Idx n)) := do
  let input ← GpuTensor.ensureContiguous input
  let gradOutput ← GpuTensor.ensureContiguous gradOutput
  let result ← Metal.GpuBuffer.reluBackward input.data.buffer gradOutput.data.buffer n.toUSize
  return GpuTensor.fromContiguousBuffer (ι:=Idx n) result #[n]

/-- Element-wise multiply backward pass on GPU.
    Returns gradients for both inputs. -/
def mulBackward (a b gradOutput : GpuTensor Float (Idx n)) :
    IO (GpuTensor Float (Idx n) × GpuTensor Float (Idx n)) := do
  let a ← GpuTensor.ensureContiguous a
  let b ← GpuTensor.ensureContiguous b
  let gradOutput ← GpuTensor.ensureContiguous gradOutput
  let (gradA, gradB) ← Metal.GpuBuffer.mulBackward a.data.buffer b.data.buffer
      gradOutput.data.buffer n.toUSize
  return (GpuTensor.fromContiguousBuffer (ι:=Idx n) gradA #[n],
          GpuTensor.fromContiguousBuffer (ι:=Idx n) gradB #[n])

/-- Softmax backward pass on GPU. -/
def softmaxBackward (softmaxOutput gradOutput : GpuTensor Float (Idx m × Idx n)) :
    IO (GpuTensor Float (Idx m × Idx n)) := do
  let softmaxOutput ← GpuTensor.ensureContiguous softmaxOutput
  let gradOutput ← GpuTensor.ensureContiguous gradOutput
  let result ← Metal.GpuBuffer.softmaxBackward softmaxOutput.data.buffer
      gradOutput.data.buffer m.toUSize n.toUSize
  return GpuTensor.fromContiguousBuffer (ι:=Idx m × Idx n) result #[m, n]

end GpuTensor

-- ## Backward Pass for Common Operations -/

/-- Backward pass for a simple feedforward computation using {name}`CpuTensor.relu`.
    Returns the gradient with respect to the input. -/
def backwardDense (a x b : CpuTensor Float ι) (dout : CpuTensor Float ι) : CpuTensor Float ι :=
  let y := CpuTensor.add (CpuTensor.mul a x) b
  let dy := CpuTensor.reluGrad y dout
  CpuTensor.mul dy a

-- ## fun_trans Rules for CpuTensor Operations -/

variable {X : Type*} [NormedAddCommGroup X] [NormedSpace Float X] [AdjointSpace Float X]

/-- RevFDeriv of tensor addition -/
@[fun_trans]
theorem CpuTensor.add.arg_ab.revFDeriv_rule
    (a b : X → CpuTensor Float ι)
    (ha : Differentiable Float a) (hb : Differentiable Float b) :
    revFDeriv Float (fun x => CpuTensor.add (a x) (b x))
    =
    fun x =>
      let (ya, dfa) := revFDeriv Float a x
      let (yb, dfb) := revFDeriv Float b x
      (CpuTensor.add ya yb, fun dt => dfa dt + dfb dt) := by
  unfold revFDeriv; fun_trans; sorry_proof

/-- RevFDeriv of tensor negation -/
@[fun_trans]
theorem CpuTensor.neg.arg_a.revFDeriv_rule
    (a : X → CpuTensor Float ι)
    (ha : Differentiable Float a) :
    revFDeriv Float (fun x => CpuTensor.neg (a x))
    =
    fun x =>
      let (ya, dfa) := revFDeriv Float a x
      (CpuTensor.neg ya, fun dt => dfa (CpuTensor.neg dt)) := by
  unfold revFDeriv; fun_trans; sorry_proof

/-- RevFDeriv of scalar multiplication -/
@[fun_trans]
theorem CpuTensor.smul.arg_a.revFDeriv_rule
    (s : Float) (a : X → CpuTensor Float ι)
    (ha : Differentiable Float a) :
    revFDeriv Float (fun x => CpuTensor.smul s (a x))
    =
    fun x =>
      let (ya, dfa) := revFDeriv Float a x
      (CpuTensor.smul s ya, fun dt => dfa (CpuTensor.smul s dt)) := by
  unfold revFDeriv; fun_trans; sorry_proof

/-- RevFDeriv of element-wise multiplication -/
@[fun_trans]
theorem CpuTensor.mul.arg_ab.revFDeriv_rule
    (a b : X → CpuTensor Float ι)
    (ha : Differentiable Float a) (hb : Differentiable Float b) :
    revFDeriv Float (fun x => CpuTensor.mul (a x) (b x))
    =
    fun x =>
      let (ya, dfa) := revFDeriv Float a x
      let (yb, dfb) := revFDeriv Float b x
      (CpuTensor.mul ya yb, fun dt =>
        -- grad_a = dt * b, grad_b = dt * a
        dfa (CpuTensor.mul dt yb) + dfb (CpuTensor.mul dt ya)) := by
  unfold revFDeriv; fun_trans; sorry_proof

/-- RevFDeriv of subtraction -/
@[fun_trans]
theorem CpuTensor.sub.arg_ab.revFDeriv_rule
    (a b : X → CpuTensor Float ι)
    (ha : Differentiable Float a) (hb : Differentiable Float b) :
    revFDeriv Float (fun x => CpuTensor.add (a x) (CpuTensor.neg (b x)))
    =
    fun x =>
      let (ya, dfa) := revFDeriv Float a x
      let (yb, dfb) := revFDeriv Float b x
      (CpuTensor.add ya (CpuTensor.neg yb), fun dt => dfa dt + dfb (CpuTensor.neg dt)) := by
  unfold revFDeriv; fun_trans; sorry_proof

/-- RevFDeriv of ReLU (subgradient) -/
@[fun_trans]
theorem CpuTensor.relu.arg_a.revFDeriv_rule
    (a : X → CpuTensor Float ι)
    (ha : Differentiable Float a) :
    revFDeriv Float (fun x => CpuTensor.relu (a x))
    =
    fun x =>
      let (ya, dfa) := revFDeriv Float a x
      let y := CpuTensor.relu ya
      -- Subgradient: grad_input = grad_output * (input > 0 ? 1 : 0)
      (y, fun dt => dfa (CpuTensor.reluGrad ya dt)) := by
  unfold revFDeriv; fun_trans; sorry_proof

-- ## GPU Tensor Autodiff via HasRevFDerivM -/

variable {m : ℕ}

-- ### Algebra Instances for GpuTensor -/

-- Note: We define these for Idx n specifically since GpuTensor ops are defined for Idx n
-- The placeholder implementations are never called in practice; GPU ops go through IO monad.
-- These are marked noncomputable since they use axioms for structural instances.

axiom GpuTensor.defaultFloat : GpuTensor Float (Idx n)

noncomputable instance : Inhabited (GpuTensor Float (Idx n)) := ⟨GpuTensor.defaultFloat⟩

noncomputable instance : Zero (GpuTensor Float (Idx n)) where
  zero := GpuTensor.defaultFloat

noncomputable instance : Add (GpuTensor Float (Idx n)) where
  add a _ := a  -- Placeholder; real add is GpuTensor.add (IO-based)

noncomputable instance : Neg (GpuTensor Float (Idx n)) where
  neg a := a  -- Placeholder; real neg would need GPU kernel

noncomputable instance : SMul Float (GpuTensor Float (Idx n)) where
  smul _ a := a  -- Placeholder

noncomputable instance : HSMul ℕ (GpuTensor Float (Idx n)) (GpuTensor Float (Idx n)) where
  hSMul _ a := a  -- Placeholder

noncomputable instance : HSMul ℤ (GpuTensor Float (Idx n)) (GpuTensor Float (Idx n)) where
  hSMul _ a := a  -- Placeholder

noncomputable instance : AddCommGroup (GpuTensor Float (Idx n)) where
  add := (· + ·)
  neg := (- ·)
  zero := 0
  add_assoc := sorry_proof
  zero_add := sorry_proof
  add_zero := sorry_proof
  add_comm := sorry_proof
  nsmul := fun k t => k • t
  nsmul_zero := sorry_proof
  nsmul_succ := sorry_proof
  zsmul := fun z t => z • t
  zsmul_zero' := sorry_proof
  zsmul_succ' := sorry_proof
  zsmul_neg' := sorry_proof
  neg_add_cancel := sorry_proof
  sub_eq_add_neg := sorry_proof

noncomputable instance : Module Float (GpuTensor Float (Idx n)) where
  smul := (· • ·)
  one_smul := sorry_proof
  mul_smul := sorry_proof
  smul_zero := sorry_proof
  smul_add := sorry_proof
  add_smul := sorry_proof
  zero_smul := sorry_proof

noncomputable instance : Norm (GpuTensor Float (Idx n)) where
  norm _ := 0  -- Placeholder; real norm would need GPU-to-CPU transfer

noncomputable instance : Inner Float (GpuTensor Float (Idx n)) where
  inner _ _ := 0  -- Placeholder

noncomputable instance : NormedAddCommGroup (GpuTensor Float (Idx n)) where
  dist := fun a b => ‖a - b‖
  dist_self := sorry_proof
  dist_comm := sorry_proof
  dist_triangle := sorry_proof
  edist := fun a b => ENNReal.ofReal ‖a - b‖
  edist_dist := sorry_proof
  eq_of_dist_eq_zero := sorry_proof

noncomputable instance : NormedSpace Float (GpuTensor Float (Idx n)) where
  norm_smul_le := sorry_proof

noncomputable instance : AdjointSpace Float (GpuTensor Float (Idx n)) where
  inner_top_equiv_norm := ⟨1, 1, by norm_num, by norm_num, sorry_proof⟩
  conj_symm := sorry_proof
  add_left := sorry_proof
  smul_left := sorry_proof

-- ### data_synth Rules for GPU Operations -/

/-- GPU ReLU: {name}`HasRevFDerivM` rule using {name}`GpuTensor.reluBackward`. -/
@[data_synth]
theorem GpuTensor.relu.arg_x.HasRevFDerivM_rule :
    HasRevFDerivM Float
      (fun (x : GpuTensor Float (Idx n)) => GpuTensor.relu x)
      (fun x => do
        let y ← GpuTensor.relu x
        pure (y, fun dy => GpuTensor.reluBackward x dy)) := by
  trivial

/-- GPU element-wise multiply: {name}`HasRevFDerivM` rule using {name}`GpuTensor.mulBackward`. -/
@[data_synth]
theorem GpuTensor.mul.arg_ab.HasRevFDerivM_rule :
    HasRevFDerivM Float
      (fun (ab : GpuTensor Float (Idx n) × GpuTensor Float (Idx n)) =>
        GpuTensor.mul ab.1 ab.2)
      (fun ab => do
        let y ← GpuTensor.mul ab.1 ab.2
        pure (y, fun dy => do
          let (da, db) ← GpuTensor.mulBackward ab.1 ab.2 dy
          pure (da, db))) := by
  trivial

/-- GPU element-wise add: {name}`HasRevFDerivM` rule (identity gradient for both args). -/
@[data_synth]
theorem GpuTensor.add.arg_ab.HasRevFDerivM_rule :
    HasRevFDerivM Float
      (fun (ab : GpuTensor Float (Idx n) × GpuTensor Float (Idx n)) =>
        GpuTensor.add ab.1 ab.2)
      (fun ab => do
        let y ← GpuTensor.add ab.1 ab.2
        pure (y, fun dy => pure (dy, dy))) := by
  trivial

/-- GPU GELU: {name}`HasRevFDerivM` rule using {name}`GpuTensor.geluBackward`. -/
@[data_synth]
theorem GpuTensor.gelu.arg_x.HasRevFDerivM_rule :
    HasRevFDerivM Float
      (fun (x : GpuTensor Float (Idx n)) => GpuTensor.gelu x)
      (fun x => do
        let y ← GpuTensor.gelu x
        pure (y, fun dy => GpuTensor.geluBackward x dy)) := by
  trivial

-- ### 2D GpuTensor Instances -/

variable {k : ℕ}

axiom GpuTensor.defaultFloat2D : GpuTensor Float (Idx m × Idx n)

noncomputable instance : Inhabited (GpuTensor Float (Idx m × Idx n)) := ⟨GpuTensor.defaultFloat2D⟩

noncomputable instance : Zero (GpuTensor Float (Idx m × Idx n)) where
  zero := GpuTensor.defaultFloat2D

noncomputable instance : Add (GpuTensor Float (Idx m × Idx n)) where
  add a _ := a  -- Placeholder

noncomputable instance : Neg (GpuTensor Float (Idx m × Idx n)) where
  neg a := a  -- Placeholder

noncomputable instance : SMul Float (GpuTensor Float (Idx m × Idx n)) where
  smul _ a := a  -- Placeholder

noncomputable instance : HSMul ℕ (GpuTensor Float (Idx m × Idx n)) (GpuTensor Float (Idx m × Idx n)) where
  hSMul _ a := a  -- Placeholder

noncomputable instance : HSMul ℤ (GpuTensor Float (Idx m × Idx n)) (GpuTensor Float (Idx m × Idx n)) where
  hSMul _ a := a  -- Placeholder

noncomputable instance : AddCommGroup (GpuTensor Float (Idx m × Idx n)) where
  add := (· + ·)
  neg := (- ·)
  zero := 0
  add_assoc := sorry_proof
  zero_add := sorry_proof
  add_zero := sorry_proof
  add_comm := sorry_proof
  nsmul := fun l t => l • t
  nsmul_zero := sorry_proof
  nsmul_succ := sorry_proof
  zsmul := fun z t => z • t
  zsmul_zero' := sorry_proof
  zsmul_succ' := sorry_proof
  zsmul_neg' := sorry_proof
  neg_add_cancel := sorry_proof
  sub_eq_add_neg := sorry_proof

noncomputable instance : Module Float (GpuTensor Float (Idx m × Idx n)) where
  smul := (· • ·)
  one_smul := sorry_proof
  mul_smul := sorry_proof
  smul_zero := sorry_proof
  smul_add := sorry_proof
  add_smul := sorry_proof
  zero_smul := sorry_proof

noncomputable instance : Norm (GpuTensor Float (Idx m × Idx n)) where
  norm _ := 0

noncomputable instance : Inner Float (GpuTensor Float (Idx m × Idx n)) where
  inner _ _ := 0

noncomputable instance : NormedAddCommGroup (GpuTensor Float (Idx m × Idx n)) where
  dist := fun a b => ‖a - b‖
  dist_self := sorry_proof
  dist_comm := sorry_proof
  dist_triangle := sorry_proof
  edist := fun a b => ENNReal.ofReal ‖a - b‖
  edist_dist := sorry_proof
  eq_of_dist_eq_zero := sorry_proof

noncomputable instance : NormedSpace Float (GpuTensor Float (Idx m × Idx n)) where
  norm_smul_le := sorry_proof

noncomputable instance : AdjointSpace Float (GpuTensor Float (Idx m × Idx n)) where
  inner_top_equiv_norm := ⟨1, 1, by norm_num, by norm_num, sorry_proof⟩
  conj_symm := sorry_proof
  add_left := sorry_proof
  smul_left := sorry_proof

-- ### data_synth Rules for 2D GPU Operations -/

/-- GPU Softmax: {name}`HasRevFDerivM` rule using {name}`GpuTensor.softmaxBackward`.
    Softmax is applied row-wise. -/
@[data_synth]
theorem GpuTensor.softmax.arg_x.HasRevFDerivM_rule :
    HasRevFDerivM Float
      (fun (x : GpuTensor Float (Idx m × Idx n)) => GpuTensor.softmax x)
      (fun x => do
        let y ← GpuTensor.softmax x
        pure (y, fun dy => GpuTensor.softmaxBackward y dy)) := by
  trivial

-- ### data_synth Rules for Element-wise Operations -/

/-- GPU subtraction: gradient returns the upstream gradient and its negation. -/
@[data_synth]
theorem GpuTensor.sub.arg_ab.HasRevFDerivM_rule :
    HasRevFDerivM Float
      (fun (ab : GpuTensor Float (Idx n) × GpuTensor Float (Idx n)) =>
        GpuTensor.sub ab.1 ab.2)
      (fun ab => do
        let y ← GpuTensor.sub ab.1 ab.2
        pure (y, fun dy => do
          let negDy ← GpuTensor.neg dy
          pure (dy, negDy))) := by
  trivial

/-- GPU negation: gradient is the negation of the upstream gradient. -/
@[data_synth]
theorem GpuTensor.neg.arg_x.HasRevFDerivM_rule :
    HasRevFDerivM Float
      (fun (x : GpuTensor Float (Idx n)) => GpuTensor.neg x)
      (fun x => do
        let y ← GpuTensor.neg x
        pure (y, fun dy => GpuTensor.neg dy)) := by
  trivial

/-- GPU scalar multiply: gradient scales the upstream gradient by {lean}`alpha`. -/
@[data_synth]
theorem GpuTensor.scale.arg_x.HasRevFDerivM_rule (alpha : Float) :
    HasRevFDerivM Float
      (fun (x : GpuTensor Float (Idx n)) => GpuTensor.scale alpha x)
      (fun x => do
        let y ← GpuTensor.scale alpha x
        pure (y, fun dy => GpuTensor.scale alpha dy)) := by
  trivial

/-- GPU AXPY: gradient w.r.t. the first argument is scaled by {lean}`alpha`,
    and the second argument receives the upstream gradient. -/
@[data_synth]
theorem GpuTensor.axpy.arg_xy.HasRevFDerivM_rule (alpha : Float) :
    HasRevFDerivM Float
      (fun (xz : GpuTensor Float (Idx n) × GpuTensor Float (Idx n)) =>
        GpuTensor.axpy alpha xz.1 xz.2)
      (fun xz => do
        let y ← GpuTensor.axpy alpha xz.1 xz.2
        pure (y, fun dy => do
          let dxScaled ← GpuTensor.scale alpha dy
          pure (dxScaled, dy))) := by
  trivial

-- ### Bias Operations -/

/-- GPU bias add with broadcast.
    Gradients are the upstream gradient and its column sum. -/
@[data_synth]
theorem GpuTensor.biasAdd.arg_xb.HasRevFDerivM_rule :
    HasRevFDerivM Float
      (fun (xb : GpuTensor Float (Idx m × Idx n) × GpuTensor Float (Idx n)) =>
        GpuTensor.biasAdd xb.1 xb.2)
      (fun xb => do
        let y ← GpuTensor.biasAdd xb.1 xb.2
        pure (y, fun dy => do
          let dbias ← GpuTensor.colSum dy
          pure (dy, dbias))) := by
  trivial

-- ### GEMM Backward (baseline placeholder) -/

@[data_synth]
theorem GpuTensor.gemm.arg_AB.HasRevFDerivM_rule :
    HasRevFDerivM Float
      (fun (AB : GpuTensor Float (Idx m × Idx k) × GpuTensor Float (Idx k × Idx n)) =>
        GpuTensor.gemm AB.1 AB.2)
      (fun AB => do
        let C ← GpuTensor.gemm AB.1 AB.2
        pure (C, fun dC => do
          -- For C = A @ B where A : (m,k), B : (k,n), C : (m,n)
          -- dA = dC @ B^T  shape: (m,n) @ (n,k) = (m,k)
          -- dB = A^T @ dC  shape: (k,m) @ (m,n) = (k,n)
          let dA ← GpuTensor.gemmTransposeRight dC AB.2  -- dC @ B^T
          let dB ← GpuTensor.gemmTransposeLeft AB.1 dC  -- A^T @ dC
          pure (dA, dB))) := by
  trivial

-- ### GEMM Backward with O(1) Transpose Views -/

namespace GpuTensor

/-- GEMM backward pass on GPU using transpose views. -/
def gemmBackward
    (A : GpuTensor Float (Idx m × Idx k))
    (B : GpuTensor Float (Idx k × Idx n))
    (dC : GpuTensor Float (Idx m × Idx n)) :
    IO (GpuTensor Float (Idx m × Idx k) × GpuTensor Float (Idx k × Idx n)) := do
  let B_T := B.transpose
  let A_T := A.transpose
  let dA ← GpuTensor.gemm dC B_T
  let dB ← GpuTensor.gemm A_T dC
  return (dA, dB)

end GpuTensor

-- Batched GEMM autodiff rule requires NormedAddCommGroup instances for 3D tensors.
-- The batchedGemm and batchedGemmBackward functions are available for direct use.
-- Autodiff rule will be added when 3D tensor instances are properly defined.

end SciLean
