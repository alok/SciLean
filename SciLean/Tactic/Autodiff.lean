import SciLean.Analysis.Calculus.RevCDeriv
import SciLean.Analysis.Calculus.RevFDeriv

import SciLean.Meta.SimpCore

import SciLean.Tactic.LetNormalize2
import SciLean.Tactic.LFunTrans

/-!

-/

namespace SciLean.Tactic

open Lean Meta Elab Tactic Mathlib.Meta.FunTrans Lean.Parser.Tactic in
syntax (name := lautodiffConvStx) "autodiff" optConfig (discharger)?
  (" [" withoutPosition((simpStar <|> simpErase <|> simpLemma),*) "]")? : conv

open Lean Meta Elab Tactic Mathlib.Meta.FunTrans Lean.Parser.Tactic in
syntax (name := lautodiffTacticStx) "autodiff" optConfig (discharger)?
  (" [" withoutPosition((simpStar <|> simpErase <|> simpLemma),*) "]")? : tactic

macro_rules
| `(conv| autodiff $cfg $[$disch]?  $[[$a,*]]?) => do
  if a.isSome then
    `(conv| ((try unfold deriv fgradient adjointFDeriv); -- todo: investigate why simp sometimes does not unfold and remove this line
             lfun_trans $cfg $[$disch]? only $[[deriv, fgradient, adjointFDeriv, simp_core, $a,*]]?))
  else
    `(conv| ((try unfold deriv fgradient adjointFDeriv);
             lfun_trans $cfg $[$disch]? only [deriv, fgradient, adjointFDeriv, simp_core]))

macro_rules
| `(tactic| autodiff $cfg $[$disch]?  $[[$a,*]]?) => do
  if a.isSome then
    `(tactic| ((try unfold deriv fgradient adjointFDeriv);
               lfun_trans $cfg $[$disch]? only $[[deriv, fgradient, adjointFDeriv, simp_core, $a,*]]?))
  else
    `(tactic| ((try unfold deriv fgradient adjointFDeriv);
               lfun_trans $cfg $[$disch]? only [deriv, fgradient, adjointFDeriv, simp_core]))

-- open Lean Meta
-- simproc_decl lift_lets_simproc (_) := fun e => do
--   let E ← inferType e
--   if (← Meta.isClass? E).isSome then return .continue
--   let e ← e.liftLets2 (fun xs b => do
--       if xs.size ≠ 0
--       then mkLetFVars xs (← Simp.dsimp b)
--       else pure b) {}
--   return .visit {expr:=e}

-- -- todo: add option, discharger, only and [...] syntax
-- macro "autodiff" : conv =>
--   `(conv| (fun_trans (config:={zeta:=false,singlePass:=true}) (disch:=sorry_proof) only [ftrans_simp,lift_lets_simproc, scalarGradient, scalarCDeriv];
--            simp (config:={zeta:=false,failIfUnchanged:=false}) only [ftrans_simp,lift_lets_simproc]))

-- macro "autodiff" : tactic =>
--   `(tactic| (fun_trans (config:={zeta:=false,singlePass:=true}) (disch:=sorry_proof) only [ftrans_simp,lift_lets_simproc, scalarGradient, scalarCDeriv];
--              try simp (config:={zeta:=false,failIfUnchanged:=false}) only [ftrans_simp,lift_lets_simproc]))
