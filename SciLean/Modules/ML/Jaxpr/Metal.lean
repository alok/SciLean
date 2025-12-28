import SciLean.Modules.ML.Jaxpr.AST

/-!
# Jaxpr → Metal Shader Code Generation

This module translates Jaxpr programs containing fusible elementwise operations
into Metal compute shaders. The key benefit is **kernel fusion**: multiple
operations are combined into a single GPU kernel, reducing memory bandwidth.

Example usage: `jaxprToMetal "fused_relu_add" jaxpr` where jaxpr defines
`in x y; let z := relu x; let w := add z y; out w`

This generates a single Metal kernel that loads x, y, computes relu(x) + y
without intermediate memory writes.
-/

namespace SciLean.ML.Jaxpr

/-- Metal expression representation (subset for elementwise ops). -/
inductive MetalExpr where
  | var (name : String)
  | lit (value : Float)
  | litInt (value : Int)
  | call (fn : String) (args : List MetalExpr)
  | binop (op : String) (lhs rhs : MetalExpr)
  | unop (op : String) (arg : MetalExpr)
  | ternary (cond thenE elseE : MetalExpr)
  | index (arr : String) (idx : MetalExpr)
  deriving Repr, Inhabited

/-- Render MetalExpr to Metal shader code. -/
partial def MetalExpr.render : MetalExpr → String
  | .var name => name
  | .lit v => s!"{v}f"
  | .litInt v => s!"{v}"
  | .call fn args => s!"{fn}({String.intercalate ", " (args.map render)})"
  | .binop op lhs rhs => s!"({lhs.render} {op} {rhs.render})"
  | .unop op arg => s!"{op}({arg.render})"
  | .ternary c t e => s!"({c.render} ? {t.render} : {e.render})"
  | .index arr idx => s!"{arr}[{idx.render}]"

/-- Helper constructors for MetalExpr. -/
def MetalExpr.add (a b : MetalExpr) : MetalExpr := .binop "+" a b
def MetalExpr.sub (a b : MetalExpr) : MetalExpr := .binop "-" a b
def MetalExpr.mul (a b : MetalExpr) : MetalExpr := .binop "*" a b
def MetalExpr.div (a b : MetalExpr) : MetalExpr := .binop "/" a b
def MetalExpr.neg (e : MetalExpr) : MetalExpr := .unop "-" e
def MetalExpr.max (a b : MetalExpr) : MetalExpr := .call "max" [a, b]
def MetalExpr.min (a b : MetalExpr) : MetalExpr := .call "min" [a, b]
def MetalExpr.exp (e : MetalExpr) : MetalExpr := .call "exp" [e]
def MetalExpr.log (e : MetalExpr) : MetalExpr := .call "log" [e]
def MetalExpr.sqrt (e : MetalExpr) : MetalExpr := .call "sqrt" [e]
def MetalExpr.tanh (e : MetalExpr) : MetalExpr := .call "tanh" [e]
def MetalExpr.abs (e : MetalExpr) : MetalExpr := .call "abs" [e]
def MetalExpr.pow (base exp : MetalExpr) : MetalExpr := .call "pow" [base, exp]
def MetalExpr.rsqrt (e : MetalExpr) : MetalExpr := .call "rsqrt" [e]

/-- Try to parse a string as an integer. -/
private def parseIntLit (s : String) : Option Int :=
  -- Handle negative numbers
  if s.startsWith "-" then
    (s.drop 1).toNat?.map (Int.negOfNat ·)
  else
    s.toNat?.map Int.ofNat

/-- Try to parse a string as a float literal (simple heuristic). -/
private def parseFloatLit (s : String) : Option Float :=
  -- If it contains a decimal point, try to parse as float
  if s.contains '.' then
    -- Simple parsing: split on '.' and combine
    let parts := s.splitOn "."
    match parts with
    | [intPart, fracPart] =>
        match intPart.toNat?, fracPart.toNat? with
        | some i, some f =>
            let fracVal := (Float.ofNat f) / (10.0 ^ Float.ofNat fracPart.length)
            let sign := if s.startsWith "-" then -1.0 else 1.0
            some (sign * (Float.ofNat i + fracVal))
        | _, _ => none
    | _ => none
  else
    -- Maybe it's an int that should be a float
    parseIntLit s |>.map Float.ofInt

/-- Convert Jaxpr Atom to MetalExpr.
    For inputs, we index into the corresponding buffer.
    For intermediates, we use the variable directly. -/
def atomToExpr (a : Atom) (inputs : List String) : MetalExpr :=
  match a with
  | .var name _ =>
      if inputs.contains name then
        .index name (.var "id")  -- Buffer access: name[id]
      else
        .var name  -- Intermediate variable
  | .lit value _ =>
      -- Try to parse as float first, then int
      match parseFloatLit value with
      | some f => .lit f
      | none =>
        match parseIntLit value with
        | some i => .litInt i
        | none => .var value  -- Fallback: treat as variable name

/-- Registry of known Jaxpr primitives and their Metal translations. -/
def primToExpr (prim : String) (args : List MetalExpr) : Option MetalExpr :=
  match prim, args with
  -- Binary ops
  | "add", [a, b] => some (.add a b)
  | "sub", [a, b] => some (.sub a b)
  | "mul", [a, b] => some (.mul a b)
  | "div", [a, b] => some (.div a b)
  | "max", [a, b] => some (.max a b)
  | "min", [a, b] => some (.min a b)
  | "pow", [a, b] => some (.pow a b)
  -- Unary ops
  | "neg", [x] => some (.neg x)
  | "exp", [x] => some (.exp x)
  | "log", [x] => some (.log x)
  | "sqrt", [x] => some (.sqrt x)
  | "tanh", [x] => some (.tanh x)
  | "abs", [x] => some (.abs x)
  | "rsqrt", [x] => some (.rsqrt x)
  -- Activations
  | "relu", [x] => some (.max (.lit 0.0) x)
  | "sigmoid", [x] =>
      -- 1 / (1 + exp(-x))
      some (.div (.lit 1.0) (.add (.lit 1.0) (.exp (.neg x))))
  | "silu", [x] =>
      -- x / (1 + exp(-x)) = x * sigmoid(x)
      some (.div x (.add (.lit 1.0) (.exp (.neg x))))
  | "softplus", [x] =>
      -- log(1 + exp(x))
      some (.log (.add (.lit 1.0) (.exp x)))
  | "gelu", [x] =>
      -- 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
      let sqrt2pi := MetalExpr.lit 0.7978845608
      let c := MetalExpr.lit 0.044715
      let x3 := MetalExpr.mul x (MetalExpr.mul x x)
      some (MetalExpr.mul (MetalExpr.mul (.lit 0.5) x)
              (MetalExpr.add (.lit 1.0)
                (MetalExpr.tanh (MetalExpr.mul sqrt2pi (MetalExpr.add x (MetalExpr.mul c x3))))))
  | "leaky_relu", [x] =>
      -- max(0.01*x, x)
      some (.max (MetalExpr.mul (.lit 0.01) x) x)
  | "elu", [x] =>
      -- x if x > 0 else exp(x) - 1
      some (.ternary (.binop ">" x (.lit 0.0)) x (MetalExpr.sub (MetalExpr.exp x) (.lit 1.0)))
  | _, _ => none

/-- Convert a single Jaxpr equation to (varName, MetalExpr) step. -/
def eqnToStep (e : Eqn) (inputs : List String) : Except String (String × MetalExpr) := do
  -- Get output variable name
  let outName ← match e.out with
    | .var name _ => pure name
    | .lit _ _ => throw "equation output must be a variable"

  -- Convert argument atoms to MetalExprs
  let argExprs := e.args.map (atomToExpr · inputs)

  -- Look up the primitive
  match primToExpr e.prim argExprs with
  | some expr => return (outName, expr)
  | none => throw s!"unsupported primitive: {e.prim} with {e.args.length} args"

/-- Get the variable name from an Atom. -/
def atomName (a : Atom) : String :=
  match a with
  | .var name _ => name
  | .lit val _ => val

/-- Convert a Jaxpr to a Metal kernel string.
    The Jaxpr should contain only fusible elementwise operations. -/
def jaxprToMetal (kernelName : String) (j : Jaxpr) : Except String String := do
  -- Collect input buffer names
  let inputs := j.invars.map atomName

  -- Convert each equation to a step
  let mut steps : List (String × MetalExpr) := []
  for eqn in j.eqns do
    let step ← eqnToStep eqn inputs
    steps := steps ++ [step]

  -- Get output variable name
  let outExpr ← match j.outvars with
    | [a] => pure (MetalExpr.var (atomName a))
    | [] => throw "jaxpr has no outputs"
    | _ => throw "jaxpr has multiple outputs (not supported for single kernel)"

  -- Generate buffer declarations
  let mut bufferDecls := ""
  let mut idx : Nat := 0
  for input in inputs do
    bufferDecls := bufferDecls ++ s!"    device const float* {input} [[buffer({idx})]],\n"
    idx := idx + 1

  let outputIdx := inputs.length
  bufferDecls := bufferDecls ++ s!"    device float* output [[buffer({outputIdx})]],\n"
  bufferDecls := bufferDecls ++ "    uint id [[thread_position_in_grid]]"

  -- Generate body steps
  let mut bodySteps := ""
  for (varName, expr) in steps do
    bodySteps := bodySteps ++ s!"    float {varName} = {expr.render};\n"

  return s!"kernel void {kernelName}(
{bufferDecls}
) \{
{bodySteps}    output[id] = {outExpr.render};
}
"

/-- Generate a Metal shader file with header and kernel. -/
def jaxprToMetalFile (kernelName : String) (j : Jaxpr) : Except String String := do
  let kernel ← jaxprToMetal kernelName j
  return s!"#include <metal_stdlib>
using namespace metal;

// Auto-generated from Jaxpr - DO NOT EDIT

{kernel}"

/-- Batch convert multiple Jaxprs to a single Metal shader file. -/
def jaxprsToMetalFile (kernels : List (String × Jaxpr)) : Except String String := do
  let mut body := "#include <metal_stdlib>\nusing namespace metal;\n\n// Auto-generated from Jaxpr - DO NOT EDIT\n\n"
  for (name, jaxpr) in kernels do
    let kernel ← jaxprToMetal name jaxpr
    body := body ++ kernel ++ "\n"
  return body

/-- Check if a Jaxpr contains only fusible elementwise operations. -/
def isFusible (j : Jaxpr) : Bool :=
  j.eqns.all fun eqn =>
    match eqn.prim with
    | "add" | "sub" | "mul" | "div" | "max" | "min" | "pow"
    | "neg" | "exp" | "log" | "sqrt" | "tanh" | "abs" | "rsqrt"
    | "relu" | "sigmoid" | "silu" | "softplus" | "gelu" | "leaky_relu" | "elu" => true
    | _ => false

/-- Partition Jaxpr equations into fusible sequences.
    Non-fusible ops (like matmul) act as barriers that split sequences. -/
def partitionFusible (j : Jaxpr) : List (List Eqn) := Id.run do
  let mut groups : List (List Eqn) := []
  let mut current : List Eqn := []

  for eqn in j.eqns do
    let fusible := match eqn.prim with
      | "add" | "sub" | "mul" | "div" | "max" | "min" | "pow"
      | "neg" | "exp" | "log" | "sqrt" | "tanh" | "abs" | "rsqrt"
      | "relu" | "sigmoid" | "silu" | "softplus" | "gelu" | "leaky_relu" | "elu" => true
      | _ => false

    if fusible then
      current := current ++ [eqn]
    else
      -- Non-fusible op is a barrier
      if !current.isEmpty then
        groups := groups ++ [current]
        current := []
      -- The non-fusible op itself is a singleton group (marked for separate lowering)
      groups := groups ++ [[eqn]]

  -- Don't forget the last group
  if !current.isEmpty then
    groups := groups ++ [current]

  return groups

end SciLean.ML.Jaxpr
