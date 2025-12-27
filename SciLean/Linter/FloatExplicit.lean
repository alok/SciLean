/-
Copyright (c) 2024 SciLean contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: SciLean contributors
-/
import Lean

/-!
# Float64 Explicit Linter

This linter warns when `Float` is used instead of `Float64`.

**Why?** Lean's `Float` type IS `Float64` under the hood, but the implicit naming
causes confusion in mixed-precision numeric code (Float32/Float16/Float64).

**Fix:** Replace `Float` with `Float64` to make the 64-bit precision explicit.

## Usage

Enable the linter with:
```
set_option linter.floatExplicit true
```

Disable it locally with:
```
set_option linter.floatExplicit false in
def myFunc (x : Float) := x  -- no warning
```

Enable globally in lakefile.lean:
```
leanOptions := #[⟨`weak.linter.floatExplicit, true⟩]
```
-/

namespace SciLean.Linter

open Lean Elab Command Meta

/-- Option to control the Float → Float64 linter -/
register_option linter.floatExplicit : Bool := {
  defValue := true
  descr := "warn when `Float` is used instead of explicit `Float64`"
}

/-- Check if the linter is enabled -/
def floatExplicitLinterEnabled : CommandElabM Bool := do
  return linter.floatExplicit.get (← getOptions)

/-- Find all occurrences of `Float` identifier in syntax -/
partial def findFloatIdents (stx : Syntax) : Array Syntax := Id.run do
  let mut result := #[]
  match stx with
  | .node _ _ args =>
    for arg in args do
      result := result ++ findFloatIdents arg
  | .ident _ rawVal _ _ =>
    -- Check if this is the `Float` identifier (not `Float64`, `Float32`, etc.)
    -- We check the raw string to catch exactly "Float" as typed
    let rawStr := rawVal.toString
    if rawStr == "Float" then
      result := result.push stx
  | .atom _ val =>
    if val == "Float" then
      result := result.push stx
  | _ => pure ()
  return result

/-- Generate a helpful error message for AI agents and developers -/
def floatWarningMessage : MessageData :=
  m!"Use `Float64` instead of `Float` for clarity.\n\n" ++
  m!"**Why?** `Float` is an alias for `Float64`, but the implicit naming causes confusion " ++
  m!"in mixed-precision code (Float32/Float16/Float64).\n\n" ++
  m!"**Fix:** Replace `Float` with `Float64` to make the 64-bit precision explicit.\n\n" ++
  m!"**Example:**\n" ++
  m!"  `def f (x : Float) : Float := x * 2.0`\n" ++
  m!"  `def f (x : Float64) : Float64 := x * 2.0`\n\n" ++
  m!"To disable this lint: `set_option linter.floatExplicit false`"

/-- The Float linter run function -/
def floatExplicitLinterRun (stx : Syntax) : CommandElabM Unit := do
  unless ← floatExplicitLinterEnabled do return
  let floatIdents := findFloatIdents stx
  for ident in floatIdents do
    logWarningAt ident floatWarningMessage

/-- The Float linter implementation -/
def floatExplicitLinter : Linter := {
  run := floatExplicitLinterRun
  name := `SciLean.Linter.floatExplicit
}

/-- Register the linter -/
initialize addLinter floatExplicitLinter

end SciLean.Linter
