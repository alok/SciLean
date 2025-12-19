---
name: lean4-elab-delab
description: Lean 4 elaborators + delaborators/unexpanders. Use for DSLs that need type info or custom pretty output.
---

# Lean 4 Elab + Delab Skill

## When to use

- You need type info, environment access, or custom validation while parsing a term.
- You want goals/diagnostics to print in DSL notation instead of raw constructors.
- You want to surface metadata (device tags, fusion info, cost, etc.) in pretty output.

## Decision tree

| Need | Use | Notes |
|---|---|---|
| Purely syntactic rewrite | `macro_rules` | Fast, hygienic, no type info |
| Type-driven parse or custom errors | `elab_rules` | Use `elabTerm` + `throwErrorAt` |
| Custom pretty output for a function | `@[app_unexpander]` or `@[app_delab]` | Prefer unexpander if simple |
| Custom pretty output for metadata | `@[delab mdata.myKey]` | Best for device/trace tags |

## Elab template

```lean
syntax (name := tg_view) "view[" term "]" : term

elab_rules : term
  | `(view[$t]) => do
      let e ← elabTerm t none
      -- Validate or rewrite here
      -- Use `withRef` or `throwErrorAt` for good error spans
      return e
```

## Unexpander template (simple apps)

```lean
/-- Print `view` applications as `view[...]`. -/
@[app_unexpander TinyGrad4.view]
def unexpandView : Lean.PrettyPrinter.Unexpander
  | `($_ $t) => `(view[$t])
  | _ => throw ()
```

Notes:
- Unexpanders only fire for simple applications where the term is a single app head.
- If matching is nontrivial (implicit args, dependent args), use a delaborator.

## Delaborator template (apps)

```lean
open Lean PrettyPrinter Delaborator

@[app_delab TinyGrad4.view]
def delabView : Delab := do
  -- `getExpr` is the full application Expr.
  -- This assumes one explicit argument.
  let stx ← withNaryArg 0 delab
  return ← `(view[$stx])
```

## Delaborator template (mdata tags)

```lean
open Lean PrettyPrinter Delaborator

/-- Show device tags like `@gpu0` inline. -/
@[delab mdata.tg.device]
def delabDeviceTag : Delab := do
  let e ← getExpr
  -- `mdata` has the tag; the payload is the inner expression.
  let Expr.mdata _ _ := e | failure
  let stx ← withMDataExpr delab
  -- Render as `(@gpu0 ...)` or whatever notation you want
  -- (Use a custom syntax category if needed.)
  return stx
```

## Pretty-print options

- Respect `pp.notation` and `pp.explicit`.
- Use `whenPPOption` to toggle verbose output for debugging.
- Provide `set_option tinygrad.pp.device true` style toggles.

## Hygiene + errors

- Use `Macro.throwErrorAt` / `throwErrorAt` to attach errors to the right span.
- Prefer `stx.reprint` for user-facing messages; `getString` is lossy.

## Debug checklist

- If output does not change, check `pp.notation`, missing imports, or wrong head constant.
- For `@[app_delab]`, make sure the name is resolved in scope.
- For `@[delab mdata.key]`, ensure the metadata contains a single key.

## TinyGrad4-specific guidance

- Keep all output customizations in `TinyGrad4/Pretty.lean`.
- Annotate IR nodes with metadata for device, fusion plan, and cost.
- Use unexpanders for surface syntax and delabs for metadata.
