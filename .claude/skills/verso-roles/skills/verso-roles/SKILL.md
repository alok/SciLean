---
name: verso-roles
description: Use when writing or fixing Lean doc comments, especially to resolve Verso role warnings about unresolved identifiers or roles.
---

# Verso Roles (Lean Doc Comments)

Use roles in `/-- ... -/` doc comments so the Verso linter can resolve identifiers and reduce warnings.

## Role Priority

1) `{name}` for declared names (constants, structures, namespaces, theorems)
2) `{lean}` for Lean expressions or code fragments
3) `{lit}` only if the text should *not* be resolved (last resort)

## Quick Rules

- Wrap inline code in backticks and add a role:
  - `{name}``MyConstant`
  - `{lean}``fun x => x + 1``
  - `{lit}``x + 1`` (only when resolution must be disabled)
- Prefer `{name}` over `{lean}` when referencing a declared identifier.
- Use `{given}` to declare variables in scope, then reference them with `{lean}`:
  - `{given}``n`` then `{lean}``n``
- Wrap math in backticks to avoid parser issues.
- If a symbol is *not* in scope (e.g. `{lit}``toByteArray``), use `{lit}`.

## Examples

- `Returns {name}``TensorLayout.transpose`` result.`
- `For {given}``n``, the shape is {lean}``#[n]``.`
- `Use {lit}``x[i,j]`` if it should stay literal.`

## Fixing Warnings

When a warning says a code element is not specific, replace:

- `` `foo` `` → `{name}``foo`` if `foo` is a declared name
- `` `foo x` `` → `{lean}``foo x`` if it is an expression
- Otherwise use `{lit}` as a last resort
