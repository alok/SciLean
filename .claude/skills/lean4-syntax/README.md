# Lean 4 Custom Syntax Skill

Focused reference for Lean 4 syntax extensions: DSLs, macros, elaborators.

## Contents

- `SKILL.md` - Decision tree, API cheat sheet, gotchas, patterns (~160 lines)
- `references/reference.md` - Elaborator monads, hygiene, unexpanders (~100 lines)
- `commands/scaffold-dsl.md` - DSL template

## Commands

| Command | Description |
|---------|-------------|
| `/lean4-syntax:scaffold-dsl` | Generate DSL boilerplate |

## Quick Reference

```lean
-- Macro API
Macro.addMacroScope `name    -- fresh hygienic name
stx.reprint                  -- original user text
xs.getElems                  -- array from $xs,*

-- Precedence
80: * /    65: + -    50: < > =    35: ∧    30: ∨    25: →
Left-assoc: term:66 on right
```

## License

MIT
