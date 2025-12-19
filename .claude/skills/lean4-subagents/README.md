# Lean 4 Specialized Subagents (EXPERIMENTAL)

**STATUS: EXPERIMENTAL** - These agents are under development. Prefer using the `/lean4-theorem-proving:*` slash commands for interactive workflows.

Autonomous subagents for batch Lean 4 proof development tasks.

## Overview

This plugin provides three **EXPERIMENTAL** specialized subagents designed to work autonomously on specific Lean 4 development workflows. Each subagent references the lean4-theorem-proving skill for complete workflows and patterns.

## Subagents

### 1. lean4-proof-golfer (EXPERIMENTAL)

**Purpose:** Optimize Lean 4 proofs by shortening length or runtime while maintaining readability

**When to use:** After proofs compile successfully to achieve 30-40% size reduction

**How it works:**
- Reads and follows `lean4-theorem-proving/references/proof-golfing.md`
- Looks for bundled scripts in `~/.claude/skills/lean4-theorem-proving/scripts/`
- Falls back to manual patterns if scripts not accessible
- 4-phase workflow (Find → Verify Safety → Apply → Report)

**Prefer:** Use `/lean4-theorem-proving:golf-proofs` command for interactive guidance

**Example dispatch:**
```
Dispatch lean4-proof-golfer subagent to optimize all proofs in ViaL2.lean
```

---

### 2. lean4-sorry-filler (EXPERIMENTAL)

**Purpose:** Fill incomplete proofs (sorries) using mathlib search and multi-candidate testing

**When to use:** When tackling incomplete proofs systematically

**How it works:**
- Reads and follows `lean4-theorem-proving` skill workflows
- Looks for bundled scripts in `~/.claude/skills/lean4-theorem-proving/scripts/`
- Falls back to WebFetch (leansearch, loogle) if scripts not accessible
- 6-phase workflow (Understand → Search → Generate → Test → Apply → Report)

**Prefer:** Use `/lean4-theorem-proving:analyze-sorries` command for interactive guidance

**Example dispatch:**
```
Dispatch lean4-sorry-filler subagent to fill all sorries in Probability/CondExp.lean
```

---

### 3. lean4-axiom-eliminator (EXPERIMENTAL)

**Purpose:** Systematically eliminate axioms and sorries from Lean 4 proofs

**When to use:** After checking axiom hygiene to reduce axiom count to zero

**How it works:**
- Reads and follows `lean4-theorem-proving/commands/check-axioms.md` workflow
- Looks for bundled scripts in `~/.claude/skills/lean4-theorem-proving/scripts/`
- Falls back to `#print axioms` and manual checking if scripts not accessible
- 6-phase workflow (Audit → Plan → Search → Eliminate → Track → Verify)

**Prefer:** Use `/lean4-theorem-proving:check-axioms` command for interactive guidance

**Example dispatch:**
```
Dispatch lean4-axiom-eliminator subagent to eliminate all custom axioms in src/
```

---

## Interactive vs Autonomous Modes

**Interactive mode (slash commands):**
- Use for 1-2 proofs/sorries/axioms
- Guided workflow with user decisions
- Real-time feedback and iteration

**Autonomous mode (subagents):**
- Use for batches of 10+ items
- Fully autonomous operation
- Background execution while you work on other tasks
- Final report with statistics

---

## Usage Pattern

**Step 1:** Assess scope
```
/analyze-sorries    # See all sorries in project
/check-axioms       # See all axioms in project
```

**Step 2:** Choose mode
- **1-3 items?** Use slash commands (interactive)
- **10+ items?** Dispatch subagent (autonomous)

**Step 3:** Review results
- Subagents provide detailed reports
- All changes tested (compilation verified)
- Failed attempts documented with strategies

---

## Requirements

- **lean4-theorem-proving** plugin (provides core skill and scripts)
- **Optional:** `lean-lsp-mcp` server (enables `lean_multi_attempt` for parallel testing)

---

## Version History

**1.0.0** (2025-10-26)
- Initial release with 3 specialized subagents
- Integrated with lean4-theorem-proving slash commands
- Built-in false-positive filtering and safety checks

---

## See Also

- [lean4-theorem-proving](../lean4-theorem-proving/README.md) - Core skill and slash commands
- [Proof-golfing reference](../lean4-theorem-proving/references/proof-golfing.md) - Optimization patterns
- [Subagent workflows](../lean4-theorem-proving/references/subagent-workflows.md) - Delegation patterns
