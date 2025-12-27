# lean4-memories

Persistent learning across Lean 4 formalization sessions.

## Overview

This skill adds persistent memory to Claude's Lean 4 assistance, transforming stateless proof help into a learning system that remembers successful patterns, avoids known dead-ends, and adapts to project-specific conventions across sessions.

**Important:** This is an optional enhancement to [lean4-theorem-proving](../lean4-theorem-proving/). It requires MCP memory server configuration to function.

## What You Get

### Memory Types

1. **ProofPattern** - Successful proof strategies
   - Goal pattern description
   - Tactics sequence that worked
   - Helper lemmas used
   - Difficulty and confidence ratings

2. **FailedApproach** - Known dead-ends to avoid
   - Failed tactic or approach
   - Error type (infinite loop, timeout, type error)
   - Context and alternative that worked

3. **ProjectConvention** - Code style and patterns
   - Naming conventions
   - Proof structure preferences
   - Standard opening moves

4. **UserPreference** - Workflow customization
   - Verbose vs. concise output
   - Preferred tools and tactics
   - Session-specific settings

5. **TheoremDependency** - Theorem relationships
   - Helper lemmas for specific goals
   - Proof dependencies
   - Reusable components

### Memory Scoping

All memories are automatically scoped by:
- **Project path** - Absolute path to git root or cwd
- **Skill name** - Tagged with `lean4-memories`
- **Entity type** - ProofPattern, FailedApproach, etc.

This prevents memories from one Lean project affecting another.

### Example Workflow

**Session 1: First proof (30 minutes exploration)**
```
Proving: measure_eq_of_fin_marginals_eq
[After exploration]
✅ Success with π-system uniqueness approach

→ Store: ProofPattern "pi_system_uniqueness"
  - Works for: measure equality via finite marginals
  - Tactics: [isPiSystem, generateFrom_eq, measure_eq_on_piSystem]
  - Confidence: 0.9
```

**Session 2: Similar theorem (weeks later)**
```
Proving: fullyExchangeable_via_pathLaw
Goal: Show two measures equal

→ Retrieve: "Similar to measure_eq_of_fin_marginals_eq"
            "Try isPiSystem approach (confidence: 0.9)"

✅ Success in 5 minutes using remembered pattern
```

**Session 3: Avoiding failure**
```
About to: simp only [condExp_indicator, mul_comm]

→ Retrieve: ⚠️ "This causes infinite loop (stored Session 1)"
            "Alternative: simp only [condExp_indicator], then ring"

✅ Avoid 20-minute debugging session
```

## Setup

### Requirements

- **MCP memory server** configured
- **Claude Desktop** or Claude Code with MCP support
- **lean4-theorem-proving skill** (core dependency)

### Installation

**Step 1: Enable MCP memory server**

**Option A: Claude Code (CLI - simplest)**

```bash
claude mcp add --transport stdio --scope user memory -- npx -y @modelcontextprotocol/server-memory
```

This automatically configures the memory server for all your projects.

**Option B: Claude Desktop or Claude Code (Config file)**

Open your config file in a text editor:

**macOS:**
```bash
open ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**Linux:**
```bash
nano ~/.config/Claude/claude_desktop_config.json
```

**Windows:**
```powershell
notepad %APPDATA%\Claude\claude_desktop_config.json
```

Add this configuration (or add the `memory` entry if you already have other MCP servers):

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}
```

Save the file and **restart Claude Desktop/Code**.

**Step 2: Install the skill**

See [main README](../README.md#installation) for skill installation instructions.

**Step 3: Verify it's working**

The skill operates automatically - no configuration needed. When you work on Lean 4 projects, memories will be stored and retrieved transparently. You can verify memory operations using the CLI helper:

```bash
cd lean4-memories
./scripts/memory_helper.py list
```

## Usage

### Automatic Operation

The skill operates automatically:

**Before starting a proof:**
- Queries for similar goal patterns
- Surfaces successful tactics for this pattern
- Checks for known issues with current context

**During proof development:**
- Before each major tactic, checks for known failures
- When stuck, retrieves alternative approaches
- Suggests next tactics based on past success

**After completing a proof:**
- Stores successful patterns (if non-trivial)
- Records helper lemmas used
- Updates confidence scores

### Memory Helper CLI

A helper script is provided for manual memory operations:

```bash
# Store a successful proof pattern
./scripts/memory_helper.py store-pattern \
  --goal "measure equality via finite marginals" \
  --tactics "isPiSystem,generateFrom_eq,measure_eq_on_piSystem" \
  --confidence 0.9

# Find similar patterns
./scripts/memory_helper.py find-patterns \
  --query "conditional expectation"

# Store a failed approach
./scripts/memory_helper.py store-failure \
  --tactic "simp only [condExp_indicator, mul_comm]" \
  --error "infinite loop"

# List all memories for current project
./scripts/memory_helper.py list
```

## When to Use

**Perfect for:**
- Multi-session projects spanning days/weeks/months
- Repeated proof patterns requiring similar approaches
- Complex proofs with multiple attempted approaches
- Team projects sharing knowledge across developers
- Learning workflows building domain expertise

**Especially helpful when:**
- You've encountered similar proofs before
- You want to avoid repeating failed approaches
- You're building project-specific conventions
- You want Claude to learn from your proof style

## Memory Quality Guidelines

### DO Store

- ✅ Successful non-trivial proofs (>10 lines)
- ✅ Failed approaches that wasted significant time
- ✅ Consistent patterns observed multiple times
- ✅ Project-specific insights

### DON'T Store

- ❌ Trivial proofs (rfl, simp, exact)
- ❌ One-off tactics unlikely to recur
- ❌ General Lean knowledge (already in training/mathlib)
- ❌ Temporary workarounds

### Confidence Scoring

- **High (0.8-1.0)** - Clean proof, no warnings, well-tested
- **Medium (0.5-0.8)** - Works but has minor issues
- **Low (0.0-0.5)** - Hacky solution, needs refinement

## Integration with lean4-theorem-proving

This skill complements (doesn't replace) lean4-theorem-proving:

**lean4-theorem-proving provides:**
- General Lean 4 workflows (4-Phase approach)
- mathlib search and tactics reference
- 16 automation scripts
- Domain-specific knowledge

**lean4-memories adds:**
- Project-specific learned patterns
- History of what worked/failed in this project
- Accumulated domain expertise from your proofs
- Personalized workflow preferences

**Use together:**
1. lean4-theorem-proving guides general workflow
2. lean4-memories provides project-specific context
3. Memories inform tactic choices from lean4-theorem-proving

## Contents

```
lean4-memories/
├── README.md                      # This file
├── SKILL.md                       # Core memory guide (loaded by Claude)
├── scripts/
│   └── memory_helper.py           # CLI helper for memory operations
└── references/
    └── memory-patterns.md         # Detailed memory operation examples
```

## Privacy and Scoping

**Local storage:** All memories are stored locally in your Claude Desktop instance via the MCP memory server.

**Project isolation:** Memories are scoped by absolute project path, preventing cross-project contamination.

**No sharing by default:** Memories are not shared across Claude instances. See "Future Enhancements" in [SKILL.md](SKILL.md) for planned export/import functionality.

## FAQ

### Does this work without MCP memory server?

No. This skill requires MCP memory server to function. Without it, the skill will not work.

If you don't have MCP memory server configured, use only [lean4-theorem-proving](../lean4-theorem-proving/) which works standalone.

### How do I know if it's working?

The skill operates transparently. You'll notice:
- Faster proofs for similar patterns
- Warnings about problematic tactics
- Suggestions based on past work

### Can I see what's stored?

Use the memory helper CLI:
```bash
./scripts/memory_helper.py list
```

### How do I clear memories?

Memory management is handled by the MCP memory server. Consult MCP memory server documentation for clearing operations.

### Will this slow down Claude?

No. Memory queries are fast and run in parallel with Claude's reasoning. The performance impact is negligible.

### Can I share memories with my team?

Currently, memories are local to your Claude Desktop instance. Future enhancements may include export/import functionality. See [SKILL.md](SKILL.md) for planned features.

## Known Limitations

- Stale memories if project evolves significantly
- Over-fitting to specific project patterns possible
- Requires manual memory hygiene for large projects
- No cross-project pattern detection (intentional for privacy)

See [SKILL.md](SKILL.md#limitations-and-caveats) for detailed discussion.

## Contributing

Contributions welcome! See [main README](../README.md#contributing) for guidelines.

**Especially interested in:**
- Memory pattern examples that worked well
- Best practices for memory hygiene
- Integration patterns with lean4-theorem-proving

## License

MIT License - see [../LICENSE](../LICENSE)

## Related Resources

**MCP (Model Context Protocol):**
- [MCP Documentation](https://modelcontextprotocol.io/)
- [MCP Memory Server](https://github.com/modelcontextprotocol/servers/tree/main/src/memory)

**Claude Skills:**
- [Claude Skills Documentation](https://www.anthropic.com/news/skills)
- [Main Repository](../README.md)
- [lean4-theorem-proving](../lean4-theorem-proving/)

---

Part of [lean4-skills](../README.md) - Lean 4 Skills for Claude
