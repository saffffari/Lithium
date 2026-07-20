# Runbook — Multi-Agent Codebase Audit

End-to-end recipe for re-running the cartography + wave-1 sweep + wave-2 verification + wave-3 synthesis pipeline that produced [_audit/REPORT.md](../../_audit/REPORT.md).

## When to use

- Before a large refactor PR that touches load-bearing files
- After significant codebase growth (you've "lost track")
- Quarterly health check
- Onboarding a new contributor who needs an honest map of where the bodies are buried

## Prerequisites

- Claude Code on a plan with token headroom (Max recommended — full pipeline burns ~3–5M tokens)
- Wall-clock budget: 90–180 minutes
- `_audit/` directory present and gitignored (see `.gitignore`)
- `.claude/agents/cartographer.md` present
- Latest `main` or feature branch checked out — pipeline reflects committed state

## Pipeline phases

```
Phase 0 — Setup           (<5 min, no agents)
Phase 1 — Cartography     (~30-60 min wall, 1 opus agent)
Phase 2 — Wave 1 sweep    (~10-20 min wall, 11 parallel agents — 5 opus, 6 sonnet)
Phase 3 — Wave 2 verify   (~10-15 min wall, 11 parallel agents)
Phase 4 — Wave 3 synth    (~5-10 min wall, 1 opus agent → REPORT.md)
```

Phases run sequentially; agents within a phase run in parallel.

## Phase 1 — Cartography

Invoke the `cartographer` subagent (or spawn `general-purpose` with `.claude/agents/cartographer.md` content inlined). Produces:

- `_audit/ARCHITECTURE.md` — module table + mermaid dep graph
- `_audit/FLOWS.md` — 6-8 traced user actions
- `_audit/SPINE.md` — the 5-10 load-bearing files
- `_audit/ORIENTATION.md` — one-page "returning developer" doc
- `_audit/_inventory/{imports,entrypoints,shaders}.json` — extracted structure

**Why one agent, not parallel:** comprehension is cross-cutting; parallel agents derive inconsistent mental models.

## Phase 2 — Wave 1 (saturation sweep)

11 parallel agents, one per concern. Each writes JSON findings to `_audit/wave1/<concern>.json`.

| Concern | Model | Focus |
|---|---|---|
| label-integrity | opus | TIER-1 silent loss vectors per `project_label_safety_priority` |
| state-coupling | opus | LibraryCatalog / cloud_store / model_registry / measure_registry ownership |
| analytics-architecture | opus | MEASUREMENT_CATALOG ↔ implementation drift |
| training-wiring | opus | PT-v3 launch gotchas per `project_ptv3_training_realities` |
| arch-drift | opus | layer violations, cycles, god modules, extraction targets |
| threading-safety | sonnet | subprocess reader → main-thread mutation |
| imaging-pipeline | sonnet | runner template adherence, surface-extraction constants |
| gpu-lifecycle | sonnet | ModernGL buffer/texture/program leaks |
| shader-uniform-drift | sonnet | GLSL declarations vs Python uniform writes |
| gui-token-drift | sonnet | tokens.py vs theme.py migration |
| dead-code | sonnet | orphaned modules, unused symbols, legacy successors |

Each prompt MUST include:

1. Required reading: `_audit/ORIENTATION.md`, `_audit/SPINE.md`, `_audit/ARCHITECTURE.md`, `CLAUDE.md`, relevant project memory files
2. Concern-specific scope (files, patterns, grep targets)
3. JSON-only output schema: `{file, line_range, category, severity, claim, code_excerpt, repro_or_argument, suggested_fix}`
4. Anti-patterns to reject: style/naming/missing tests/speculation/findings without code_excerpt
5. File budget cap (e.g. 30-40 files) to enforce depth over breadth
6. Research-mode framing — skip security/audit/regulatory concerns

Skip security as a concern unless the project changes posture — research-mode software per `feedback_research_mode_not_clinical`.

## Phase 3 — Wave 2 (adversarial verification)

11 parallel verification agents, one per concern. Each reads its wave-1 JSON and argues against every finding. Writes `_audit/wave2/<concern>-verified.json`.

Verdicts per finding:

- **VERIFIED** — real defect, severity reasonable
- **DOWNGRADED** — real but severity should be lower (state new severity + reason)
- **UPGRADED** — real, severity should be higher (rare)
- **REJECTED** — not a real defect (cite contextual reason)

Per-finding context size is small (one finding + the cited file), so this phase is cheap relative to its value. Expect 10-30% downgrades or rejections; tier-1 concerns (label-integrity, state-coupling) typically verify near-100%.

## Phase 4 — Wave 3 (adversarial synthesis)

Single opus agent reads all 11 verified JSONs + cartography artifacts. Writes `_audit/REPORT.md` with:

- Executive summary (5-10 must-fix items, ordered by user-impact × confidence)
- High-confidence findings grouped by theme
- Worth-checking findings (sev 3 single-verifier)
- Systemic patterns (bug-shapes that repeat across concerns)
- Suggested fix order / batches (coordinated PR groupings)
- Confirmed healthy (negative findings worth highlighting)
- Latent / noise (one-line each)

Synthesis MUST dedup across concerns — the same defect often appears under threading-safety AND state-coupling AND label-integrity.

## Outputs to commit decisions on

| Artifact | Default action |
|---|---|
| `_audit/ARCHITECTURE.md`, `FLOWS.md`, `SPINE.md`, `ORIENTATION.md` | Keep (gitignored; regenerable) |
| `_audit/_inventory/*.json` | Keep |
| `_audit/wave1/*.json`, `wave2/*-verified.json` | Keep (cheap, useful for diff against next audit) |
| `_audit/REPORT.md` | Reference from PRs and roadmap planning |

## Escalation

- **Any sev-5 finding in label-integrity or state-coupling** → TIER-1 per `project_label_safety_priority`. Fix in the next PR.
- **New systemic pattern** (≥3 instances) → architectural conversation; don't fix piecemeal.
- **Architecture extraction targets** → batch into a single refactor PR per target (don't chip away).

## Rollback

N/A — entire pipeline is read-only on source code. Only `_audit/` is written. To restart, delete `_audit/` and re-run.

## Related

- `.claude/agents/cartographer.md` — the cartographer subagent definition
- `_audit/REPORT.md` — the most recent audit output
- `CLAUDE.md` — project conventions referenced by every agent
- `AGENTS.md` — measurement-catalog discipline + frame conventions
- Subagent docs: [code.claude.com/docs/en/sub-agents](https://code.claude.com/docs/en/sub-agents)

## Notes from the first run (2026-05-20)

- Cartography found ~15% of the codebase had drifted from prior memory snapshots (PolyPose status especially). Cartography output supersedes memory when they disagree.
- ~131 raw wave-1 findings → ~94 verified after wave 2. False positive rate ~28% — dominated by gui-token-drift over-rating cosmetic issues as sev-5.
- Highest-yield single concern: **analytics-architecture** — surfaced the "MEASUREMENT_CATALOG is fictional" architectural lie that AGENTS.md had been promising as a feature. Worth the entire opus budget on its own.
- Lowest-yield single concern: gui-token-drift (28 findings, all sev-1/2 after verification). Consider sampling rather than full sweep next time.
