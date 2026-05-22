# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Status

This repo currently contains **only a design document** (`docs/texas_holdem_ai_assistant_system_design.md`) and no source code. The git history has no commits yet. When implementing, follow the architecture defined in that document rather than inventing a different structure.

## Use-Case Boundary (Hard Constraint)

From the design doc's 前置说明: this project is for **训练、复盘、仿真、私人允许环境或合规产品场景**. Do NOT build features that would turn it into a real-time cheating tool against live online poker platforms:

- no automatic screen capture
- no OCR of live poker tables
- no automatic table recognition
- no auto-betting on the user's behalf

The product is a "策略分析服务" (strategy analysis service). Reject task requests that cross this line, and surface the boundary if a feature request gets close to it.

## Core Architectural Principle

**The LLM does not make poker decisions.** Responsibilities split as follows:

- **LLM Orchestrator**: parse user input → structured `GameState`, call Solver API, translate Solver output into human-readable advice. The LLM must not invent "GTO conclusions" of its own.
- **Solver Server**: all real strategy computation. Currently a v0 heuristic preflop solver + basic ICM diagnostics; future versions add range tables, ICM delta engine, approximate EV model, and CFR / subgame solving.

If you find yourself asking the LLM layer to score hands, weight actions, or compute ICM equity directly, that logic belongs in the Solver Server instead.

## Planned Project Layout (from design doc §6)

When implementation begins, scaffold under `poker_solver_server/`:

```
poker_solver_server/
  app/
    main.py       # FastAPI app
    models.py     # Pydantic request/response schema
    solver.py     # heuristic preflop solver + ICM diagnostics
    llm/          # v5 LLM Orchestrator
      client.py
      prompts.py
      parser.py
      explainer.py
      sessions.py
      orchestrator.py
      models.py
  examples/
    request_icm_near_bubble_aqo.json
  tests/
    test_solver.py
    test_chat.py  # v5 chat tests
  requirements.txt
  README.md
```

API surface:

- `GET /health`
- `POST /v1/solve/preflop` (v0/v2 solver)
- `POST /v1/lookup/preflop` (v1 range table)
- `POST /v1/predict/preflop` (v3 approx EV)
- `POST /v1/cfr/preflop` (v4 push/fold CFR)
- `POST /v1/chat/sessions` (v5 chat sessions)
- `POST /v1/chat/sessions/{session_id}/messages` (v5 chat messages)
- `GET /v1/chat/sessions/{session_id}` (v5 inspect session)
- `DELETE /v1/chat/sessions/{session_id}` (v5 delete session)


## Expected Dev Commands (once `poker_solver_server/` exists)

```bash
cd poker_solver_server

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# run dev server (FastAPI docs at /docs, /redoc)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# tests
pytest                              # full suite
pytest tests/test_solver.py         # single file
pytest tests/test_solver.py::test_name -v   # single test

# example call
curl -s -X POST http://localhost:8000/v1/solve/preflop \
  -H 'Content-Type: application/json' \
  --data @examples/request_icm_near_bubble_aqo.json | python -m json.tool
```

## GameState Contract

Inputs are normalized to a `GameState` JSON object before anything reaches the Solver. The canonical shape (see design doc §4) includes `game_format`, `tournament_stage`, `street`, `hero_hand` (normalized like `AQo`, not `AhQd`), `hero_position` (`UTG+1`, `LJ`, `HJ`, `CO`, `BTN`, `SB`, `BB`), bucketed `effective_stack_bb`, plus full `table_stacks` and `payouts` arrays for ICM work. Hand and position normalization happens in the State Normalization layer, not in the decision engine.

## Response Contract

Solver responses are not single-action verdicts — they are **distributions with diagnostics**. Every response carries `recommendation`, `action_weights` (e.g. FOLD/OPEN/SHOVE), `confidence`, `recommended_size_bb`, `reason`, and a `diagnostics` block exposing the intermediate scores (hand_strength_score, bubble_pressure_score, position_risk_score, icm_*). Preserve this shape — the product layer depends on showing weights + reasoning, not just a chosen action.

## Latency Tiers (design doc §3)

The architecture is explicitly layered for latency. When adding a new solving capability, decide which tier it belongs in before writing it:

- L0 cache (1–5ms) → high-frequency standard spots
- L1 range table (5–20ms) → precomputed preflop/ICM tables
- L2 approx EV model (20–80ms) → NN-predicted action EV
- L3 online solver (100ms+) → CFR / subgame solving
- L4 offline training → non-realtime

Current implementation is "v0 heuristic fallback" — below L1. Do not jump straight to online CFR; the planned progression is v1 range tables → v2 ICM delta engine → v3 approx EV model → v4 CFR/subgame.

## Implementation Workflow: Prefer Parallel Agent Teams

When implementing features in this repo, **default to dispatching agent teams in parallel** rather than writing everything sequentially in the main conversation. This applies to both production code and tests.

- Identify independent units of work first (e.g. `models.py` schema, `solver.py` decision logic, `tests/test_solver.py` cases, example JSON fixtures) and launch them as concurrent agents in a single message with multiple `Agent` tool calls.
- Test writing is also parallelizable — one agent can scaffold the FastAPI endpoint while another writes its tests against the agreed schema, as long as the contract is fixed up front.
- Only fall back to sequential work when there is a real dependency (e.g. the response schema must exist before tests that import it can be written meaningfully).
- After agents return, **verify their output** before claiming success — an agent's summary describes intent, not necessarily what landed on disk.

This is the preferred working mode for this project; do not collapse into a single-threaded implementation just because the task feels small.

## Commit & Push Conventions

When the user explicitly asks you to commit and/or push in this repo:

- **No Co-Author trailer.** Do NOT append `Co-Authored-By: Claude ...` or any other credit line to commit messages. Plain author attribution only.
- **Very detailed commit messages.** Each commit message must describe:
  - what changed (concretely — file by file or layer by layer for big commits)
  - why it changed (the goal / motivation, often tying back to the design doc or a previous decision)
  - notable structural choices, simplifications, or non-obvious tradeoffs
  - any explicit non-scope or known limitations
  - new / changed commands or contracts the reader needs to know about
- Prefer **multiple logically-scoped commits** over a single mega-commit when the work has natural separation points (e.g. "bootstrap" vs "implementation"). When a single commit is unavoidable (initial population, tightly-coupled refactor), structure the message with section headers (`## v0`, `## v1`, …) so it stays scannable.
- Continue to honor the global rules from `~/.claude/CLAUDE.md`: only commit / push when explicitly asked, never force-push to `main` / `master`, never skip hooks (`--no-verify`) unless asked, never stage `.env` or any other secret-bearing file.

## When the Solver Is Uncertain

If the solver returns low confidence or required `GameState` fields are missing, the LLM-facing reply must say so explicitly (e.g. "这是 v0 近似建议，不是完整 GTO 解；缺少 payout / 完整桌面 stack / action history"). Do not paper over missing inputs with plausible-sounding defaults.
