# v5: LLM Orchestrator (Natural Language Frontend)

**Status:** Completed 2026-05-22

---

## Context

v5 完成设计文档 §1 + §2.1 的最后一层 — LLM Orchestrator。v0–v4 已交付 4 个 solver endpoints，但用户必须自己手搓 GameState JSON。v5 加一层自然语言前端：

```
用户 NL → LLM parser → GameState → solve_preflop (v0+v2) → LLM explainer → NL 回答
```

设计文档 §2.1 明文："LLM 不直接做牌局决策"。v5 严格遵守 — LLM 只做 parse + explain，**不**做 routing、**不**用 tool-calling、**不**自己算 EV。

---

## Locked Decisions

| # | 决策 | 选择 | 理由 |
|---|---|---|---|
| 1 | LLM 职责 | parser + explainer only | 最小职责面，错误窄、可测、可解释 |
| 2 | Solver 路由 | 硬编码 → `solve_preflop` (v0+v2) | 信息量最大；含 v0 weights + v2 ICM EVs |
| 3 | Provider | OpenRouter | 复用 `.env` 现有 key，model 可换 |
| 4 | 默认 model | `google/gemini-flash-3.5`（slug 实现时去 OpenRouter 目录确认）；env 可换 | Flash 对 bounded parse+explain 性价比最佳 |
| 5 | 会话形态 | 多轮 session-based | "what if 10BB instead" 这种 follow-up 是核心 UX |
| 6 | Streaming | **不**开启（留 v5.1） | 简单、可测、metadata 可一次性返 |

---

## Endpoints

```
POST   /v1/chat/sessions                           create new session
POST   /v1/chat/sessions/{session_id}/messages     send a message
GET    /v1/chat/sessions/{session_id}              inspect session (debug)
DELETE /v1/chat/sessions/{session_id}              end session
```

### Request / Response

`POST /v1/chat/sessions`
- Req: `{}`
- Res: `{session_id: str, created_at: ISO8601}`

`POST /v1/chat/sessions/{session_id}/messages`
- Req: `{message: str}`
- Res:
  ```jsonc
  {
    "message_id": "uuid",
    "answer": "建议 MIXED：在 14BB UTG+1 ...",  // NL from LLM
    "parsed_state": { /* GameState | null */ },
    "solver_data":  { /* SolveResponse | null — null if missing fields */ },
    "missing_fields": ["hero_stack_bb"],         // [] if complete
    "usage": {"prompt_tokens": 123, "completion_tokens": 45, "model": "..."}
  }
  ```

---

## File Layout

新增（全部在 `poker_solver_server/`）：

```
app/llm/
  __init__.py
  client.py         OpenRouter HTTP wrapper (httpx)
  prompts.py        system prompts + few-shot examples
  parser.py         NL → GameState  (LLM JSON mode + Pydantic validation)
  explainer.py      SolveResponse → NL  (or clarification question)
  sessions.py       in-memory session store with TTL
  orchestrator.py   glue: process_message(session, user_msg) → response
  models.py         ChatRequest, ChatResponse, Session, Message
tests/
  test_chat.py      mocked-LLM unit + integration tests
```

修改：
- `app/main.py` — 加 4 个 endpoints
- `app/models.py` — 不动（聊天模型放 `app/llm/models.py` 隔离）
- `requirements.txt` — +`python-dotenv` (读 `.env` 里的 OPENROUTER_API_KEY)
- `README.md` — v5 段
- `CLAUDE.md` — 把 v5 加进 layer table；说明 chat endpoints

---

## LLM Integration Details

### client.py
- `httpx` 直接打 `POST https://openrouter.ai/api/v1/chat/completions`
- 不引入 `openai`/`anthropic`/`google-genai` SDK — provider-agnostic
- 重试一次（仅在 429 / 5xx / parse error）

### Env vars

| Var | Default |
|---|---|
| `OPENROUTER_API_KEY` | (required, 已在 `.env`) |
| `OPENROUTER_MODEL` | `google/gemini-flash-3.5` |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` |
| `CHAT_SESSION_TTL_SEC` | `1800` (30 min) |
| `CHAT_HISTORY_MAX_MESSAGES` | `10` |

### Parser
- System prompt: "You are a strict JSON parser for Texas Hold'em MTT preflop spots. Return ONLY a JSON object matching this schema; use `null` for fields the user didn't specify."
- Pass schema (inline GameState fields) + 3 few-shot examples
- 用 OpenRouter 的 `response_format: {"type": "json_object"}`（JSON mode）
- 验证后转 Pydantic GameState；缺失字段 → 进 `missing_fields`

### Explainer
两条路径：
- **State complete**：直接调 `solver.solve_preflop(state)` (in-process, ~ms)；把 SolveResponse 喂给 LLM；prompt = "You are a poker coach. Given user's question + solver output, explain in 中文：mention recommendation, action_weights, ICM context, 1–2 actionable factors. Never claim GTO; v0 is heuristic."
- **State incomplete**：跳过 solver；LLM 生成澄清问句（"我需要你的位置和 stack — 你 UTG+1 还是 ...?"）

---

## Multi-turn Behavior

- 每个请求把最近 `CHAT_HISTORY_MAX_MESSAGES` 条 messages 喂给 LLM（含 prior parsed_state 作为 hint）
- Follow-up 例（"what if 10BB instead"）：parser 在上一轮 state 基础上 update `effective_stack_bb=10`，其他字段继承
- Session store：进程内 `dict[session_id, SessionRecord]`，lazy TTL（访问时检查过期）

---

## Error Handling

| 场景 | 行为 |
|---|---|
| LLM 返回非 JSON | retry 1 次（更严的 prompt）；仍失败 → 500 + `"LLM parse error"` |
| Pydantic 验证失败 | 当 missing fields 处理，进 ask-for-clarification |
| `hero_hand` / `hero_position` / `effective_stack_bb` 任一缺失 | 跳过 solver，LLM 出澄清问 |
| Session 不存在 / 已 TTL | 404 |
| OpenRouter 5xx / 网络错误 | 重试 1 次；仍失败 → 502 |

---

## Testing Strategy

- 所有 chat unit/integration tests **mock OpenRouter client**（注入 fake → 返回硬编码 JSON）
- 真打 OpenRouter 的 test 标 `@pytest.mark.live`，默认 skip
- 新增至少 8 个测试用例：
  1. `test_parse_clean_query` — "UTG+1 AQo 14BB near bubble" → 完整 GameState
  2. `test_parse_partial_asks_clarification` — "AA" → missing_fields 非空、solver 未被调
  3. `test_followup_inherits_state` — 多轮，"what if 10BB" 拿到 prior state
  4. `test_session_create_and_inspect` — 完整 CRUD
  5. `test_session_ttl_expires` — fake clock + 31 min later → 404
  6. `test_chat_endpoint_full_loop` — TestClient happy path
  7. `test_chat_endpoint_session_404`
  8. `test_solver_recommendation_surfaces_in_answer` — fixture 让 LLM 返 fake explainer，但验证 solver_data 字段是真 v0+v2 结果

---

## Verification

```bash
cd poker_solver_server
source .venv/bin/activate

# 1) 单元/集成测试（所有 LLM 都是 mock）
pytest -q                  # 期望 41 现有 + ~8 新 = ~49 passed

# 2) 实跑（真打 OpenRouter）
export $(cat ../.env | xargs)
uvicorn app.main:app --port 8000 &

sid=$(curl -sX POST localhost:8000/v1/chat/sessions | jq -r .session_id)
curl -sX POST localhost:8000/v1/chat/sessions/$sid/messages \
  -H 'Content-Type: application/json' \
  -d '{"message":"我 UTG+1 AQo 14BB near bubble 怎么打"}' | jq

# 期望 answer 是中文解释；parsed_state 完整；solver_data 含 recommendation=MIXED；usage 含 tokens

curl -sX POST localhost:8000/v1/chat/sessions/$sid/messages \
  -d '{"message":"那如果是 10BB 呢?"}' | jq
# 期望 follow-up inherits hand/position/stage，只改 stack_bb

# 3) Compliance grep（开发时和最终）
grep -riE "import openai|import anthropic|google\.genai" app/    # → none
grep -riE "StreamingResponse|SSE" app/llm/                       # → none
```

---

## Explicit Out of Scope (→ v5.1+)

- LLM tool-calling agent（让 LLM 选 v1/v3/v4 tier）
- Streaming responses (SSE / chunked)
- Persistent session storage (Redis / SQLite / 文件)
- Auth + per-user session ownership
- Per-IP rate limit
- Voice transcript / 图片输入
- Hand history file import / 复盘
- Postflop / multi-way（LLM 遇到这种 query 直接说 out-of-scope）

---

## Critical Files to Reuse

- `app/solver.py:solve_preflop` — 唯一 solver 入口，in-process 调用
- `app/models.py:GameState, SolveResponse` — parser 校验目标 & explainer 输入
- `app/solver.py:normalize_hand, normalize_position` — 在 parser post-validation 用，做最后一道兜底
- 现有 FastAPI app 模式（参考 `app/main.py` 现有 endpoints）
