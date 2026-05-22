# Poker Solver Server

基于 FastAPI 的 v0 启发式 preflop solver + 简化 ICM 诊断服务（heuristic preflop advisor with simplified ICM diagnostics）。

---

## ⚠️ 免责声明

**v0 是启发式建议（heuristic recommendation），不是完整 GTO 解（not a full GTO solve）。**

本服务在 preflop 阶段使用基于手牌强度、位置、ICM bubble pressure 等因子的可解释打分函数（scoring function）给出建议动作（FOLD / OPEN / SHOVE / MIXED）。它不进行 CFR 迭代、不构建完整 range tree、也不计算严格意义上的 Nash 均衡。输出仅供参考。

适用场景：
- 训练（training）：让用户在练习中获得即时反馈
- 复盘（review / hand history analysis）：解释一手牌的决策因子
- 仿真（simulation）：与离线模拟器、bot 训练管线（offline simulators, bot training pipelines）联动

**不适用且明确禁止的场景**：用于在任何线上扑克平台（PokerStars、GGPoker 等）的实时对局中实时取牌、自动决策或以任何方式绕过平台规则。参考仓库根 `docs/texas_holdem_ai_assistant_system_design.md` 的「前置说明：使用边界」一节。

---

## 安装与运行

```bash
cd poker_solver_server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

FastAPI 提供的自动交互式文档：

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

健康检查：`GET http://localhost:8000/health`

---

## 调用示例

仓库内已附带一个 near-bubble、AQo、UTG+1、14bb 的示例请求：

```bash
curl -s -X POST http://localhost:8000/v1/solve/preflop \
  -H 'Content-Type: application/json' \
  --data @examples/request_icm_near_bubble_aqo.json | python -m json.tool
```

---

## 响应字段说明

顶层字段：

| 字段 | 含义 |
| --- | --- |
| `recommendation` | 主推荐动作，取值 `FOLD` / `OPEN` / `SHOVE` / `MIXED`。`MIXED` 表示没有一个动作的权重显著占优，建议查看 `action_weights` |
| `action_weights` | 各候选动作的归一化权重（normalized weights），用于表达混合策略（mixed strategy）的相对倾向 |
| `confidence` | solver 对该推荐的置信度（0.0 - 1.0），由打分边际（margin）和输入完整度决定 |
| `recommended_size_bb` | 在推荐动作为 OPEN / SHOVE 时给出的下注尺度（bb 数）；FOLD 时为 `null` |
| `reason` | 人类可读的简短解释（human-readable rationale），列出主导决策的因子 |
| `diagnostics` | 决策诊断数据，便于训练复盘与 UI 展示 |

`diagnostics` 关键字段：

| 字段 | 含义 |
| --- | --- |
| `hand_strength_score` | 起手牌强度（hand strength）打分，结合点数、同花/非同花、连张性等 |
| `bubble_pressure_score` | bubble 压力（ICM bubble pressure），距离 paid places 越近、stack 分布越紧张分数越高 |
| `position_risk_score` | 位置风险（positional risk）打分，UTG / EP 高于 CO / BTN |
| `icm_current` | 当前筹码量在 ICM 模型下的期望奖金份额（expected prize equity） |
| `icm_if_win` | 假设本手获胜后的 ICM 期望奖金份额 |
| `icm_if_lose` | 假设本手失利后的 ICM 期望奖金份额 |
| `icm_risk_premium` | ICM 风险溢价（risk premium）= 失利损失相对于获胜收益的不对称放大系数，用来收紧 calling / shoving 区间 |

---

## 测试

```bash
pytest -q
```

测试覆盖 API 合约、heuristic 打分边界条件以及 ICM 计算的对称性。

---

## 架构定位

按照设计文档（`docs/texas_holdem_ai_assistant_system_design.md`）§3 中的分层规划，本服务现在实现到 **v4 push/fold 子集**：

| 层 | 状态 | 端点 / 入口 | 备注 |
| --- | --- | --- | --- |
| v0 启发式 | ✅ 已实现 | `POST /v1/solve/preflop` 的 `action_weights` / `recommendation` | 可解释 softmax，依赖 hand_strength × position_risk × bubble_pressure 三层打分 |
| v1 range table | ✅ 已实现 | `POST /v1/lookup/preflop` | 离线 grid 把 v2 的 EV 烤进 (stage × bucket × position × hand_class) 的查表。在线 < 5 ms，完全不调 Malmuth-Harville |
| v2 ICM delta engine | ✅ 已实现 | `POST /v1/solve/preflop` 响应 `diagnostics` 内的 `ev_fold` / `ev_open` / `ev_shove` | 对每个动作分支做 chip outcome → Malmuth-Harville 映射，输出 ICM equity 单位 |
| v3 approx EV model | ✅ 已实现 | `POST /v1/predict/preflop` | 纯 Python 线性近似器，cold-call 比 v2 快 10× 以上；预留接口让未来替换为 NN |
| v4 CFR (push/fold) | ✅ 已实现 | `POST /v1/cfr/preflop` | HU push/fold vanilla CFR + Kuhn baseline。多 stack-depth 蓝图，离线训练 + 在线查表 |
| v4.1+ (multi-street + subgame solving) | ⏳ 规划中 | — | flop/turn/river abstraction + safe/nested subgame resolving |

底层共享：`solver.malmuth_harville` 已重写成 bitmask DP（O(n²·2ⁿ) vs 旧 O(n·n!)），并加了 LRU cache。这让 v0/v2 在重复请求或共享 table_stacks 的批处理上几乎是免费的。

v0 不会消失：在更高阶 solver 因延迟、覆盖度或失败回退（fallback）时，它仍作为最后一道可用的建议来源存在。

---

## v1 / v2 / v3 接口细节

### v1 Range Table（精度量化 + 极低延迟）

```bash
curl -s -X POST http://localhost:8000/v1/lookup/preflop \
  -H 'Content-Type: application/json' \
  --data @examples/request_icm_near_bubble_aqo.json | python -m json.tool
```

返回（实测 ~0.6 ms HTTP 端到端）：

```json
{
  "recommendation": "SHOVE",
  "action_weights": {"FOLD": 0.14, "OPEN": 0.34, "SHOVE": 0.52},
  "ev_fold": 28.52,
  "ev_open": 29.27,
  "ev_shove": 27.68,
  "confidence": 1.0,
  "source": "range_table_v1",
  "matched_stage": "near_bubble",
  "matched_bucket": 12,
  "miss_reason": null
}
```

**网格轴**：

| 轴 | 值 |
| --- | --- |
| `stage` | early / mid / near_bubble / bubble / itm / ft_bubble / ft（7） |
| `stack_bucket` (BB) | 7 / 12 / 20 / 30 / 50（5） |
| `position` | UTG / UTG+1 / UTG+2 / LJ / HJ / CO / BTN / SB / BB（9） |
| `hand_class` | 全 169 个 canonical preflop 类（13 pair + 78 suited + 78 offsuit） |

共 **53,235 cells**，离线 build 时间 ~19 秒，JSON 输出 ~7.9 MB。文件落在 `app/data/range_table.json`，模块启动时优先 load；如不存在则 build 并 save。

**量化注意**：v1 把 `effective_stack_bb` 吸附到最近的 grid bucket，所以 14 BB 会落到 bucket=12，结果可能与 `/v1/solve/preflop` 在精确 14 BB 上略有不同。要保留 stack 精度时用 v0+v2 端点，要极低延迟时用 v1。

### v2 ICM Delta Engine（在 `diagnostics` 中）

### v2 ICM Delta Engine（在 `diagnostics` 中）

`POST /v1/solve/preflop` 返回的 `diagnostics` 块在 v0 字段基础上新增：

| 字段 | 含义 |
| --- | --- |
| `ev_fold` | 弃牌分支的 ICM equity（≈ `icm_current`） |
| `ev_open` | open 分支的 ICM equity，按"all fold / 3bet / called"三分支加权 |
| `ev_shove` | shove 分支的 ICM equity，按"all fold / single-caller-equivalent"加权（multi-way 用 call-prob-weighted 平均近似） |

实现简化点：

- single-caller approximation：multi-way 跟注分支收敛到"任一跟注"的加权平均（避免 5 个对手时 32 种多路组合的枚举爆炸）
- villain calling range 按位置/筹码深度查表
- hand-vs-range equity 用静态 lookup（premium hands ~0.65 / pocket pairs ~0.55-0.85 / 边缘手 ~0.40-0.50）

### v4 CFR Push/Fold Blueprint

```bash
# Train (writes app/data/cfr_blueprint.json, ~2s for 5 stack depths × 10k iters)
python -m scripts.train_cfr

curl -s -X POST http://localhost:8000/v1/cfr/preflop \
  -H 'Content-Type: application/json' \
  -d '{"game_format":"mtt_icm","tournament_stage":"near_bubble","street":"preflop","hero_hand":"AA","hero_position":"SB","action_to_hero":"unopened","hero_stack_bb":10,"effective_stack_bb":10,"pot_bb":1.5,"open_size_bb":2.0,"ante_bb":0,"players_left":2,"paid_places":2,"payouts":[100,0],"hero_index":0,"table_stacks":[{"seat":0,"stack_bb":10},{"seat":1,"stack_bb":10}]}'
```

返回：

```json
{
  "action_probs": {"SHOVE": 0.99995, "FOLD": 5e-05},
  "role": "SB",
  "matched_stack_bb": 10.0,
  "matched_bucket": 9,
  "iterations_trained": 10000,
  "exploitability": 0.001657,
  "source": "cfr_pushfold_v1",
  "hand_class": "AA",
  "miss_reason": null
}
```

**实现内容**：

- `app/cfr/kuhn.py`：教科书 Kuhn poker vanilla CFR，作为算法 sanity baseline。pytest 验证它收敛到已知 Nash 性质（K 总是 call/bet、J 总是 fold、game value ≈ -1/18）。
- `app/cfr/pushfold.py`：heads-up preflop push/fold CFR。10 个 hand-strength buckets × 2 角色 (SB / BB) = 20 个 info sets，全树展开 traversal。`exploitability()` 用 best-response gap 衡量距离 Nash。
- `scripts/train_cfr.py`：在 5 个 stack depth ([5, 7, 10, 15, 20] BB) 各跑 10k iters，导出蓝图到 `app/data/cfr_blueprint.json` (~9 KB)。完整训练 ~2 s。
- `app/cfr/lookup.py`：在线查表。把 `(position → SB/BB, effective_stack → closest depth, hand → bucket)` 映射到蓝图中的 action 分布。

**显式不做的范围**（设计文档 §9.4 vision 中的剩余部分）：

- multi-street（flop / turn / river）— 需要 card abstraction（~thousands of buckets per street）和 multi-round tree。
- safe / nested subgame solving（Brown & Sandholm 2017）— 真正解决"在线 refinement based on observed actions"。
- 多人（>2）CFR — 算法存在但收敛不保证 Nash（公共信念限制）。
- 真神经网络的 blueprint compression — 比如 DeepStack 那种 value network。

这些放在 v4.1+ roadmap，对应论文：

- Zinkevich et al. 2007 (NeurIPS), *Regret Minimization in Games with Incomplete Information*
- Brown & Sandholm 2017, *Safe and Nested Subgame Solving for Imperfect-Information Games*

### v3 Approximate EV Predictor

```bash
curl -s -X POST http://localhost:8000/v1/predict/preflop \
  -H 'Content-Type: application/json' \
  --data @examples/request_icm_near_bubble_aqo.json | python -m json.tool
```

返回：

```json
{
  "ev_fold": 28.67,
  "ev_open": 27.49,
  "ev_shove": 25.73,
  "confidence": 1.0
}
```

特性：

- 纯 Python，**零外部模型依赖**（无 numpy / torch / onnx）
- 用 chip-share × payout-pool × concavity 近似 ICM equity，跳过 Malmuth-Harville 的 O(n!) 枚举
- 线性系数对 v2 输出离线手工拟合
- `confidence` 在所有 categorical 输入（hand class / position / stage）都被识别且 stack 不在极端区间时为 1.0
- 当未来要把这一层换成真 NN，只需替换 `_EV_*_WEIGHTS` 与 `_linear`，公开 API 保持稳定


---

## v5 LLM Orchestrator (Natural Language Frontend)

自然语言前端，接收用户输入的自然语言请求（如：“我 UTG+1 AQo 14BB near bubble 怎么打”），在后台自动进行 GameState 结构化提取、调用 Solver (v0+v2) 进行策略计算，并由 LLM 生成中文策略分析解释。支持多轮对话状态继承与合并。

### 端点设计

- `POST /v1/chat/sessions` — 创建新会话
- `POST /v1/chat/sessions/{session_id}/messages` — 发送聊天消息
- `GET /v1/chat/sessions/{session_id}` — 获取会话历史与当前提取的状态
- `DELETE /v1/chat/sessions/{session_id}` — 结束/删除当前会话

### 调用示例

```bash
# 1) 创建会话
sid=$(curl -s -X POST http://localhost:8000/v1/chat/sessions | python -c "import sys, json; print(json.load(sys.stdin)['session_id'])")

# 2) 提问完整问题
curl -s -X POST http://localhost:8000/v1/chat/sessions/$sid/messages \
  -H 'Content-Type: application/json' \
  -d '{"message": "我 UTG+1 AQo 14BB near bubble 怎么打"}' | python -m json.tool

# 3) 多轮追问，修改筹码量（自动继承先前的手牌、位置和阶段）
curl -s -X POST http://localhost:8000/v1/chat/sessions/$sid/messages \
  -H 'Content-Type: application/json' \
  -d '{"message": "那如果是 10BB 呢?"}' | python -m json.tool
```

### 返回字段说明

- `message_id`: 消息唯一 ID。
- `answer`: LLM 生成的中文策略解释或信息澄清提问。
- `parsed_state`: 当前提取的完整 `GameState`。如信息缺失，返回 `null`。
- `solver_data`: Solver 返回的 `SolveResponse`。如信息缺失，返回 `null`。
- `missing_fields`: 缺失的核心字段列表（例如 `["hero_position"]`）。
- `usage`: 此次调用消耗的 LLM token 统计及使用的模型名称。

