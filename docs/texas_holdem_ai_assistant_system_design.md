# 德州扑克实时策略规划 AI 助手系统设计

## 前置说明：使用边界

本设计建议用于**训练、复盘、仿真、私人允许环境或合规产品场景**。不要把它设计成规避线上扑克平台规则的实时作弊工具，也不要做自动截屏、OCR、自动识别线上牌桌、自动替用户下注等能力。系统可以做“策略分析服务”，但产品侧需要明确使用边界。

---

## 1. 总体架构

核心思路是：**大模型不直接做牌局决策**。大模型负责理解用户输入、规范化牌局状态、调用 Solver API、解释 Solver 输出；真正的策略计算由 Solver Server 完成。

```text
用户 / 前端
  │
  ▼
Input Normalizer
自然语言/表单/JSON → 标准 GameState
  │
  ▼
LLM Orchestrator
- 解析局势
- 补齐缺失字段
- 调用 Solver API
- 把结果翻译成可读建议
  │
  ▼
Solver API Gateway
- 鉴权
- 限流
- 日志
- 延迟预算
- 请求校验
  │
  ▼
Solver Server
  ├── Fast Path Cache
  │     常见 preflop spot、range table、ICM spot 缓存
  │
  ├── Heuristic / Rule Solver v0
  │     当前已实现：preflop 粗略策略建议
  │
  ├── ICM Module
  │     当前已实现：基础 ICM equity 诊断
  │
  ├── Approx EV Model
  │     后续：神经网络预测 EV / action value
  │
  ├── CFR / Subgame Solver
  │     后续：CFR、CFR+、MCCFR、nested subgame solving
  │
  └── Explanation Builder
        把 action weights、risk premium、stack depth 转成解释
```

从算法路线看，德州扑克属于不完全信息博弈，CFR 一类算法长期用于 poker abstraction 和近似 Nash equilibrium 求解。Zinkevich 等人的 CFR 论文明确把 counterfactual regret minimization 用于不完全信息博弈和扑克抽象求解。现代强 poker AI 还会使用 subgame solving；Brown 和 Sandholm 的工作强调，在不完全信息博弈中，子博弈不能简单孤立求解，因此需要 safe/nested subgame solving 这类技术。

---

## 2. 核心模块设计

### 2.1 LLM Orchestrator

大模型层只做三件事：

```text
1. 把用户输入转为结构化 GameState
2. 调用 Solver API
3. 解释 Solver 返回值
```

例如用户输入：

```text
MTT ICM near bubble，我 UTG+1，手牌 AQo，14BB，要 all-in 还是 open？
```

LLM 应转成：

```json
{
  "game_format": "mtt_icm",
  "tournament_stage": "near_bubble",
  "street": "preflop",
  "hero_hand": "AQo",
  "hero_position": "UTG+1",
  "action_to_hero": "unopened",
  "hero_stack_bb": 14,
  "effective_stack_bb": 14
}
```

LLM 不应该自己编造 “GTO 结论”。如果 Solver 返回置信度低或缺少字段，LLM 应说明：

```text
这是 v0 近似建议，不是完整 GTO 解。
当前缺少 payout、完整桌面 stack、后手玩家 stack 和 action history，因此 ICM 部分只做粗略估计。
```

### 2.2 Solver Server

Solver Server 应该拆成以下层级：

```text
API Layer
  - 请求校验
  - schema versioning
  - timeout budget
  - idempotency / trace id

State Normalization
  - 手牌标准化：AhQd → AQo
  - 位置标准化：UTG+1、LJ、HJ、CO、BTN、SB、BB
  - stack bucket：7BB、10BB、15BB、20BB、30BB 等
  - action abstraction：open、3bet、jam、call 等

Decision Engine
  - v0: heuristic preflop solver
  - v1: precomputed range table + cache
  - v2: ICM exact delta + approximate EV model
  - v3: CFR / subgame solving

Response Builder
  - recommendation
  - action weights
  - confidence
  - recommended size
  - diagnostics
  - assumptions / warnings
```

---

## 3. 延迟策略

实时或极低延迟场景不应每次都跑深度 Solver。推荐采用分层策略：

| 层级 | 名称 | 用途 | 目标 |
|---|---|---:|---:|
| L0 | Cache | 高频标准 spot | 1–5ms |
| L1 | Range Table | 预计算 preflop / ICM 表 | 5–20ms |
| L2 | Approx EV Model | 神经网络预测 action EV | 20–80ms |
| L3 | Online Solver | 在线 CFR / subgame solving | 100ms+ |
| L4 | Offline Training | 训练、生成表、评估 exploitability | 非实时 |

当前实现的是 **L1 之前的 v0 heuristic fallback**，它不是 GTO，但可以先把 API、数据结构、服务边界跑起来。

---

## 4. GameState 设计

第一版最关键的是把输入标准化，否则后面 Solver 很难接。

```json
{
  "game_format": "mtt_icm",
  "tournament_stage": "near_bubble",
  "street": "preflop",
  "hero_hand": "AQo",
  "hero_position": "UTG+1",
  "action_to_hero": "unopened",
  "hero_stack_bb": 14,
  "effective_stack_bb": 14,
  "pot_bb": 2.4,
  "open_size_bb": 2.0,
  "ante_bb": 0.1,
  "players_left": 101,
  "paid_places": 99,
  "payouts": [100, 70, 50, 35, 25, 18, 12, 8, 5],
  "hero_index": 3,
  "table_stacks": [
    {"seat": 0, "stack_bb": 26},
    {"seat": 1, "stack_bb": 18},
    {"seat": 2, "stack_bb": 9},
    {"seat": 3, "stack_bb": 14},
    {"seat": 4, "stack_bb": 31},
    {"seat": 5, "stack_bb": 22},
    {"seat": 6, "stack_bb": 7},
    {"seat": 7, "stack_bb": 44},
    {"seat": 8, "stack_bb": 11}
  ],
  "max_latency_ms": 100
}
```

ICM 的核心作用是把 tournament chips 映射成 prize equity，而不是简单按 chip EV 评估决策；ICM 通常基于 stack sizes 估计各玩家名次概率，再结合 payout structure 计算锦标赛权益。

---

## 5. API 设计

当前 v0 工程中已经实现：

```text
GET  /health
POST /v1/solve/preflop
```

FastAPI 会基于类型定义生成 OpenAPI schema 和交互式文档，其默认文档入口通常包括 Swagger UI `/docs` 和 ReDoc `/redoc`。

启动后可以访问：

```text
http://localhost:8000/docs
```

---

## 6. v0 Solver Server 已实现内容

v0 版本包括：

```text
poker_solver_server/
  app/
    __init__.py
    main.py          # FastAPI app
    models.py        # Pydantic request/response schema
    solver.py        # heuristic preflop solver + ICM diagnostics
  examples/
    request_icm_near_bubble_aqo.json
  tests/
    test_solver.py
  requirements.txt
  README.md
```

运行方式：

```bash
cd poker_solver_server

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

调用示例：

```bash
curl -s -X POST http://localhost:8000/v1/solve/preflop \
  -H 'Content-Type: application/json' \
  --data @examples/request_icm_near_bubble_aqo.json | python -m json.tool
```

本地测试结果：

```text
3 passed
```

---

## 7. AQo / UTG+1 / Near Bubble 示例输出

对于示例输入：

```text
MTT ICM near bubble
Hero: AQo
Position: UTG+1
Effective stack: 14BB
Action to hero: unopened
```

v0 返回的是接近的 mixed spot：

```json
{
  "recommendation": "MIXED",
  "action_weights": {
    "FOLD": 0.1033,
    "OPEN": 0.4586,
    "SHOVE": 0.4381
  },
  "confidence": 0.471,
  "recommended_size_bb": 2.0,
  "reason": "Best action is close; use a mixed strategy. Normalized hand=AQo, position=UTG+1, effective_stack=14.0BB. The v0 model scores hand strength=0.86, ICM/bubble pressure=0.77, position risk=0.95. Action weights: OPEN=0.46, SHOVE=0.44, FOLD=0.10.",
  "diagnostics": {
    "hand_strength_score": 0.86,
    "bubble_pressure_score": 0.77,
    "position_risk_score": 0.95,
    "effective_stack_bb": 14.0,
    "normalized_hand": "AQo",
    "icm_current": 30.508275,
    "icm_if_win": 44.781784,
    "icm_if_lose": 5.0,
    "icm_risk_premium": 0.6412
  }
}
```

解释成产品语言：

```text
当前 v0 不给出绝对 GTO 结论，而是判断 OPEN 和 SHOVE 很接近。
在 14BB、UTG+1、near bubble、AQo 的情况下，AQo 牌力足够强，但 ICM/bubble pressure 较高，直接全下的风险溢价也较高。
因此 v0 建议采用 mixed 策略：更偏向小 open，但 shove 也在可选范围内。
```

一个简单产品规则可以先这样展示：

```text
≤ 10BB：AQo UTG+1 near bubble 更偏 shove
11–16BB：open / shove 接近，按桌面压力、后手短码、赏金结构、对手跟注倾向调整
17BB+：更偏 open，不建议直接 open shove
```

注意：这是 v0 的启发式建议，不是完整 GTO 解算结果。

---

## 8. 当前 v0 Solver 的决策逻辑

v0 使用几个可解释的分数：

```text
hand_strength_score
  AQo = strong hand bucket

bubble_pressure_score
  near bubble + medium stack → pressure 较高

position_risk_score
  UTG / UTG+1 风险较高

effective_stack_bb
  决定 open 和 shove 的相对吸引力

icm_risk_premium
  使用简化 ICM equity 对 all-in win/lose 分支做诊断
```

核心思想：

```text
SHOVE score =
  hand strength
  + short-stack urgency
  - early-position penalty
  - ICM shove penalty
  - deep-stack shove penalty

OPEN score =
  hand strength
  + playable stack depth
  - early-position penalty
  - ICM penalty for marginal hands

FOLD score =
  weak hand tendency
  + ICM pressure
  + early-position risk
```

然后用 softmax 得到 action weights。

---

## 9. 后续升级路线

### 9.1 v1：Range Table Solver

先不要直接上在线 CFR。更合理的是先做预计算表：

```text
key = {
  game_format,
  tournament_stage,
  position,
  effective_stack_bucket,
  action_to_hero,
  hand_class,
  icm_pressure_bucket
}
```

返回：

```json
{
  "AQo": {
    "open": 0.54,
    "shove": 0.41,
    "fold": 0.05
  }
}
```

这能极大降低实时延迟。

### 9.2 v2：ICM Delta Engine

v0 已经有基础 ICM equity 计算，但还没有完整地枚举：

```text
open → 后手 fold / call / 3bet / jam
shove → everyone fold / one caller / multiple callers
fold → 保留 stack
```

v2 应该加入：

```text
EV_fold
EV_open
EV_shove
EV_call
EV_3bet
```

并且每个分支都用：

```text
chip outcome → ICM equity
```

而不是只看 chip EV。

### 9.3 v3：Approx EV Model

训练一个模型做快速预测：

```text
Input:
  - hand embedding
  - position embedding
  - stack vector
  - payout embedding
  - action history embedding
  - table pressure features

Output:
  - EV_fold
  - EV_open
  - EV_shove
  - regret estimate
  - confidence / uncertainty
```

模型不直接替代 Solver，而是用于：

```text
1. 低延迟 fallback
2. 给在线 Solver 提供 warm start
3. 发现高价值 spot，触发更深求解
```

### 9.4 v4：CFR / Subgame Solver

长期架构应该是：

```text
Offline:
  - 建 abstraction
  - 训练 blueprint strategy
  - 生成 preflop / flop / turn / river 表

Online:
  - 根据当前局势取 blueprint
  - 做 action abstraction
  - 对关键节点做 subgame solving
  - 输出 action distribution
```

因为不完全信息博弈中的子博弈求解不能简单当作完全独立问题处理，长期要采用 safe/nested subgame solving 这类方法，而不是简单从当前节点往下暴力搜索。

---

## 10. 推荐的产品响应格式

前端不要只显示 “All-in” 或 “Open”。建议显示：

```text
建议：Mixed，偏 Open

操作权重：
- Open 46%
- Shove 44%
- Fold 10%

推荐 sizing：
- Open: 2.0BB

解释：
AQo 在 UTG+1 属于强牌，但 near bubble 下 ICM 压力较高。
14BB 处于 open 与 shove 都可行的边界区间。
如果后手玩家偏紧、短码很多、fold equity 高，shove 权重上升。
如果后手有大码覆盖你且跟注范围不够紧，open 更稳。
```

这比直接给单点结论更符合 poker solver 产品的表达方式。

---

## 参考资料

1. Zinkevich et al., “Regret Minimization in Games with Incomplete Information”, NeurIPS Proceedings.  
   <https://papers.nips.cc/paper/3306-regret-minimization-in-games-with-incomplete-information>
2. Brown & Sandholm, “Safe and Nested Subgame Solving for Imperfect-Information Games”, arXiv.  
   <https://arxiv.org/abs/1705.02955>
3. GTO Wizard, “ICM Basics”.  
   <https://blog.gtowizard.com/icm-basics/>
4. FastAPI Documentation, “Metadata and Docs URLs”.  
   <https://fastapi.tiangolo.com/tutorial/metadata/>
