from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .client import OpenRouterClient
from .prompts import EXPLAINER_SYSTEM_PROMPT, CLARIFICATION_SYSTEM_PROMPT
from ..models import GameState, SolveResponse


def explain_solve_result(
    user_msg: str,
    state: GameState,
    solve_res: SolveResponse,
    client: Optional[OpenRouterClient] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Generate a Chinese coaching explanation for the solver result using the LLM."""
    if client is None:
        client = OpenRouterClient()

    # Format the prompt context
    state_json = json.dumps(state.model_dump(), indent=2, ensure_ascii=False)
    solve_res_json = json.dumps(solve_res.model_dump(), indent=2, ensure_ascii=False)

    context_prompt = (
        f"User Message: {user_msg}\n\n"
        f"Solver Input (GameState):\n{state_json}\n\n"
        f"Solver Output (SolveResponse):\n{solve_res_json}\n"
    )

    api_messages = [
        {"role": "system", "content": EXPLAINER_SYSTEM_PROMPT},
        {"role": "user", "content": context_prompt},
    ]

    try:
        completion_data = client.chat_completion(messages=api_messages)
        choices = completion_data.get("choices", [])
        if choices:
            answer = choices[0].get("message", {}).get("content", "").strip()
            usage = completion_data.get("usage", {})
            usage["model"] = completion_data.get("model", client.model)
            return answer, usage
    except Exception as e:
        # Fallback explanation if API fails
        answer = (
            f"抱歉，分析服务暂时不可用，但后台已成功计算出结果：\n"
            f"主推荐动作：{solve_res.recommendation}\n"
            f"权重分布：{solve_res.action_weights}\n"
            f"推荐下注大小：{solve_res.recommended_size_bb} BB\n"
            f"简要原因：{solve_res.reason}"
        )
        return answer, {}

    return "生成解释失败", {}


def generate_clarification(
    user_msg: str,
    missing_fields: List[str],
    client: Optional[OpenRouterClient] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Ask the LLM to generate a friendly Chinese clarification request for missing fields."""
    if client is None:
        client = OpenRouterClient()

    context_prompt = (
        f"User Message: {user_msg}\n"
        f"Missing Core Fields: {', '.join(missing_fields)}\n"
    )

    api_messages = [
        {"role": "system", "content": CLARIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": context_prompt},
    ]

    try:
        completion_data = client.chat_completion(messages=api_messages)
        choices = completion_data.get("choices", [])
        if choices:
            answer = choices[0].get("message", {}).get("content", "").strip()
            usage = completion_data.get("usage", {})
            usage["model"] = completion_data.get("model", client.model)
            return answer, usage
    except Exception as e:
        # Fallback question if API fails
        fields_cn = []
        if "hero_hand" in missing_fields:
            fields_cn.append("手牌(例如 AA)")
        if "hero_position" in missing_fields:
            fields_cn.append("位置(例如 BTN)")
        if "effective_stack_bb" in missing_fields:
            fields_cn.append("筹码量(例如 14BB)")
        answer = f"我需要你补充以下信息：{ '、'.join(fields_cn) }，然后我才能帮你进行策略计算。"
        return answer, {}

    return "我需要更多信息来进行扑克策略的求解。请告诉我的您的手牌、位置和筹码量是多少？", {}
