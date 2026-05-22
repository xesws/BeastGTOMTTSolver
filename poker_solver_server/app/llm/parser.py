from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from .client import OpenRouterClient
from .models import MessageRecord
from .prompts import PARSER_SYSTEM_PROMPT
from ..solver import normalize_hand, normalize_position

logger = logging.getLogger(__name__)


def parse_natural_language(
    message: str,
    history: List[MessageRecord],
    prior_state_dict: Optional[dict] = None,
    client: Optional[OpenRouterClient] = None,
) -> Tuple[dict, List[str], dict]:
    """Parse user natural language query into a GameState dictionary.

    Returns:
        merged_state: The final merged state dict (prior state overlaid with the
            current turn's non-null fields, plus mechanical normalizations like
            hand/position string canonicalization).
        missing_fields: Always returned as an empty list. Strict-gate decision
            on which required fields are missing now lives in
            `orchestrator.build_gamestate`, which can apply derivations the
            parser layer cannot.
        usage: Usage info from the LLM response.
    """
    if client is None:
        client = OpenRouterClient()

    # Construct messages for OpenRouter
    api_messages: List[Dict[str, str]] = [
        {"role": "system", "content": PARSER_SYSTEM_PROMPT}
    ]

    # Add prior state context if available
    if prior_state_dict:
        # Filter out null values for cleaner context
        clean_prior = {k: v for k, v in prior_state_dict.items() if v is not None}
        api_messages.append({
            "role": "system",
            "content": f"Context (Prior parsed state): {json.dumps(clean_prior)}"
        })

    # Add history (up to last CHAT_HISTORY_MAX_MESSAGES)
    for msg in history:
        api_messages.append({"role": msg.role, "content": msg.content})

    # Add current user message
    api_messages.append({"role": "user", "content": message})

    try:
        completion_data = client.chat_completion(
            messages=api_messages,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        logger.error(f"OpenRouter call failed during parsing: {e}")
        return {}, [], {}

    # Extract JSON content
    choices = completion_data.get("choices", [])
    if not choices:
        return {}, [], {}

    content_str = choices[0].get("message", {}).get("content", "").strip()
    usage = completion_data.get("usage", {})
    usage["model"] = completion_data.get("model", client.model)

    try:
        parsed_delta = json.loads(content_str)
    except Exception as e:
        logger.error(f"Failed to parse LLM response JSON: {content_str}. Error: {e}")
        return {}, [], usage

    # Merge delta with prior state
    merged_state = {}
    if prior_state_dict:
        merged_state.update(prior_state_dict)

    # Overwrite prior state with non-null values from current parse delta
    for k, v in parsed_delta.items():
        if v is not None:
            merged_state[k] = v

    # Normalize fields if present and valid
    if "hero_hand" in merged_state and merged_state["hero_hand"] is not None:
        try:
            merged_state["hero_hand"] = normalize_hand(str(merged_state["hero_hand"]))
        except ValueError:
            # Keep as is, it will fail Pydantic validation later
            pass

    if "hero_position" in merged_state and merged_state["hero_position"] is not None:
        try:
            merged_state["hero_position"] = normalize_position(str(merged_state["hero_position"]))
        except ValueError:
            pass

    # Mechanical mirror between hero_stack_bb and effective_stack_bb when one
    # is supplied — definitional in HU play and harmless when both happen to be
    # provided.
    if merged_state.get("effective_stack_bb") is not None and merged_state.get("hero_stack_bb") is None:
        merged_state["hero_stack_bb"] = merged_state["effective_stack_bb"]
    elif merged_state.get("hero_stack_bb") is not None and merged_state.get("effective_stack_bb") is None:
        merged_state["effective_stack_bb"] = merged_state["hero_stack_bb"]

    return merged_state, [], usage
