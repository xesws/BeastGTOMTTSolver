"""Online lookup against the trained push/fold blueprint.

Pure dict access — no CFR iteration at request time. Matches the
requested (position → role, effective_stack → nearest depth, hand → bucket)
against the precomputed blueprint and returns the trained action
distribution.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from ..models import GameState
from ..solver import hand_strength_score, normalize_hand, normalize_position
from .pushfold import NUM_BUCKETS, hand_to_bucket

_BLUEPRINT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "cfr_blueprint.json",
)
_BLUEPRINT: Optional[Dict[str, Any]] = None


def _ensure_loaded() -> Dict[str, Any]:
    global _BLUEPRINT
    if _BLUEPRINT is None:
        with open(_BLUEPRINT_PATH) as f:
            _BLUEPRINT = json.load(f)
    return _BLUEPRINT


def _closest_stack(stack_bb: float, available: List[float]) -> float:
    return min(available, key=lambda s: abs(s - stack_bb))


def _miss(reason: str) -> Dict[str, Any]:
    return {
        "action_probs": {"FOLD": 1.0},
        "role": None,
        "matched_stack_bb": None,
        "matched_bucket": None,
        "iterations_trained": 0,
        "exploitability": 0.0,
        "source": "cfr_pushfold_v1",
        "hand_class": None,
        "miss_reason": reason,
    }


def lookup(state: GameState) -> Dict[str, Any]:
    """Map a GameState to a trained CFR action distribution."""
    try:
        position = normalize_position(state.hero_position)
    except ValueError as e:
        return _miss(f"unknown position: {e}")
    try:
        hand_class = normalize_hand(state.hero_hand)
    except ValueError as e:
        return _miss(f"unknown hand: {e}")

    if position == "SB":
        role = "SB"
    elif position == "BB":
        role = "BB"
    else:
        return _miss(
            f"position {position!r} not in HU push/fold scope (SB/BB only)"
        )

    blueprint = _ensure_loaded()
    stack_depths: List[float] = blueprint["stack_depths_bb"]
    matched_stack = _closest_stack(state.effective_stack_bb, stack_depths)
    bp = blueprint["blueprints"][str(matched_stack)]

    bucket = hand_to_bucket(hand_strength_score(hand_class))
    if role == "SB":
        action_probs = bp["sb_strategy_by_bucket"][bucket]
    else:
        action_probs = bp["bb_strategy_by_bucket"][bucket]

    return {
        "action_probs": dict(action_probs),
        "role": role,
        "matched_stack_bb": matched_stack,
        "matched_bucket": bucket,
        "iterations_trained": int(bp.get("iterations_trained", 0)),
        "exploitability": float(bp.get("exploitability", 0.0)),
        "source": "cfr_pushfold_v1",
        "hand_class": hand_class,
        "miss_reason": None,
    }
