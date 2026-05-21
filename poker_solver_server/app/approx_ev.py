"""v3: Approximate EV Predictor (low-latency fallback tier).

Per design doc §9.3. Architectural role:

  1. Sub-millisecond fallback when v2's O(n!) Malmuth-Harville is too slow.
  2. Warm-start signal for online CFR (v4) by surfacing high-value spots.
  3. Quickly flag spots that need deeper solving.

This v0-of-v3 is intentionally a small linear model with hand-tuned
coefficients calibrated against v2 outputs. Pure Python — no torch / numpy /
onnx, so import cost is zero and prediction is microsecond-scale. When the
production system later trains a real NN for this tier, the public API
(extract_features → predict_action_evs) stays stable; only the inner
coefficients change.
"""

from __future__ import annotations

from typing import Dict

from .models import GameState
from .solver import (
    bubble_pressure_score,
    hand_strength_score,
    normalize_hand,
    normalize_position,
    position_risk_score,
)


# ============================================================
# Feature extraction
# ============================================================

KNOWN_STAGES = {
    "early", "mid", "near_bubble", "bubble", "itm", "ft_bubble", "ft", "heads_up",
}


def _approximate_icm_current(state: GameState) -> float:
    """O(n) approximation of icm_current (vs v2's O(n!) exact Malmuth-Harville).

    Uses chip-share-weighted payout pool with a simple concavity adjustment
    that bumps short stacks up and big stacks down — capturing the qualitative
    shape of true ICM equity without the combinatorial explosion.
    """
    if not state.table_stacks or not state.payouts:
        return 0.0
    hero_idx = state.hero_index if 0 <= state.hero_index < len(state.table_stacks) else 0
    hero_stack = state.table_stacks[hero_idx].stack_bb
    total_chips = sum(t.stack_bb for t in state.table_stacks)
    if total_chips <= 0:
        return 0.0
    chip_share = hero_stack / total_chips
    payout_pool = sum(state.payouts)
    n_seats = max(2, len(state.table_stacks))
    # 1 - n*share centers around chip_share = 1/n (equal stacks).
    # Above-equal stacks get pulled down; below-equal stacks get bumped up.
    concavity = max(0.10, 1.0 + 0.5 * (1.0 - chip_share * n_seats))
    return chip_share * payout_pool * concavity


def extract_features(state: GameState) -> Dict[str, float]:
    """Vectorize a GameState into a flat feature dict.

    Named dict (rather than a list) so adding a new feature later does
    not shift indices and silently break the linear coefficients.
    """
    try:
        hand_class = normalize_hand(state.hero_hand)
        hand_known = 1.0
    except ValueError:
        hand_class = "??"
        hand_known = 0.0

    try:
        position = normalize_position(state.hero_position)
        position_known = 1.0
    except ValueError:
        position = "BTN"
        position_known = 0.0

    h = hand_strength_score(hand_class) if hand_known else 0.30
    p = position_risk_score(position)
    b = bubble_pressure_score(state.tournament_stage, state.effective_stack_bb)

    icm_approx = _approximate_icm_current(state)
    s = state.effective_stack_bb

    return {
        "hand_strength": h,
        "position_risk": p,
        "bubble_pressure": b,
        "stack_norm": min(1.0, s / 50.0),
        "short_stack": 1.0 if s <= 12 else 0.0,
        "deep_stack": 1.0 if s >= 30 else 0.0,
        "icm_approx": icm_approx,
        "hand_known": hand_known,
        "position_known": position_known,
        "stage_known": 1.0 if state.tournament_stage in KNOWN_STAGES else 0.0,
        "pot_bb_norm": min(1.0, state.pot_bb / 10.0),
        "bias": 1.0,
    }


# ============================================================
# Linear model coefficients (hand-tuned against v2 grid)
# ============================================================

_EV_FOLD_WEIGHTS: Dict[str, float] = {
    # ev_fold ≈ icm_current. The approximation does the work.
    "icm_approx": 1.00,
}

_EV_OPEN_WEIGHTS: Dict[str, float] = {
    "icm_approx": 0.98,
    "hand_strength": 4.0,
    "position_risk": -2.5,
    "bubble_pressure": -1.5,
    "deep_stack": 1.0,
    "short_stack": -0.5,
    "pot_bb_norm": 2.0,
    "bias": -1.0,
}

_EV_SHOVE_WEIGHTS: Dict[str, float] = {
    "icm_approx": 0.95,
    "hand_strength": 8.0,
    "position_risk": -3.0,
    "bubble_pressure": -3.5,
    "short_stack": 4.0,
    "deep_stack": -6.0,
    "stack_norm": -3.0,
    "bias": -2.0,
}


def _linear(features: Dict[str, float], weights: Dict[str, float]) -> float:
    return sum(features.get(k, 0.0) * v for k, v in weights.items())


def _confidence(features: Dict[str, float]) -> float:
    """Confidence proxy: high when all categorical inputs are recognized and
    the stack depth is inside the model's nominal regime."""
    knowns = features["hand_known"] + features["position_known"] + features["stage_known"]
    base = knowns / 3.0
    stack_pen = 0.15 if (features["stack_norm"] >= 0.95 or features["stack_norm"] <= 0.05) else 0.0
    return max(0.0, min(1.0, base - stack_pen))


# ============================================================
# Public API
# ============================================================

def predict_action_evs(state: GameState) -> Dict[str, float]:
    """Fast-tier approximate per-action ICM EVs.

    Returns the same EV keys as v2 (`ev_fold`, `ev_open`, `ev_shove`) plus
    a `confidence` scalar in [0, 1]. Output units match v2 (ICM equity).
    """
    features = extract_features(state)
    return {
        "ev_fold": round(_linear(features, _EV_FOLD_WEIGHTS), 4),
        "ev_open": round(_linear(features, _EV_OPEN_WEIGHTS), 4),
        "ev_shove": round(_linear(features, _EV_SHOVE_WEIGHTS), 4),
        "confidence": round(_confidence(features), 4),
    }
