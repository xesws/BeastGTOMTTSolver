from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .models import GameState, SolveResponse


# ============================================================
# Normalization layer
# ============================================================

RANK_ORDER = "23456789TJQKA"
RANK_TO_VALUE = {r: i for i, r in enumerate(RANK_ORDER, start=2)}


def normalize_hand(hand: str) -> str:
    """Normalize hand input to canonical class (e.g. 'AhQd' -> 'AQo', 'AhAd' -> 'AA').

    Accepts either raw two-card form (e.g. 'AhQd') or already-canonical
    classes (e.g. 'AQo', 'AA'). Suit chars are case-insensitive.
    """
    h = hand.strip().replace(" ", "")
    # Already canonical: pair like "AA"
    if len(h) == 2 and h[0] == h[1] and h[0].upper() in RANK_TO_VALUE:
        return h[0].upper() + h[1].upper()
    # Already canonical: "AQo" / "AKs"
    if len(h) == 3 and h[0].upper() in RANK_TO_VALUE and h[1].upper() in RANK_TO_VALUE and h[2].lower() in ("o", "s"):
        r1, r2 = h[0].upper(), h[1].upper()
        if RANK_TO_VALUE[r1] < RANK_TO_VALUE[r2]:
            r1, r2 = r2, r1
        return f"{r1}{r2}{h[2].lower()}"
    # Two-card form: AhQd
    if len(h) == 4:
        r1, s1, r2, s2 = h[0].upper(), h[1].lower(), h[2].upper(), h[3].lower()
        if r1 not in RANK_TO_VALUE or r2 not in RANK_TO_VALUE:
            raise ValueError(f"unrecognized hand: {hand!r}")
        if r1 == r2:
            return r1 + r2
        if RANK_TO_VALUE[r1] < RANK_TO_VALUE[r2]:
            r1, r2 = r2, r1
            s1, s2 = s2, s1
        suited = "s" if s1 == s2 else "o"
        return f"{r1}{r2}{suited}"
    raise ValueError(f"unrecognized hand: {hand!r}")


POSITION_CANON = {
    "UTG": "UTG",
    "UTG+1": "UTG+1",
    "UTG1": "UTG+1",
    "UTG+2": "UTG+2",
    "UTG2": "UTG+2",
    "MP": "UTG+2",
    "MP1": "UTG+1",
    "MP2": "UTG+2",
    "LJ": "LJ",
    "LOJACK": "LJ",
    "HJ": "HJ",
    "HIJACK": "HJ",
    "CO": "CO",
    "CUTOFF": "CO",
    "BTN": "BTN",
    "BUTTON": "BTN",
    "SB": "SB",
    "SMALL_BLIND": "SB",
    "BB": "BB",
    "BIG_BLIND": "BB",
}


def normalize_position(pos: str) -> str:
    p = pos.strip().upper().replace(" ", "_")
    if p in POSITION_CANON:
        return POSITION_CANON[p]
    raise ValueError(f"unrecognized position: {pos!r}")


STACK_BUCKETS_BB: List[int] = [7, 10, 12, 15, 20, 25, 30, 40, 60, 100]


def stack_bucket(stack_bb: float) -> int:
    """Return the canonical stack bucket label (in BB) for a given stack."""
    for b in STACK_BUCKETS_BB:
        if stack_bb <= b:
            return b
    return STACK_BUCKETS_BB[-1] + 1


# ============================================================
# Heuristic scoring tables
# ============================================================

HAND_STRENGTH: Dict[str, float] = {
    # pairs
    "AA": 1.00, "KK": 0.97, "QQ": 0.94, "JJ": 0.90, "TT": 0.86,
    "99": 0.80, "88": 0.74, "77": 0.68, "66": 0.62, "55": 0.56,
    "44": 0.50, "33": 0.44, "22": 0.38,
    # broadways
    "AKs": 0.95, "AKo": 0.93,
    "AQs": 0.91, "AQo": 0.86,
    "AJs": 0.87, "AJo": 0.81,
    "ATs": 0.83, "ATo": 0.74,
    "KQs": 0.86, "KQo": 0.79,
    "KJs": 0.83, "KJo": 0.74,
    "KTs": 0.80, "KTo": 0.69,
    "QJs": 0.82, "QJo": 0.73,
    "QTs": 0.78, "QTo": 0.67,
    "JTs": 0.78, "JTo": 0.66,
    # ace-rag
    "A9s": 0.76, "A9o": 0.65,
    "A8s": 0.73, "A8o": 0.61,
    "A7s": 0.71, "A7o": 0.57,
    "A6s": 0.69, "A6o": 0.53,
    "A5s": 0.71, "A5o": 0.52,
    "A4s": 0.69, "A4o": 0.49,
    "A3s": 0.68, "A3o": 0.47,
    "A2s": 0.67, "A2o": 0.45,
    # king-rag
    "K9s": 0.72, "K9o": 0.59,
    "K8s": 0.65, "K8o": 0.50,
    "K7s": 0.62, "K7o": 0.46,
    "K6s": 0.58, "K6o": 0.42,
    "K5s": 0.55, "K5o": 0.39,
    # queen-rag
    "Q9s": 0.70, "Q9o": 0.56,
    "Q8s": 0.63, "Q8o": 0.46,
    # connectors
    "J9s": 0.68, "J9o": 0.53,
    "T9s": 0.69, "T9o": 0.54,
    "T8s": 0.62, "T8o": 0.46,
    "98s": 0.63, "98o": 0.47,
    "97s": 0.56, "97o": 0.40,
    "87s": 0.58, "87o": 0.41,
    "76s": 0.54, "76o": 0.37,
    "65s": 0.50, "65o": 0.34,
}


def hand_strength_score(hand_class: str) -> float:
    return HAND_STRENGTH.get(hand_class, 0.30)


POSITION_RISK: Dict[str, float] = {
    "UTG": 1.00,
    "UTG+1": 0.95,
    "UTG+2": 0.85,
    "LJ": 0.75,
    "HJ": 0.60,
    "CO": 0.40,
    "BTN": 0.15,
    "SB": 0.50,
    "BB": 0.40,
}


def position_risk_score(position: str) -> float:
    return POSITION_RISK.get(position, 0.50)


STAGE_BASELINE: Dict[str, float] = {
    "early": 0.10,
    "mid": 0.30,
    "near_bubble": 0.77,
    "bubble": 0.92,
    "itm": 0.40,
    "ft_bubble": 0.85,
    "ft": 0.55,
    "heads_up": 0.40,
}


def bubble_pressure_score(stage: str, effective_stack_bb: float) -> float:
    base = STAGE_BASELINE.get(stage, 0.30)
    if effective_stack_bb <= 5:
        factor = 0.85
    elif effective_stack_bb <= 25:
        factor = 1.00
    elif effective_stack_bb <= 50:
        factor = 0.85
    else:
        factor = 0.70
    return min(1.0, base * factor)


# ============================================================
# ICM (Malmuth-Harville, simplified single-table)
# ============================================================

def malmuth_harville(stacks: Sequence[float], payouts: Sequence[float]) -> List[float]:
    """Compute per-player ICM equity via Malmuth-Harville.

    Bitmask DP over live players: O(n^2 * 2^n) time, O(n * 2^n) memory.
    For n=9 (full table) this is ~41k operations vs the naive O(n*n!) ≈ 3.3M.
    The numerical answer is identical to the brute-force enumeration.

    Dead players (stack <= 0) are pre-assigned the lowest available payouts
    in seat order, matching the v0 implementation's tie-breaking convention.

    Repeated calls with identical (stacks, payouts) are served from an
    internal LRU cache — the v1 range-table builder relies on this heavily.
    """
    return list(_malmuth_harville_cached(tuple(stacks), tuple(payouts)))


@lru_cache(maxsize=50000)
def _malmuth_harville_cached(
    stacks: Tuple[float, ...], payouts: Tuple[float, ...]
) -> Tuple[float, ...]:
    n = len(stacks)
    if n == 0:
        return tuple()
    pay = list(payouts[:n])
    while len(pay) < n:
        pay.append(0.0)

    ev: List[float] = [0.0] * n
    live_indices = [i for i in range(n) if stacks[i] > 0]
    dead_indices = [i for i in range(n) if stacks[i] <= 0]
    m = len(live_indices)

    # Dead players occupy the lowest ranks (= worst payouts) in seat order.
    for offset, idx in enumerate(dead_indices):
        pos = m + offset
        if pos < n:
            ev[idx] = pay[pos]

    if m == 0:
        return tuple(ev)
    if m == 1:
        ev[live_indices[0]] = pay[0]
        return tuple(ev)

    live_stacks = [stacks[i] for i in live_indices]
    # Live players compete for ranks [0, m-1]; pay[0..m-1] is their payout range.

    # f[mask] is a list of length m: f[mask][j] = E[payout to live-player j
    # given remaining live players = mask]. The rank assigned to the next
    # pick is (m - popcount(mask)).
    full_mask = (1 << m) - 1
    f: List[Optional[List[float]]] = [None] * (full_mask + 1)

    # Base case: single-player masks.
    for j in range(m):
        vec = [0.0] * m
        vec[j] = pay[m - 1]
        f[1 << j] = vec

    # Iterate masks in increasing popcount order so subproblems are ready.
    for mask in sorted(range(1, full_mask + 1), key=lambda mk: bin(mk).count("1")):
        pc = bin(mask).count("1")
        if pc <= 1:
            continue
        rank = m - pc
        in_mask = [j for j in range(m) if mask & (1 << j)]
        total = sum(live_stacks[j] for j in in_mask)
        if total <= 0:  # pragma: no cover — all-zero live should not occur
            continue
        vec = [0.0] * m
        for j in in_mask:
            p_j_first = live_stacks[j] / total
            vec[j] += p_j_first * pay[rank]
            sub_vec = f[mask ^ (1 << j)]
            for i in in_mask:
                if i == j:
                    continue
                vec[i] += p_j_first * sub_vec[i]
        f[mask] = vec

    final = f[full_mask]
    for j_live in range(m):
        ev[live_indices[j_live]] += final[j_live]
    return tuple(ev)


def icm_diagnostics(state: GameState) -> Dict[str, float]:
    if not state.table_stacks or not state.payouts:
        return {
            "icm_current": 0.0,
            "icm_if_win": 0.0,
            "icm_if_lose": 0.0,
            "icm_risk_premium": 0.0,
        }
    stacks = [t.stack_bb for t in state.table_stacks]
    hero_idx = state.hero_index if 0 <= state.hero_index < len(stacks) else 0

    ev_current = malmuth_harville(stacks, state.payouts)
    icm_current = ev_current[hero_idx]

    stacks_lose = list(stacks)
    stacks_lose[hero_idx] = 0.0
    ev_lose = malmuth_harville(stacks_lose, state.payouts)
    icm_if_lose = ev_lose[hero_idx]

    stacks_win = list(stacks)
    stacks_win[hero_idx] = stacks[hero_idx] * 2.0
    ev_win = malmuth_harville(stacks_win, state.payouts)
    icm_if_win = ev_win[hero_idx]

    upside = max(0.0, icm_if_win - icm_current)
    downside = max(0.0, icm_current - icm_if_lose)
    total_swing = upside + downside
    risk_premium = downside / total_swing if total_swing > 0 else 0.5

    return {
        "icm_current": round(icm_current, 6),
        "icm_if_win": round(icm_if_win, 6),
        "icm_if_lose": round(icm_if_lose, 6),
        "icm_risk_premium": round(risk_premium, 4),
    }


# ============================================================
# Action scoring
# ============================================================

def _open_value(stack_bb: float) -> float:
    return max(0.0, min(1.0, (stack_bb - 8) / 14))


def _shove_value(stack_bb: float) -> float:
    return max(0.0, min(1.0, (19 - stack_bb) / 12))


def _shove_depth_penalty(stack_bb: float) -> float:
    return max(0.0, (stack_bb - 12) / 16)


def score_actions(
    *,
    hand_strength: float,
    bubble_pressure: float,
    position_risk: float,
    icm_risk_premium: float,
    effective_stack_bb: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return (raw_scores, softmax_weights) over FOLD / OPEN / SHOVE.

    Score design lives in design doc §8. Coefficients are hand-tuned so that
    AQo / UTG+1 / 14BB / near_bubble lands close to the §7 example.
    """
    h = hand_strength
    b = bubble_pressure
    p = position_risk
    rp = icm_risk_premium
    s = effective_stack_bb

    open_v = _open_value(s)
    shove_v = _shove_value(s)
    shove_depth_pen = _shove_depth_penalty(s)
    bubble_marginal_pen = 0.3 * b * (1.0 - h)
    icm_open_pen = 0.2 * rp * (1.0 - h)

    open_score = 2.0 * h + 1.0 * open_v - 0.5 * p - bubble_marginal_pen - icm_open_pen
    shove_score = 2.0 * h + 1.0 * shove_v - 0.5 * p - 0.5 * shove_depth_pen - 0.15 * rp
    fold_score = 1.0 * (1.0 - h) + 0.7 * b + 0.3 * p

    raw = {"FOLD": fold_score, "OPEN": open_score, "SHOVE": shove_score}

    tau = 0.6
    max_l = max(raw.values()) / tau
    exp_l = {k: math.exp(v / tau - max_l) for k, v in raw.items()}
    total = sum(exp_l.values())
    weights = {k: round(v / total, 4) for k, v in exp_l.items()}
    return raw, weights


# ============================================================
# Recommendation + assembly
# ============================================================

MIXED_MARGIN = 0.15


def _pick_recommendation(weights: Dict[str, float]) -> str:
    ordered = sorted(weights.items(), key=lambda kv: -kv[1])
    top_action, top_w = ordered[0]
    _, second_w = ordered[1]
    if top_w - second_w < MIXED_MARGIN:
        return "MIXED"
    return top_action


def _recommended_size_bb(recommendation: str, weights: Dict[str, float], state: GameState) -> Any:
    if recommendation == "SHOVE":
        return state.effective_stack_bb
    open_w = weights.get("OPEN", 0.0)
    shove_w = weights.get("SHOVE", 0.0)
    if recommendation == "OPEN" or (recommendation == "MIXED" and open_w >= shove_w):
        return state.open_size_bb if state.open_size_bb else 2.0
    if recommendation == "MIXED" and shove_w > open_w:
        # mixed leaning shove — surface shove sizing as "all-in"
        return state.effective_stack_bb
    return None


def _build_reason(
    *,
    recommendation: str,
    weights: Dict[str, float],
    hand_class: str,
    position: str,
    effective_stack_bb: float,
    hand_strength: float,
    bubble_pressure: float,
    position_risk: float,
) -> str:
    if recommendation == "MIXED":
        head = "Best action is close; use a mixed strategy."
    else:
        head = f"Best action is {recommendation}."
    return (
        f"{head} Normalized hand={hand_class}, position={position}, "
        f"effective_stack={effective_stack_bb:.1f}BB. "
        f"The v0 model scores hand strength={hand_strength:.2f}, "
        f"ICM/bubble pressure={bubble_pressure:.2f}, position risk={position_risk:.2f}. "
        f"Action weights: OPEN={weights.get('OPEN', 0):.2f}, "
        f"SHOVE={weights.get('SHOVE', 0):.2f}, FOLD={weights.get('FOLD', 0):.2f}."
    )


def solve_preflop(state: GameState) -> SolveResponse:
    hand_class = normalize_hand(state.hero_hand)
    position = normalize_position(state.hero_position)

    h = hand_strength_score(hand_class)
    p = position_risk_score(position)
    b = bubble_pressure_score(state.tournament_stage, state.effective_stack_bb)
    icm = icm_diagnostics(state)
    rp = icm["icm_risk_premium"]

    _raw, weights = score_actions(
        hand_strength=h,
        bubble_pressure=b,
        position_risk=p,
        icm_risk_premium=rp,
        effective_stack_bb=state.effective_stack_bb,
    )

    recommendation = _pick_recommendation(weights)
    size = _recommended_size_bb(recommendation, weights, state)
    confidence = round(max(weights.values()), 4)

    reason = _build_reason(
        recommendation=recommendation,
        weights=weights,
        hand_class=hand_class,
        position=position,
        effective_stack_bb=state.effective_stack_bb,
        hand_strength=h,
        bubble_pressure=b,
        position_risk=p,
    )

    diagnostics: Dict[str, Any] = {
        "hand_strength_score": round(h, 4),
        "bubble_pressure_score": round(b, 4),
        "position_risk_score": round(p, 4),
        "effective_stack_bb": float(state.effective_stack_bb),
        "normalized_hand": hand_class,
        "normalized_position": position,
        "stack_bucket_bb": stack_bucket(state.effective_stack_bb),
        **icm,
    }

    # v2 (ICM Delta Engine) — per-action ICM equity. Lives alongside v0
    # heuristic weights; does not (yet) drive the recommendation.
    try:
        from .icm_delta import compute_action_evs  # local import to avoid cycle at module load
        diagnostics.update(compute_action_evs(state))
    except Exception:  # pragma: no cover — v2 is informational, never breaks v0
        pass

    return SolveResponse(
        recommendation=recommendation,
        action_weights=weights,
        confidence=confidence,
        recommended_size_bb=size,
        reason=reason,
        diagnostics=diagnostics,
    )
