"""v2: ICM Delta Engine.

Per design doc §9.2. For each candidate hero action (FOLD / OPEN / SHOVE),
enumerate the relevant terminal chip outcomes, then map each chip outcome
back to ICM equity via the Malmuth-Harville model already in solver.py.

This is a deliberate simplification of full game-tree EV:
- SHOVE: single-caller approximation (multi-way calls collapsed into the
  most-likely single caller's branch). Acceptable because in MTT ICM
  shove spots multi-way calls are rare and contribute small probability.
- OPEN: three coarse branches (all fold / called / 3bet). Postflop is
  approximated as a single chip-EV-neutral lump — v2 does not model
  postflop, that lands in v3+ approx model and v4 CFR.
- FOLD: chip stack unchanged → ICM equity = icm_current.

All EVs returned are in **ICM equity units** (same units as `icm_current`
in v0 diagnostics), not chip BB.
"""

from __future__ import annotations

from typing import Dict, List

from .models import GameState
from .solver import (
    HAND_STRENGTH,
    bubble_pressure_score,
    hand_strength_score,
    malmuth_harville,
    normalize_hand,
    normalize_position,
)


# ============================================================
# Villain modelling — calling ranges and hand-vs-range equity
# ============================================================

def _call_probability(
    *,
    hero_stack_bb: float,
    villain_stack_bb: float,
    villain_position: str,
    pot_bb: float,
) -> float:
    """Probability that a given opponent calls hero's all-in.

    Hand-tuned approximations:
    - Blinds defend wider (already invested chips).
    - Very short villains are already committed, call wider.
    - Very deep villains have play, fold tighter to a small shove.
    - Short hero shoves attract loose calls (low risk premium).
    """
    base = 0.18
    if villain_position in ("SB", "BB"):
        base += 0.08
    if villain_stack_bb <= max(5.0, hero_stack_bb * 0.5):
        base += 0.07
    if villain_stack_bb >= hero_stack_bb * 3.0:
        base -= 0.05
    if hero_stack_bb <= 10:
        base += 0.06
    if hero_stack_bb >= 25:
        base -= 0.04
    pot_odds_bump = min(0.05, pot_bb / 50.0)
    base += pot_odds_bump
    return max(0.04, min(0.55, base))


# Equity of hero's hand vs an average MTT ICM calling range.
# Tuned conservatively: even premium hands lose ~15% to calling range mix.
_HAND_VS_RANGE_EQUITY: Dict[str, float] = {
    "AA": 0.85, "KK": 0.82, "QQ": 0.79, "JJ": 0.76, "TT": 0.72,
    "99": 0.62, "88": 0.58, "77": 0.55, "66": 0.52, "55": 0.50,
    "44": 0.48, "33": 0.46, "22": 0.44,
    "AKs": 0.65, "AKo": 0.63,
    "AQs": 0.60, "AQo": 0.55,
    "AJs": 0.56, "AJo": 0.50,
    "ATs": 0.54, "ATo": 0.46,
    "KQs": 0.52, "KQo": 0.45,
    "KJs": 0.48, "KJo": 0.42,
    "QJs": 0.46, "QJo": 0.40,
    "JTs": 0.45, "JTo": 0.39,
}


def hand_vs_range_equity(hand_class: str) -> float:
    """Hero hand's equity vs a typical MTT ICM calling range."""
    if hand_class in _HAND_VS_RANGE_EQUITY:
        return _HAND_VS_RANGE_EQUITY[hand_class]
    # Fall back: scale from raw hand strength, but discounted because
    # calling ranges are skewed strong.
    strength = HAND_STRENGTH.get(hand_class, 0.30)
    return max(0.20, 0.20 + strength * 0.40)


# ============================================================
# Branch enumeration
# ============================================================

def _hero_position_label(state: GameState) -> str:
    try:
        return normalize_position(state.hero_position)
    except ValueError:
        return "BTN"


def _villain_position_for_seat(state: GameState, seat_idx: int) -> str:
    """Best-effort label for an opponent's position.

    We don't know the table button rotation, so this is a coarse heuristic:
    seats closer to hero_index ± 1 (in modular order) are treated as the
    blinds; we use 'BB' as the default for the rest. This affects only
    call_probability shading.
    """
    n = len(state.table_stacks)
    if n == 0:
        return "BB"
    hero_idx = state.hero_index
    rel = (seat_idx - hero_idx) % n
    if rel == 1:
        return "SB"
    if rel == 2:
        return "BB"
    return "BTN"


def _dead_money_bb(state: GameState) -> float:
    """Antes + blinds already in the pot before the action reaches hero."""
    return max(0.0, state.pot_bb)


def compute_action_evs(state: GameState) -> Dict[str, float]:
    """Return per-action ICM equity for FOLD / OPEN / SHOVE.

    All outputs are in the same units as solver.icm_diagnostics()['icm_current'].
    Order is canonical (FOLD/OPEN/SHOVE) but the returned dict uses lowercase
    keys to distinguish from action_weights.
    """
    if not state.table_stacks or not state.payouts:
        return {"ev_fold": 0.0, "ev_open": 0.0, "ev_shove": 0.0}

    stacks = [t.stack_bb for t in state.table_stacks]
    n = len(stacks)
    hero_idx = state.hero_index if 0 <= state.hero_index < n else 0
    hero_stack = stacks[hero_idx]
    payouts = list(state.payouts)
    hand_class = normalize_hand(state.hero_hand)
    equity = hand_vs_range_equity(hand_class)
    dead = _dead_money_bb(state)

    # ------------------------------------------------------------
    # EV_fold: hero keeps current stack.
    # ------------------------------------------------------------
    ev_current = malmuth_harville(stacks, payouts)[hero_idx]
    ev_fold = ev_current

    # ------------------------------------------------------------
    # EV_shove: enumerate (all-fold) + single-caller branches.
    # ------------------------------------------------------------
    opp_indices = [i for i in range(n) if i != hero_idx and stacks[i] > 0]
    call_probs: List[float] = []
    for i in opp_indices:
        cp = _call_probability(
            hero_stack_bb=hero_stack,
            villain_stack_bb=stacks[i],
            villain_position=_villain_position_for_seat(state, i),
            pot_bb=state.pot_bb,
        )
        call_probs.append(cp)

    # P(everyone folds) = prod(1 - p_i)
    p_all_fold = 1.0
    for cp in call_probs:
        p_all_fold *= (1.0 - cp)

    # All-fold branch: hero picks up dead money.
    stacks_allfold = list(stacks)
    stacks_allfold[hero_idx] = hero_stack + dead
    ev_allfold = malmuth_harville(stacks_allfold, payouts)[hero_idx]

    # Single-caller-equivalent branches. Multi-way calls are approximated as
    # the call-probability-weighted average over single-caller branches; this
    # avoids leaving (1 - p_all_fold - sum p_single) of probability mass
    # unassigned, which would systematically underweight ev_shove.
    weighted_caller_ev = 0.0
    total_call_weight = 0.0
    for k, i in enumerate(opp_indices):
        if call_probs[k] <= 0:
            continue
        committed = min(hero_stack, stacks[i])
        stacks_win = list(stacks)
        stacks_win[hero_idx] = hero_stack + committed + dead
        stacks_win[i] = stacks[i] - committed
        ev_win = malmuth_harville(stacks_win, payouts)[hero_idx]

        stacks_lose = list(stacks)
        stacks_lose[hero_idx] = max(0.0, hero_stack - committed)
        stacks_lose[i] = stacks[i] + committed + dead
        ev_lose = malmuth_harville(stacks_lose, payouts)[hero_idx]

        ev_j = equity * ev_win + (1.0 - equity) * ev_lose
        weighted_caller_ev += call_probs[k] * ev_j
        total_call_weight += call_probs[k]

    ev_when_called = (
        weighted_caller_ev / total_call_weight if total_call_weight > 0 else ev_current
    )
    p_any_call = 1.0 - p_all_fold

    ev_shove = p_all_fold * ev_allfold + p_any_call * ev_when_called

    # ------------------------------------------------------------
    # EV_open: simplified three-branch model.
    #   - all fold behind: hero gains dead money
    #   - 3bet (hero folds): hero loses open_size_bb
    #   - called: chip-neutral lump (postflop in v3+/v4)
    # Probabilities depend on hero position (earlier → more players behind)
    # and effective stack depth.
    # ------------------------------------------------------------
    open_size = state.open_size_bb if state.open_size_bb is not None else 2.0
    p_3bet = 0.08 + 0.04 * (1.0 - hand_strength_score(hand_class))  # weak hands face more 3bets
    p_all_fold_open = 0.55 - 0.05 * sum(1 for s in stacks if 0 < s <= 12)  # short stacks behind shove more
    p_all_fold_open = max(0.15, min(0.75, p_all_fold_open))
    p_called = max(0.0, 1.0 - p_all_fold_open - p_3bet)

    stacks_open_steal = list(stacks)
    stacks_open_steal[hero_idx] = hero_stack + dead
    ev_open_steal = malmuth_harville(stacks_open_steal, payouts)[hero_idx]

    stacks_open_3bet_fold = list(stacks)
    stacks_open_3bet_fold[hero_idx] = max(0.0, hero_stack - open_size)
    ev_open_3bet = malmuth_harville(stacks_open_3bet_fold, payouts)[hero_idx]

    # Called branch: assume chip-neutral, equity ~= 0.5 over a 2*open_size pot
    # → blend ev_current with a small +/- swing weighted by hand strength.
    strength_swing = (hand_strength_score(hand_class) - 0.50) * open_size * 0.6
    stacks_open_called_avg = list(stacks)
    stacks_open_called_avg[hero_idx] = hero_stack + strength_swing
    ev_open_called = malmuth_harville(stacks_open_called_avg, payouts)[hero_idx]

    ev_open = (
        p_all_fold_open * ev_open_steal
        + p_3bet * ev_open_3bet
        + p_called * ev_open_called
    )

    return {
        "ev_fold": round(ev_fold, 6),
        "ev_open": round(ev_open, 6),
        "ev_shove": round(ev_shove, 6),
    }
