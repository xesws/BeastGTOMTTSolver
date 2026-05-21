"""v1: Precomputed preflop range table.

Per design doc §9.1. We sweep v2 (`solve_preflop` → ICM Delta) over a
discrete grid of (stage × stack_bucket × position × hand_class) and store
the resulting recommendation, action weights, and per-action EVs in a
JSON file. At request time the v1 endpoint does a pure dict lookup —
no Malmuth-Harville, no scoring, no enumeration — so response latency
is dominated by HTTP overhead, not solver work.

The grid build itself relies on the LRU cache inside
`solver._malmuth_harville_cached` to dedupe MH calls that share table
shapes across hand classes.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .models import GameState
from .solver import normalize_hand, normalize_position, solve_preflop


# ============================================================
# Grid axes
# ============================================================

GRID_STAGES: List[str] = [
    "early", "mid", "near_bubble", "bubble", "itm", "ft_bubble", "ft",
]
GRID_BUCKETS: List[int] = [7, 12, 20, 30, 50]
GRID_POSITIONS: List[str] = [
    "UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB",
]
POSITION_SEAT: Dict[str, int] = {p: i for i, p in enumerate(GRID_POSITIONS)}

_RANKS = "23456789TJQKA"


def all_hand_classes() -> List[str]:
    """All 169 canonical preflop hand classes (13 pairs + 78 suited + 78 offsuit)."""
    classes: List[str] = []
    for r in _RANKS:
        classes.append(r + r)
    for i in range(len(_RANKS)):
        for j in range(i):
            high, low = _RANKS[i], _RANKS[j]
            classes.append(high + low + "s")
            classes.append(high + low + "o")
    return classes


# ============================================================
# Canonical table layouts per stage
# ============================================================
#
# Per stage we fix one representative (table_stacks, payouts). At build
# time we replace the hero's seat with the grid stack-bucket value, so
# the same shape is reused across all (bucket, position) combinations.
# These layouts capture the qualitative ICM shape of each stage; they
# are not meant to be statistically average tournaments.

_LAYOUTS: Dict[str, Dict[str, Any]] = {
    "early": {
        "stacks": [50, 60, 55, 70, 45, 80, 65, 75, 60],
        "payouts": [100, 80, 65, 55, 45, 35, 28, 22, 18],
    },
    "mid": {
        "stacks": [30, 25, 40, 35, 28, 32, 38, 26, 45],
        "payouts": [100, 75, 55, 40, 30, 22, 16, 12, 8],
    },
    "near_bubble": {
        "stacks": [26, 18, 9, 14, 31, 22, 7, 44, 11],
        "payouts": [100, 70, 50, 35, 25, 18, 12, 8, 5],
    },
    "bubble": {
        "stacks": [20, 15, 8, 12, 18, 14, 25, 30, 10],
        "payouts": [100, 70, 50, 35, 25, 18, 12, 8, 5],
    },
    "itm": {
        "stacks": [22, 25, 28, 30, 20, 35, 24, 33, 27],
        "payouts": [100, 70, 50, 35, 25, 18, 12, 8, 5],
    },
    "ft_bubble": {
        "stacks": [15, 18, 12, 22, 25, 16, 28, 14, 20],
        "payouts": [300, 200, 140, 100, 70, 50, 35, 25, 0],
    },
    "ft": {
        "stacks": [25, 20, 30, 18, 35, 22, 28, 15, 32],
        "payouts": [300, 200, 140, 100, 70, 50, 35, 25, 18],
    },
}


# ============================================================
# Build
# ============================================================

def _make_grid_state(stage: str, bucket: int, position: str, hand_class: str) -> GameState:
    layout = _LAYOUTS[stage]
    seat = POSITION_SEAT[position]
    stacks = list(layout["stacks"])
    stacks[seat] = bucket
    return GameState(
        game_format="mtt_icm",
        tournament_stage=stage,  # type: ignore[arg-type]
        street="preflop",
        hero_hand=hand_class,
        hero_position=position,
        action_to_hero="unopened",
        hero_stack_bb=float(bucket),
        effective_stack_bb=float(bucket),
        pot_bb=1.5,
        open_size_bb=2.0,
        ante_bb=0.1,
        players_left=99,
        paid_places=99,
        payouts=layout["payouts"],
        hero_index=seat,
        table_stacks=[{"seat": i, "stack_bb": float(s)} for i, s in enumerate(stacks)],
    )


def build_table(verbose: bool = False) -> Dict[str, Any]:
    """Run the full grid through v2 (`solve_preflop`) and return the table."""
    hand_classes = all_hand_classes()
    table: Dict[str, Any] = {"_schema": 1, "axes": {
        "stages": GRID_STAGES,
        "buckets": GRID_BUCKETS,
        "positions": GRID_POSITIONS,
        "hand_classes": hand_classes,
    }, "cells": {}}
    cells = table["cells"]
    total = 0
    for stage in GRID_STAGES:
        cells[stage] = {}
        for bucket in GRID_BUCKETS:
            cells[stage][str(bucket)] = {}
            for position in GRID_POSITIONS:
                cells[stage][str(bucket)][position] = {}
                for hand_class in hand_classes:
                    state = _make_grid_state(stage, bucket, position, hand_class)
                    resp = solve_preflop(state)
                    d = resp.diagnostics
                    cells[stage][str(bucket)][position][hand_class] = {
                        "recommendation": resp.recommendation,
                        "action_weights": resp.action_weights,
                        "ev_fold": d.get("ev_fold", 0.0),
                        "ev_open": d.get("ev_open", 0.0),
                        "ev_shove": d.get("ev_shove", 0.0),
                    }
                    total += 1
            if verbose:
                print(f"  ...{stage}/{bucket} done")
    if verbose:
        print(f"Built {total} cells.")
    return table


# ============================================================
# Persistence
# ============================================================

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_DATA_PATH = os.path.join(_DATA_DIR, "range_table.json")


def save_table(table: Dict[str, Any], path: str = _DATA_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(table, f, separators=(",", ":"))


def load_table(path: str = _DATA_PATH) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_or_build_table(path: str = _DATA_PATH, *, verbose: bool = False) -> Dict[str, Any]:
    if os.path.exists(path):
        return load_table(path)
    table = build_table(verbose=verbose)
    save_table(table, path)
    return table


# ============================================================
# Lookup
# ============================================================

_TABLE: Optional[Dict[str, Any]] = None


def _ensure_loaded() -> Dict[str, Any]:
    global _TABLE
    if _TABLE is None:
        _TABLE = load_or_build_table()
    return _TABLE


def closest_bucket(stack_bb: float, buckets: List[int] = GRID_BUCKETS) -> int:
    return min(buckets, key=lambda b: abs(b - stack_bb))


def _miss_response(reason: str) -> Dict[str, Any]:
    return {
        "recommendation": "FOLD",
        "action_weights": {"FOLD": 1.0, "OPEN": 0.0, "SHOVE": 0.0},
        "ev_fold": 0.0,
        "ev_open": 0.0,
        "ev_shove": 0.0,
        "confidence": 0.0,
        "source": "range_table_v1",
        "matched_stage": None,
        "matched_bucket": None,
        "miss_reason": reason,
    }


def lookup(state: GameState) -> Dict[str, Any]:
    """Pure dict lookup. Does not invoke Malmuth-Harville."""
    table = _ensure_loaded()
    try:
        position = normalize_position(state.hero_position)
    except ValueError as e:
        return _miss_response(f"unknown position: {e}")
    try:
        hand_class = normalize_hand(state.hero_hand)
    except ValueError as e:
        return _miss_response(f"unknown hand: {e}")

    matched_stage = state.tournament_stage if state.tournament_stage in GRID_STAGES else "near_bubble"
    matched_bucket = closest_bucket(state.effective_stack_bb)

    cell = (
        table["cells"]
        .get(matched_stage, {})
        .get(str(matched_bucket), {})
        .get(position, {})
        .get(hand_class)
    )
    if cell is None:
        return _miss_response(
            f"no cell for stage={matched_stage} bucket={matched_bucket} "
            f"pos={position} hand={hand_class}"
        )

    return {
        "recommendation": cell["recommendation"],
        "action_weights": cell["action_weights"],
        "ev_fold": cell["ev_fold"],
        "ev_open": cell["ev_open"],
        "ev_shove": cell["ev_shove"],
        "confidence": 1.0,
        "source": "range_table_v1",
        "matched_stage": matched_stage,
        "matched_bucket": matched_bucket,
    }
