"""Heads-up preflop push/fold CFR.

Game model (a deliberate simplification of real preflop holdem):
- HU (2 players). SB acts first.
- Both post a 1-BB ante. Effective stack = `stack` BB each (after ante).
- SB decision: SHOVE (push all `stack`) or FOLD.
- If SB FOLD: SB loses ante 1, BB wins 1.
- If SB SHOVE → BB decision: CALL (match shove) or FOLD.
  - If BB FOLD: SB wins ante 1.
  - If BB CALL: showdown for pot 2*stack + 2.

Hand abstraction: each hand class is collapsed to one of `NUM_BUCKETS`
strength buckets using `solver.hand_strength_score`. Showdown is
deterministic by bucket comparison (higher bucket wins; ties split).
This is the canonical "Nash push/fold chart" formulation used as a
training-wheels solver in tournament study tools.

Reachable info sets: NUM_BUCKETS × 2 roles = 20. Trains in milliseconds.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..solver import HAND_STRENGTH, hand_strength_score, normalize_hand

NUM_BUCKETS: int = 10
DEFAULT_STACK_BB: float = 10.0

SB_ACTIONS: List[str] = ["SHOVE", "FOLD"]
BB_ACTIONS: List[str] = ["CALL", "FOLD"]


# ============================================================
# Hand bucketing
# ============================================================

def hand_to_bucket(hand_strength: float) -> int:
    """Map a 0-1 hand strength to an integer bucket in [0, NUM_BUCKETS)."""
    return min(NUM_BUCKETS - 1, max(0, int(hand_strength * NUM_BUCKETS)))


def hand_class_to_bucket(hand_class: str) -> int:
    return hand_to_bucket(hand_strength_score(hand_class))


def all_hand_classes_by_bucket() -> Dict[int, List[str]]:
    """Group all known hand classes (from solver.HAND_STRENGTH) into buckets."""
    out: Dict[int, List[str]] = {b: [] for b in range(NUM_BUCKETS)}
    for cls, strength in HAND_STRENGTH.items():
        out[hand_to_bucket(strength)].append(cls)
    return out


# ============================================================
# Showdown
# ============================================================

def showdown_chip_swing(sb_bucket: int, bb_bucket: int, stack: float) -> float:
    """SB's net chip change at showdown (BB perspective is the negative)."""
    if sb_bucket > bb_bucket:
        return stack
    if sb_bucket < bb_bucket:
        return -stack
    return 0.0  # ties split


# ============================================================
# CFR
# ============================================================

class PushFoldCFR:
    """Vanilla CFR over a 1-step push/fold tree, traversed exhaustively each iter."""

    def __init__(self, stack: float = DEFAULT_STACK_BB) -> None:
        self.stack = stack
        self.iterations_trained = 0
        # regrets and cumulative strategies per (role, bucket)
        self.regret_sb: List[Dict[str, float]] = [
            {a: 0.0 for a in SB_ACTIONS} for _ in range(NUM_BUCKETS)
        ]
        self.regret_bb: List[Dict[str, float]] = [
            {a: 0.0 for a in BB_ACTIONS} for _ in range(NUM_BUCKETS)
        ]
        self.strategy_sum_sb: List[Dict[str, float]] = [
            {a: 0.0 for a in SB_ACTIONS} for _ in range(NUM_BUCKETS)
        ]
        self.strategy_sum_bb: List[Dict[str, float]] = [
            {a: 0.0 for a in BB_ACTIONS} for _ in range(NUM_BUCKETS)
        ]

    # ------------------------------------------------------------
    # Regret matching
    # ------------------------------------------------------------

    @staticmethod
    def _regret_match(regret: Dict[str, float], actions: List[str]) -> Dict[str, float]:
        positive = {a: max(0.0, regret[a]) for a in actions}
        total = sum(positive.values())
        if total > 0:
            return {a: positive[a] / total for a in actions}
        return {a: 1.0 / len(actions) for a in actions}

    def current_sb_strategy(self, bucket: int) -> Dict[str, float]:
        return self._regret_match(self.regret_sb[bucket], SB_ACTIONS)

    def current_bb_strategy(self, bucket: int) -> Dict[str, float]:
        return self._regret_match(self.regret_bb[bucket], BB_ACTIONS)

    def average_sb_strategy(self, bucket: int) -> Dict[str, float]:
        s = self.strategy_sum_sb[bucket]
        total = sum(s.values())
        if total > 0:
            return {a: v / total for a, v in s.items()}
        return {a: 1.0 / len(SB_ACTIONS) for a in SB_ACTIONS}

    def average_bb_strategy(self, bucket: int) -> Dict[str, float]:
        s = self.strategy_sum_bb[bucket]
        total = sum(s.values())
        if total > 0:
            return {a: v / total for a, v in s.items()}
        return {a: 1.0 / len(BB_ACTIONS) for a in BB_ACTIONS}

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    def iteration(self) -> None:
        """One CFR pass over all NUM_BUCKETS x NUM_BUCKETS hand combinations."""
        # Snapshot strategies for both players (computed once per iter for stability).
        sb_strategies = [self.current_sb_strategy(b) for b in range(NUM_BUCKETS)]
        bb_strategies = [self.current_bb_strategy(b) for b in range(NUM_BUCKETS)]

        # Accumulate strategy sums (own reach prob is uniform 1/N here).
        chance = 1.0 / NUM_BUCKETS
        for b in range(NUM_BUCKETS):
            for a in SB_ACTIONS:
                self.strategy_sum_sb[b][a] += chance * sb_strategies[b][a]
            for a in BB_ACTIONS:
                self.strategy_sum_bb[b][a] += chance * bb_strategies[b][a]

        pair_chance = 1.0 / (NUM_BUCKETS * NUM_BUCKETS)
        for sb_b in range(NUM_BUCKETS):
            sb_strat = sb_strategies[sb_b]
            for bb_b in range(NUM_BUCKETS):
                bb_strat = bb_strategies[bb_b]

                # SB perspective
                ev_sb_fold = -1.0
                ev_sb_shove = (
                    bb_strat["CALL"] * showdown_chip_swing(sb_b, bb_b, self.stack)
                    + bb_strat["FOLD"] * 1.0
                )
                ev_sb_node = sb_strat["SHOVE"] * ev_sb_shove + sb_strat["FOLD"] * ev_sb_fold

                # SB regret update (counterfactual reach for SB = opponent's reach = 1)
                self.regret_sb[sb_b]["SHOVE"] += pair_chance * (ev_sb_shove - ev_sb_node)
                self.regret_sb[sb_b]["FOLD"] += pair_chance * (ev_sb_fold - ev_sb_node)

                # BB perspective — only reached if SB shoves
                bb_reach = sb_strat["SHOVE"]
                ev_bb_fold = -1.0
                ev_bb_call = -showdown_chip_swing(sb_b, bb_b, self.stack)
                ev_bb_node = bb_strat["CALL"] * ev_bb_call + bb_strat["FOLD"] * ev_bb_fold

                # BB regret update weighted by BB's counterfactual reach
                # (which equals SB's reach prob to this node = sb_reach)
                self.regret_bb[bb_b]["CALL"] += pair_chance * bb_reach * (ev_bb_call - ev_bb_node)
                self.regret_bb[bb_b]["FOLD"] += pair_chance * bb_reach * (ev_bb_fold - ev_bb_node)

        self.iterations_trained += 1

    def train(self, iterations: int) -> None:
        for _ in range(iterations):
            self.iteration()

    # ------------------------------------------------------------
    # Exploitability — measures distance from Nash
    # ------------------------------------------------------------

    def exploitability(self) -> float:
        """Return the sum of best-response gains over equilibrium for both players.

        For HU zero-sum games, true exploitability ≥ 0; a Nash strategy has 0
        exploitability. The metric returned here is an approximation that
        treats each side's average strategy as fixed and computes the best-
        response gain for the other side, summed.
        """
        sb_avg = [self.average_sb_strategy(b) for b in range(NUM_BUCKETS)]
        bb_avg = [self.average_bb_strategy(b) for b in range(NUM_BUCKETS)]

        pair_chance = 1.0 / (NUM_BUCKETS * NUM_BUCKETS)

        # Best response for SB given fixed BB avg strategy
        sb_br_gain = 0.0
        for sb_b in range(NUM_BUCKETS):
            # Compute EV per SB action given uniform BB distribution
            bucket_chance = 1.0 / NUM_BUCKETS
            ev_shove = 0.0
            ev_fold = -1.0
            for bb_b in range(NUM_BUCKETS):
                ev_shove += bucket_chance * (
                    bb_avg[bb_b]["CALL"] * showdown_chip_swing(sb_b, bb_b, self.stack)
                    + bb_avg[bb_b]["FOLD"] * 1.0
                )
            ev_best = max(ev_shove, ev_fold)
            ev_avg = sb_avg[sb_b]["SHOVE"] * ev_shove + sb_avg[sb_b]["FOLD"] * ev_fold
            sb_br_gain += (1.0 / NUM_BUCKETS) * (ev_best - ev_avg)

        # Best response for BB given fixed SB avg strategy
        bb_br_gain = 0.0
        for bb_b in range(NUM_BUCKETS):
            bucket_chance = 1.0 / NUM_BUCKETS
            ev_call = 0.0
            ev_fold = 0.0
            for sb_b in range(NUM_BUCKETS):
                # Probability of reaching BB's decision = sb shove prob.
                reach = sb_avg[sb_b]["SHOVE"]
                ev_call += bucket_chance * reach * (-showdown_chip_swing(sb_b, bb_b, self.stack))
                ev_fold += bucket_chance * reach * (-1.0)
            ev_best = max(ev_call, ev_fold)
            ev_avg = bb_avg[bb_b]["CALL"] * ev_call + bb_avg[bb_b]["FOLD"] * ev_fold
            bb_br_gain += (1.0 / NUM_BUCKETS) * (ev_best - ev_avg)

        return sb_br_gain + bb_br_gain

    # ------------------------------------------------------------
    # Blueprint export
    # ------------------------------------------------------------

    def export_blueprint(self) -> Dict[str, object]:
        return {
            "_schema": 1,
            "model": "pushfold_v1",
            "stack_bb": self.stack,
            "num_buckets": NUM_BUCKETS,
            "iterations_trained": self.iterations_trained,
            "sb_strategy_by_bucket": [
                self.average_sb_strategy(b) for b in range(NUM_BUCKETS)
            ],
            "bb_strategy_by_bucket": [
                self.average_bb_strategy(b) for b in range(NUM_BUCKETS)
            ],
            "hand_classes_by_bucket": all_hand_classes_by_bucket(),
        }
