"""Kuhn poker CFR — canonical sanity test for our CFR core.

Kuhn poker: 3-card deck (J=0, Q=1, K=2), each of two players is dealt one
card, both ante 1. P1 acts first.

History encoding:
- ""       → P1 to act (check or bet)
- "p"      → P1 checked; P2 to act
- "b"      → P1 bet; P2 to act
- "pp"     → both checked → showdown, pot=2
- "pb"     → P1 checked, P2 bet; P1 to act
- "pbp"    → P1 folds after P2 bet → P2 wins ante
- "pbb"    → P1 called → showdown, pot=4
- "bp"     → P2 folds after P1 bet → P1 wins ante
- "bb"     → P2 called → showdown, pot=4

Vanilla CFR (Zinkevich et al. 2007) with full game-tree traversal each
iteration. 12 information sets, converges in <1k iterations to a known
Nash family with game value ≈ -1/18 for P1.
"""

from __future__ import annotations

from itertools import permutations
from typing import Dict, List

CARDS: List[int] = [0, 1, 2]  # J, Q, K
ACTIONS: List[str] = ["p", "b"]  # pass/check/fold, bet/call


class KuhnCFR:
    def __init__(self) -> None:
        self.regret: Dict[str, Dict[str, float]] = {}
        self.strategy_sum: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------
    # Game tree
    # ------------------------------------------------------------

    @staticmethod
    def is_terminal(history: str) -> bool:
        if len(history) < 2:
            return False
        last_two = history[-2:]
        return last_two in ("pp", "bb", "bp", "pbp"[-2:])  # any 'p' after 'b' is fold

    @staticmethod
    def terminal_utility(history: str, cards: tuple) -> float:
        """Utility for the player about to act (which never gets to act in terminal)."""
        player = len(history) % 2
        opponent = 1 - player
        if history[-2:] == "pp":
            return 1.0 if cards[player] > cards[opponent] else -1.0
        if history[-2:] == "bb":
            return 2.0 if cards[player] > cards[opponent] else -2.0
        # Last action was 'p' after 'b' → opponent folded; current player wins ante.
        return 1.0

    # ------------------------------------------------------------
    # Strategy / regret matching
    # ------------------------------------------------------------

    def _get_strategy(self, info_set: str) -> Dict[str, float]:
        if info_set not in self.regret:
            self.regret[info_set] = {a: 0.0 for a in ACTIONS}
            self.strategy_sum[info_set] = {a: 0.0 for a in ACTIONS}
        regret = self.regret[info_set]
        positive = {a: max(0.0, regret[a]) for a in ACTIONS}
        total = sum(positive.values())
        if total > 0:
            return {a: positive[a] / total for a in ACTIONS}
        return {a: 1.0 / len(ACTIONS) for a in ACTIONS}

    def average_strategy(self, info_set: str) -> Dict[str, float]:
        s = self.strategy_sum.get(info_set, {a: 0.0 for a in ACTIONS})
        total = sum(s.values())
        if total > 0:
            return {a: v / total for a, v in s.items()}
        return {a: 1.0 / len(ACTIONS) for a in ACTIONS}

    # ------------------------------------------------------------
    # CFR traversal
    # ------------------------------------------------------------

    def _cfr(self, cards: tuple, history: str, p0: float, p1: float) -> float:
        if self.is_terminal(history):
            return self.terminal_utility(history, cards)

        player = len(history) % 2
        info_set = f"{cards[player]}:{history}"
        strategy = self._get_strategy(info_set)

        reach_self = p0 if player == 0 else p1
        for a in ACTIONS:
            self.strategy_sum[info_set][a] += reach_self * strategy[a]

        action_util: Dict[str, float] = {}
        node_util = 0.0
        for a in ACTIONS:
            next_history = history + a
            if player == 0:
                u = -self._cfr(cards, next_history, p0 * strategy[a], p1)
            else:
                u = -self._cfr(cards, next_history, p0, p1 * strategy[a])
            action_util[a] = u
            node_util += strategy[a] * u

        cf_reach = p1 if player == 0 else p0
        for a in ACTIONS:
            self.regret[info_set][a] += cf_reach * (action_util[a] - node_util)

        return node_util

    def train(self, iterations: int) -> float:
        """Run CFR for `iterations`, returning the average game value for P1."""
        util_sum = 0.0
        deals = list(permutations(CARDS, 2))
        for _ in range(iterations):
            for cards in deals:
                util_sum += self._cfr(cards, "", 1.0, 1.0)
        return util_sum / (iterations * len(deals))

    # ------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------

    def info_sets(self) -> List[str]:
        return sorted(self.strategy_sum.keys())
