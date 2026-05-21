"""Train HU push/fold CFR at multiple stack depths and persist the blueprint.

Run from the poker_solver_server/ directory:

    python -m scripts.train_cfr

Output: app/data/cfr_blueprint.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

# Allow running as a script from poker_solver_server/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cfr.pushfold import PushFoldCFR, all_hand_classes_by_bucket  # noqa: E402

STACK_DEPTHS_BB: List[float] = [5.0, 7.0, 10.0, 15.0, 20.0]
ITERATIONS: int = 10000


def main(out_path: str | None = None) -> Dict[str, Any]:
    blueprints: Dict[str, Dict[str, Any]] = {}
    for stack in STACK_DEPTHS_BB:
        cfr = PushFoldCFR(stack=stack)
        t0 = time.perf_counter()
        cfr.train(ITERATIONS)
        dt = time.perf_counter() - t0
        expl = cfr.exploitability()
        bp = cfr.export_blueprint()
        bp["wall_clock_s"] = round(dt, 3)
        bp["exploitability"] = round(expl, 6)
        blueprints[str(stack)] = bp
        print(
            f"  stack={stack:5.1f}BB  iters={ITERATIONS}  dt={dt:.2f}s  "
            f"exploitability={expl:.5f}"
        )

    out = {
        "_schema": 1,
        "model": "pushfold_v1_multi_stack",
        "iterations": ITERATIONS,
        "stack_depths_bb": STACK_DEPTHS_BB,
        "hand_classes_by_bucket": all_hand_classes_by_bucket(),
        "blueprints": blueprints,
    }

    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "app", "data", "cfr_blueprint.json",
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nWrote blueprint to {out_path} ({size_kb:.1f} KB)")
    return out


if __name__ == "__main__":
    main()
