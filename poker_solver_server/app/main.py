from __future__ import annotations

from fastapi import FastAPI

from .approx_ev import predict_action_evs
from .cfr.lookup import lookup as cfr_lookup
from .models import (
    ApproxEVResponse,
    CFRResponse,
    GameState,
    LookupResponse,
    SolveResponse,
)
from .range_table import lookup as range_table_lookup
from .solver import solve_preflop

app = FastAPI(
    title="Poker Solver Server",
    description=(
        "Layered preflop solver:\n"
        "- v0 heuristic + v2 EVs (POST /v1/solve/preflop) — explainable\n"
        "- v1 range table (POST /v1/lookup/preflop) — < 5ms dict lookup, no MH\n"
        "- v3 approximate EV (POST /v1/predict/preflop) — sub-millisecond fast tier\n"
        "- v4 push/fold CFR blueprint (POST /v1/cfr/preflop) — trained Nash strategy\n"
        "Not a GTO solver. For training, study, and product prototyping only."
    ),
    version="0.5.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.post("/v1/solve/preflop", response_model=SolveResponse)
def solve_preflop_endpoint(state: GameState) -> SolveResponse:
    return solve_preflop(state)


@app.post("/v1/lookup/preflop", response_model=LookupResponse)
def lookup_preflop_endpoint(state: GameState) -> LookupResponse:
    """v1 range-table tier: precomputed (stage × bucket × position × hand_class) lookup."""
    return LookupResponse(**range_table_lookup(state))


@app.post("/v1/predict/preflop", response_model=ApproxEVResponse)
def predict_preflop_endpoint(state: GameState) -> ApproxEVResponse:
    """v3 fast tier: approximate per-action ICM EVs, no Malmuth-Harville."""
    return ApproxEVResponse(**predict_action_evs(state))


@app.post("/v1/cfr/preflop", response_model=CFRResponse)
def cfr_preflop_endpoint(state: GameState) -> CFRResponse:
    """v4 push/fold CFR blueprint lookup. HU only (SB/BB), bucket-quantized."""
    return CFRResponse(**cfr_lookup(state))
