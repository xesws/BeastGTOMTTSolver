from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


GameFormat = Literal["mtt_icm", "mtt_chip_ev", "cash"]
TournamentStage = Literal[
    "early",
    "mid",
    "near_bubble",
    "bubble",
    "itm",
    "ft_bubble",
    "ft",
    "heads_up",
]
Street = Literal["preflop", "flop", "turn", "river"]
ActionToHero = Literal["unopened", "limped", "open", "3bet", "4bet", "jam"]
Recommendation = Literal["FOLD", "OPEN", "SHOVE", "CALL", "3BET", "MIXED"]


class TableStackEntry(BaseModel):
    seat: int = Field(..., ge=0, le=9)
    stack_bb: float = Field(..., ge=0)


class GameState(BaseModel):
    game_format: GameFormat
    tournament_stage: TournamentStage
    street: Street
    hero_hand: str = Field(..., description="Two cards like 'AhQd' or canonical class like 'AQo'")
    hero_position: str
    action_to_hero: ActionToHero
    hero_stack_bb: float = Field(..., ge=0)
    effective_stack_bb: float = Field(..., ge=0)
    pot_bb: float = Field(default=0.0, ge=0)
    open_size_bb: Optional[float] = Field(default=None, ge=0)
    ante_bb: Optional[float] = Field(default=None, ge=0)
    players_left: int = Field(..., ge=2)
    paid_places: int = Field(..., ge=1)
    payouts: List[float] = Field(default_factory=list)
    hero_index: int = Field(..., ge=0, le=9)
    table_stacks: List[TableStackEntry] = Field(default_factory=list)
    max_latency_ms: Optional[int] = Field(default=None, ge=1)


class SolveResponse(BaseModel):
    recommendation: Recommendation
    action_weights: Dict[str, float]
    confidence: float = Field(..., ge=0, le=1)
    recommended_size_bb: Optional[float] = None
    reason: str
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class ApproxEVResponse(BaseModel):
    """v3 fast-tier response: approximate per-action ICM EVs."""
    ev_fold: float
    ev_open: float
    ev_shove: float
    confidence: float = Field(..., ge=0, le=1)


class LookupResponse(BaseModel):
    """v1 range-table response — pure dict lookup, no solver work at request time."""
    recommendation: Recommendation
    action_weights: Dict[str, float]
    ev_fold: float
    ev_open: float
    ev_shove: float
    confidence: float = Field(..., ge=0, le=1)
    source: str = "range_table_v1"
    matched_stage: Optional[str] = None
    matched_bucket: Optional[int] = None
    miss_reason: Optional[str] = None


class CFRResponse(BaseModel):
    """v4 CFR push/fold blueprint response.

    Returned as a probability distribution over the role's actions
    (SB → SHOVE/FOLD, BB → CALL/FOLD).
    """
    action_probs: Dict[str, float]
    role: Optional[str] = None  # "SB" or "BB"
    matched_stack_bb: Optional[float] = None
    matched_bucket: Optional[int] = None
    iterations_trained: int = 0
    exploitability: float = 0.0
    source: str = "cfr_pushfold_v1"
    hand_class: Optional[str] = None
    miss_reason: Optional[str] = None
