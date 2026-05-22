from __future__ import annotations

import uuid
import logging
from typing import List, Optional, Tuple
from pydantic import ValidationError

from .client import OpenRouterClient
from .models import ChatResponse, MessageRecord, SessionRecord, UsageInfo
from .parser import parse_natural_language
from .explainer import explain_solve_result, generate_clarification
from .sessions import SessionStore
from ..models import GameState
from ..solver import solve_preflop
from ..range_table import POSITION_SEAT

logger = logging.getLogger(__name__)


REQUIRED_FIELDS: Tuple[str, ...] = (
    "game_format",
    "tournament_stage",
    "hero_hand",
    "hero_position",
    "hero_stack_bb",
    "effective_stack_bb",
    "action_to_hero",
    "pot_bb",
    "open_size_bb",
    "ante_bb",
    "players_left",
    "paid_places",
    "payouts",
    "table_stacks",
    "hero_index",
)


def _apply_derivations(merged: dict) -> dict:
    """Apply mechanically-safe derivations that do not invent information.

    Only derivations from values the user (or a prior turn) explicitly supplied
    are applied. No fabricated table compositions, payout schedules, or stage
    defaults are introduced.
    """
    state = dict(merged)

    # Project-invariant: solver is preflop-only (CLAUDE.md scope).
    state["street"] = "preflop"

    # Definitional mirror between hero stack and effective stack when exactly
    # one is provided.
    if state.get("hero_stack_bb") is None and state.get("effective_stack_bb") is not None:
        state["hero_stack_bb"] = state["effective_stack_bb"]
    elif state.get("effective_stack_bb") is None and state.get("hero_stack_bb") is not None:
        state["effective_stack_bb"] = state["hero_stack_bb"]

    # Paid places is the length of the user-supplied payouts list.
    if state.get("paid_places") is None and state.get("payouts"):
        state["paid_places"] = len(state["payouts"])

    # Hero seat index derived from user-supplied table_stacks + hero_position
    # using the standard 9-max positional convention.
    if (
        state.get("hero_index") is None
        and state.get("table_stacks")
        and state.get("hero_position") in POSITION_SEAT
    ):
        state["hero_index"] = POSITION_SEAT[state["hero_position"]]

    return state


def _is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, list) and len(value) == 0:
        return True
    return False


def build_gamestate(merged: dict) -> Tuple[Optional[GameState], List[str]]:
    """Return (GameState, []) if every required field is present (post-derivation),
    or (None, missing_fields) otherwise.

    Never invents defaults for fields that affect solver math.
    """
    derived = _apply_derivations(merged)

    missing = [f for f in REQUIRED_FIELDS if _is_empty(derived.get(f))]
    if missing:
        return None, missing

    try:
        state = GameState(**{k: derived[k] for k in (
            "game_format", "tournament_stage", "street", "hero_hand",
            "hero_position", "action_to_hero", "hero_stack_bb",
            "effective_stack_bb", "pot_bb", "open_size_bb", "ante_bb",
            "players_left", "paid_places", "payouts", "hero_index",
            "table_stacks",
        ) if k in derived})
    except ValidationError as e:
        bad_fields = sorted({str(err["loc"][0]) for err in e.errors() if err.get("loc")})
        return None, bad_fields or ["__validation_error__"]

    return state, []


def process_chat_message(
    session: SessionRecord,
    store: SessionStore,
    user_msg: str,
    client: Optional[OpenRouterClient] = None,
) -> ChatResponse:
    """Orchestrate the parsing, solving, explaining, and session updates for a user message."""
    if client is None:
        client = OpenRouterClient()

    # 1. Parse the natural language input, passing prior state
    merged_state_dict, _parser_missing, parser_usage = parse_natural_language(
        message=user_msg,
        history=session.messages,
        prior_state_dict=session.last_parsed_state_dict,
        client=client,
    )

    # 2. Strict-gate state assembly. No silent defaults are injected here.
    state, missing_fields = build_gamestate(merged_state_dict)

    # 3. Clarification branch: any required field missing → ask the user.
    if state is None:
        answer, clar_usage = generate_clarification(
            user_msg=user_msg,
            missing_fields=missing_fields,
            client=client,
        )

        user_record = MessageRecord(role="user", content=user_msg)
        assistant_record = MessageRecord(
            role="assistant",
            content=answer,
            parsed_state_dict=merged_state_dict,
        )
        store.add_message(session.session_id, user_record)
        store.add_message(session.session_id, assistant_record)

        combined_usage = UsageInfo(
            prompt_tokens=parser_usage.get("prompt_tokens", 0) + clar_usage.get("prompt_tokens", 0),
            completion_tokens=parser_usage.get("completion_tokens", 0) + clar_usage.get("completion_tokens", 0),
            model=parser_usage.get("model") or clar_usage.get("model") or client.model,
        )

        return ChatResponse(
            message_id=str(uuid.uuid4()),
            answer=answer,
            parsed_state=None,
            solver_data=None,
            missing_fields=missing_fields,
            usage=combined_usage,
        )

    # 4. State fully user-specified → invoke solver.
    solve_res = solve_preflop(state)

    # 5. Explain result.
    answer, explainer_usage = explain_solve_result(
        user_msg=user_msg,
        state=state,
        solve_res=solve_res,
        client=client,
    )

    # 6. Save message history.
    user_record = MessageRecord(role="user", content=user_msg)
    assistant_record = MessageRecord(
        role="assistant",
        content=answer,
        parsed_state_dict=state.model_dump(),
    )
    store.add_message(session.session_id, user_record)
    store.add_message(session.session_id, assistant_record)

    combined_usage = UsageInfo(
        prompt_tokens=parser_usage.get("prompt_tokens", 0) + explainer_usage.get("prompt_tokens", 0),
        completion_tokens=parser_usage.get("completion_tokens", 0) + explainer_usage.get("completion_tokens", 0),
        model=parser_usage.get("model") or explainer_usage.get("model") or client.model,
    )

    return ChatResponse(
        message_id=str(uuid.uuid4()),
        answer=answer,
        parsed_state=state,
        solver_data=solve_res,
        missing_fields=[],
        usage=combined_usage,
    )
