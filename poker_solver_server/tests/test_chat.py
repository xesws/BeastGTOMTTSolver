from __future__ import annotations

import json
import pytest
from datetime import datetime, timedelta
from typing import List
from fastapi.testclient import TestClient

from app.main import app
from app.llm.sessions import SessionStore
from app.llm.models import MessageRecord
from app.llm.parser import parse_natural_language
from app.llm.orchestrator import build_gamestate


# A fully-specified parser output that passes the strict gate.
FULL_PARSER_DICT = {
    "game_format": "mtt_icm",
    "tournament_stage": "near_bubble",
    "street": "preflop",
    "hero_hand": "AQo",
    "hero_position": "UTG+1",
    "action_to_hero": "unopened",
    "hero_stack_bb": 14.0,
    "effective_stack_bb": 14.0,
    "pot_bb": 1.5,
    "open_size_bb": 2.0,
    "ante_bb": 0.1,
    "players_left": 9,
    "paid_places": 9,
    "payouts": [40, 25, 15, 10, 5, 3, 1, 0.5, 0.5],
    "table_stacks": [
        {"seat": 0, "stack_bb": 14},
        {"seat": 1, "stack_bb": 20},
        {"seat": 2, "stack_bb": 15},
        {"seat": 3, "stack_bb": 22},
        {"seat": 4, "stack_bb": 18},
        {"seat": 5, "stack_bb": 16},
        {"seat": 6, "stack_bb": 12},
        {"seat": 7, "stack_bb": 25},
        {"seat": 8, "stack_bb": 20},
    ],
    "hero_index": 1,
}


class MockOpenRouterClient:
    def __init__(self, parser_dict: dict | None = None, explainer_str: str | None = None):
        self.parser_dict = parser_dict if parser_dict is not None else dict(FULL_PARSER_DICT)
        self.explainer_str = explainer_str or "建议 MIXED：在 14BB UTG+1 ..."
        self.calls: List[tuple] = []
        self.model = "google/gemini-flash-3.5"

    def chat_completion(self, messages: List[dict], response_format: dict | None = None) -> dict:
        self.calls.append((messages, response_format))
        if response_format and response_format.get("type") == "json_object":
            content = json.dumps(self.parser_dict)
            prompt_tokens = 10
            completion_tokens = 20
        else:
            content = self.explainer_str
            prompt_tokens = 15
            completion_tokens = 25

        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": content
                }
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            },
            "model": self.model
        }


# ---------------------------------------------------------------------------
# parser tests — parser is now a pure NL→dict translator with mechanical
# derivations only; gating happens in the orchestrator.
# ---------------------------------------------------------------------------

def test_parser_returns_merged_state_with_empty_missing() -> None:
    mock_client = MockOpenRouterClient(parser_dict={
        "hero_hand": "AQo",
        "hero_position": "UTG+1",
        "effective_stack_bb": 14.0,
        "tournament_stage": "near_bubble",
    })

    merged, missing, usage = parse_natural_language(
        message="我 UTG+1 AQo 14BB near bubble 怎么打",
        history=[],
        prior_state_dict=None,
        client=mock_client,  # type: ignore[arg-type]
    )

    assert merged["hero_hand"] == "AQo"
    assert merged["hero_position"] == "UTG+1"
    assert merged["effective_stack_bb"] == 14.0
    # Mirror derivation:
    assert merged["hero_stack_bb"] == 14.0
    # Parser no longer computes missing_fields — the orchestrator does.
    assert missing == []
    assert usage["prompt_tokens"] == 10


def test_parser_followup_inherits_state() -> None:
    prior = {
        "hero_hand": "AQo",
        "hero_position": "UTG+1",
        "effective_stack_bb": 14.0,
        "hero_stack_bb": 14.0,
        "tournament_stage": "near_bubble",
        "action_to_hero": "unopened",
        "game_format": "mtt_icm",
    }
    mock_client = MockOpenRouterClient(parser_dict={
        "effective_stack_bb": 10.0,
        "hero_stack_bb": 10.0,
    })

    merged, missing, _ = parse_natural_language(
        message="那如果是 10BB 呢?",
        history=[],
        prior_state_dict=prior,
        client=mock_client,  # type: ignore[arg-type]
    )

    assert merged["hero_hand"] == "AQo"
    assert merged["hero_position"] == "UTG+1"
    assert merged["effective_stack_bb"] == 10.0
    assert merged["hero_stack_bb"] == 10.0
    # The prior unrelated fields are preserved.
    assert merged["tournament_stage"] == "near_bubble"
    assert missing == []


# ---------------------------------------------------------------------------
# strict-gate tests — orchestrator never injects silent defaults.
# ---------------------------------------------------------------------------

def test_strict_gate_blocks_when_core_fields_missing() -> None:
    """A typical short query parses to only 3 core fields → solver must NOT run."""
    state, missing = build_gamestate({
        "hero_hand": "AQo",
        "hero_position": "UTG+1",
        "effective_stack_bb": 14.0,
        "hero_stack_bb": 14.0,
    })
    assert state is None
    # All the ICM/format/stage/pot fields must be reported as missing.
    for f in [
        "game_format", "tournament_stage", "action_to_hero",
        "pot_bb", "open_size_bb", "ante_bb",
        "players_left", "paid_places",
        "payouts", "table_stacks", "hero_index",
    ]:
        assert f in missing, f"expected {f} in missing_fields, got {missing}"


def test_strict_gate_blocks_when_payouts_and_table_missing() -> None:
    merged = dict(FULL_PARSER_DICT)
    merged["payouts"] = None
    merged["table_stacks"] = None
    merged["hero_index"] = None
    merged["paid_places"] = None
    state, missing = build_gamestate(merged)
    assert state is None
    assert "payouts" in missing
    assert "table_stacks" in missing
    assert "hero_index" in missing
    assert "paid_places" in missing


def test_strict_gate_blocks_when_action_to_hero_missing() -> None:
    merged = dict(FULL_PARSER_DICT)
    merged["action_to_hero"] = None
    state, missing = build_gamestate(merged)
    assert state is None
    assert missing == ["action_to_hero"]


def test_derivation_paid_places_from_payouts() -> None:
    """Mechanical derivation: paid_places = len(payouts) when payouts present."""
    merged = dict(FULL_PARSER_DICT)
    merged["paid_places"] = None  # user did not specify
    state, missing = build_gamestate(merged)
    assert state is not None, f"expected derivation to succeed, got missing={missing}"
    assert state.paid_places == len(FULL_PARSER_DICT["payouts"])


def test_derivation_hero_index_from_position_and_table_stacks() -> None:
    """hero_index derives from hero_position + table_stacks via POSITION_SEAT."""
    merged = dict(FULL_PARSER_DICT)
    merged["hero_position"] = "BTN"
    merged["hero_index"] = None
    state, missing = build_gamestate(merged)
    assert state is not None, f"expected derivation to succeed, got missing={missing}"
    # BTN is index 6 in GRID_POSITIONS = ["UTG","UTG+1","UTG+2","LJ","HJ","CO","BTN","SB","BB"].
    assert state.hero_index == 6


def test_derivation_stack_mirror() -> None:
    """If only one of hero_stack_bb / effective_stack_bb is provided, mirror it."""
    merged = dict(FULL_PARSER_DICT)
    merged["effective_stack_bb"] = None  # only hero_stack_bb is given
    state, missing = build_gamestate(merged)
    assert state is not None, f"expected derivation to succeed, got missing={missing}"
    assert state.effective_stack_bb == state.hero_stack_bb == 14.0


def test_strict_gate_allows_when_fully_specified() -> None:
    state, missing = build_gamestate(dict(FULL_PARSER_DICT))
    assert missing == []
    assert state is not None
    assert state.hero_hand == "AQo"
    assert state.tournament_stage == "near_bubble"
    assert len(state.table_stacks) == 9


def test_no_silent_table_synthesis_in_orchestrator() -> None:
    """Regression guard: orchestrator must never fabricate table_stacks / payouts."""
    import inspect
    from app.llm import orchestrator as o
    src = inspect.getsource(o)
    # The previous implementation pulled defaults from _LAYOUTS or 100/0 HU payouts.
    assert "_LAYOUTS" not in src
    assert "[100.0, 0.0]" not in src
    assert "[100.0,0.0]" not in src


# ---------------------------------------------------------------------------
# session tests (unchanged behavior).
# ---------------------------------------------------------------------------

def test_session_create_and_inspect() -> None:
    store = SessionStore(ttl_sec=10, max_history=3)
    record = store.create_session()

    assert record.session_id is not None
    assert not record.messages

    msg = MessageRecord(role="user", content="hello")
    store.add_message(record.session_id, msg)

    retrieved = store.get_session(record.session_id)
    assert len(retrieved.messages) == 1
    assert retrieved.messages[0].content == "hello"

    # Truncation check
    store.add_message(record.session_id, MessageRecord(role="assistant", content="r1"))
    store.add_message(record.session_id, MessageRecord(role="user", content="m2"))
    store.add_message(record.session_id, MessageRecord(role="assistant", content="r2"))

    retrieved = store.get_session(record.session_id)
    assert len(retrieved.messages) == 3
    assert retrieved.messages[0].content == "r1"
    assert retrieved.messages[2].content == "r2"

    store.delete_session(record.session_id)
    with pytest.raises(KeyError):
        store.get_session(record.session_id)


def test_session_ttl_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    store = SessionStore(ttl_sec=10)
    record = store.create_session()

    assert store.get_session(record.session_id) is not None

    future_time = datetime.utcnow() + timedelta(seconds=15)

    class MockDatetime:
        @classmethod
        def utcnow(cls) -> datetime:
            return future_time

    monkeypatch.setattr("app.llm.sessions.datetime", MockDatetime)

    with pytest.raises(KeyError):
        store.get_session(record.session_id)


# ---------------------------------------------------------------------------
# Endpoint integration tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_openrouter(monkeypatch: pytest.MonkeyPatch) -> MockOpenRouterClient:
    client = MockOpenRouterClient()
    monkeypatch.setattr("app.llm.orchestrator.OpenRouterClient", lambda *args, **kwargs: client)
    monkeypatch.setattr("app.llm.parser.OpenRouterClient", lambda *args, **kwargs: client)
    monkeypatch.setattr("app.llm.explainer.OpenRouterClient", lambda *args, **kwargs: client)
    return client


def test_chat_endpoint_full_loop(mock_openrouter: MockOpenRouterClient) -> None:
    client = TestClient(app)

    resp = client.post("/v1/chat/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    session_id = data["session_id"]

    msg_resp = client.post(
        f"/v1/chat/sessions/{session_id}/messages",
        json={"message": "我 UTG+1 AQo 14BB near bubble 怎么打 (with full ICM table data)"}
    )
    assert msg_resp.status_code == 200
    chat_data = msg_resp.json()
    assert chat_data["message_id"] is not None
    assert chat_data["answer"] == mock_openrouter.explainer_str
    assert chat_data["parsed_state"]["hero_hand"] == "AQo"
    assert chat_data["parsed_state"]["hero_position"] == "UTG+1"
    assert chat_data["parsed_state"]["effective_stack_bb"] == 14.0
    assert chat_data["solver_data"]["recommendation"] in (
        "FOLD", "OPEN", "SHOVE", "CALL", "3BET", "MIXED",
    )
    assert not chat_data["missing_fields"]
    assert chat_data["usage"]["prompt_tokens"] == 25  # 10 parser + 15 explainer

    get_resp = client.get(f"/v1/chat/sessions/{session_id}")
    assert get_resp.status_code == 200
    session_data = get_resp.json()
    assert len(session_data["messages"]) == 2
    assert session_data["last_parsed_state_dict"]["hero_hand"] == "AQo"

    del_resp = client.delete(f"/v1/chat/sessions/{session_id}")
    assert del_resp.status_code == 200
    assert del_resp.json() == {"status": "deleted"}

    get_resp_del = client.get(f"/v1/chat/sessions/{session_id}")
    assert get_resp_del.status_code == 404


def test_chat_endpoint_underspecified_routes_to_clarification(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sparse query like the original Example 1 must NOT reach the solver."""
    sparse_dict = {
        "hero_hand": "AQo",
        "hero_position": "UTG+1",
        "effective_stack_bb": 14.0,
        "hero_stack_bb": 14.0,
        "tournament_stage": "near_bubble",
        "game_format": "mtt_icm",
        "action_to_hero": "unopened",
    }
    mock_client = MockOpenRouterClient(parser_dict=sparse_dict, explainer_str="请告诉我 payouts ...")
    monkeypatch.setattr("app.llm.orchestrator.OpenRouterClient", lambda *args, **kwargs: mock_client)
    monkeypatch.setattr("app.llm.parser.OpenRouterClient", lambda *args, **kwargs: mock_client)
    monkeypatch.setattr("app.llm.explainer.OpenRouterClient", lambda *args, **kwargs: mock_client)

    client = TestClient(app)
    session_id = client.post("/v1/chat/sessions").json()["session_id"]

    msg_resp = client.post(
        f"/v1/chat/sessions/{session_id}/messages",
        json={"message": "我 UTG+1 AQo 14BB near bubble 怎么打"}
    )
    assert msg_resp.status_code == 200
    chat_data = msg_resp.json()
    assert chat_data["solver_data"] is None
    assert chat_data["parsed_state"] is None
    for f in ("pot_bb", "open_size_bb", "ante_bb",
              "players_left", "paid_places",
              "payouts", "table_stacks", "hero_index"):
        assert f in chat_data["missing_fields"], (
            f"expected {f} in missing_fields, got {chat_data['missing_fields']}"
        )


def test_chat_endpoint_session_404() -> None:
    client = TestClient(app)
    bad_id = "non-existent-session-id"

    assert client.get(f"/v1/chat/sessions/{bad_id}").status_code == 404
    assert client.post(
        f"/v1/chat/sessions/{bad_id}/messages",
        json={"message": "hello"}
    ).status_code == 404
    assert client.delete(f"/v1/chat/sessions/{bad_id}").status_code == 404


def test_solver_recommendation_surfaces_in_answer(mock_openrouter: MockOpenRouterClient) -> None:
    client = TestClient(app)

    resp = client.post("/v1/chat/sessions")
    session_id = resp.json()["session_id"]

    # Fully-specified AA on BTN with 10BB at the bubble.
    mock_openrouter.parser_dict = {
        "game_format": "mtt_icm",
        "tournament_stage": "bubble",
        "street": "preflop",
        "hero_hand": "AA",
        "hero_position": "BTN",
        "action_to_hero": "unopened",
        "hero_stack_bb": 10.0,
        "effective_stack_bb": 10.0,
        "pot_bb": 1.5,
        "open_size_bb": 2.0,
        "ante_bb": 0.1,
        "players_left": 9,
        "paid_places": 8,
        "payouts": [40, 25, 15, 10, 5, 3, 1, 1],
        "table_stacks": [
            {"seat": 0, "stack_bb": 18},
            {"seat": 1, "stack_bb": 22},
            {"seat": 2, "stack_bb": 14},
            {"seat": 3, "stack_bb": 20},
            {"seat": 4, "stack_bb": 16},
            {"seat": 5, "stack_bb": 12},
            {"seat": 6, "stack_bb": 10},
            {"seat": 7, "stack_bb": 25},
            {"seat": 8, "stack_bb": 20},
        ],
        "hero_index": 6,
    }

    msg_resp = client.post(
        f"/v1/chat/sessions/{session_id}/messages",
        json={"message": "BTN AA 10BB bubble — full ICM table provided"}
    )
    assert msg_resp.status_code == 200
    chat_data = msg_resp.json()

    assert chat_data["solver_data"] is not None
    assert chat_data["solver_data"]["recommendation"] in ("OPEN", "SHOVE", "MIXED")
    # AA on BTN at 10BB still strongly favors entering the pot.
    weights = chat_data["solver_data"]["action_weights"]
    assert (weights.get("SHOVE", 0) + weights.get("OPEN", 0)) > 0.6
    assert weights.get("FOLD", 0) < 0.2
