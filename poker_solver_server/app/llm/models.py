from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from ..models import GameState, SolveResponse


class ChatRequest(BaseModel):
    message: str


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


class ChatResponse(BaseModel):
    message_id: str
    answer: str
    parsed_state: Optional[GameState] = None
    solver_data: Optional[SolveResponse] = None
    missing_fields: List[str] = Field(default_factory=list)
    usage: UsageInfo


class MessageRecord(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    parsed_state_dict: Optional[dict] = None


class SessionRecord(BaseModel):
    session_id: str
    messages: List[MessageRecord] = Field(default_factory=list)
    last_parsed_state_dict: Optional[dict] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SessionCreateResponse(BaseModel):
    session_id: str
    created_at: datetime
