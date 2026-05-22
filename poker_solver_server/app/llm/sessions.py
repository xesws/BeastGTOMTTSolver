from __future__ import annotations

import os
import uuid
import threading
from datetime import datetime
from typing import Dict, Optional

from .models import SessionRecord, MessageRecord

CHAT_SESSION_TTL_SEC = int(os.getenv("CHAT_SESSION_TTL_SEC", "1800"))
CHAT_HISTORY_MAX_MESSAGES = int(os.getenv("CHAT_HISTORY_MAX_MESSAGES", "10"))


class SessionStore:
    def __init__(self, ttl_sec: int = CHAT_SESSION_TTL_SEC, max_history: int = CHAT_HISTORY_MAX_MESSAGES):
        self.ttl_sec = ttl_sec
        self.max_history = max_history
        self._sessions: Dict[str, SessionRecord] = {}
        self._lock = threading.Lock()

    def create_session(self) -> SessionRecord:
        """Create a new session record and store it."""
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()
        record = SessionRecord(
            session_id=session_id,
            messages=[],
            last_parsed_state_dict=None,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._sessions[session_id] = record
        return record

    def get_session(self, session_id: str) -> SessionRecord:
        """Retrieve a session. Raise KeyError if not found or expired."""
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError("Session not found")
            
            record = self._sessions[session_id]
            now = datetime.utcnow()
            elapsed = (now - record.updated_at).total_seconds()
            if elapsed > self.ttl_sec:
                # Lazy eviction
                del self._sessions[session_id]
                raise KeyError("Session expired")
            
            # Touch session
            record.updated_at = now
            return record

    def delete_session(self, session_id: str) -> None:
        """Delete a session by ID. Ignore if not exists."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

    def add_message(self, session_id: str, message: MessageRecord) -> None:
        """Add a message to the session's history and truncate if it exceeds max_history."""
        record = self.get_session(session_id)
        with self._lock:
            record.messages.append(message)
            if len(record.messages) > self.max_history:
                record.messages = record.messages[-self.max_history:]
            
            # Update last parsed state dict if the message had one
            if message.parsed_state_dict is not None:
                record.last_parsed_state_dict = message.parsed_state_dict
            
            record.updated_at = datetime.utcnow()
