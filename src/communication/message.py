from dataclasses import dataclass, field, asdict
from typing import List
import json
import time


@dataclass
class StructuredMessage:
    role: str
    intent: str
    reasoning: List[str]
    action: str
    confidence: float
    content: str
    round_id: int = 0
    timestamp: float = field(default_factory=time.time)
    token_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "StructuredMessage":
        return cls(**data)

    def get_reasoning_text(self) -> str:
        return "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(self.reasoning))

    def disagrees_with(self, other: "StructuredMessage") -> bool:
        AGREE_TOKENS = {"agree", "correct", "valid", "right", "accurate", "true", "sound"}
        DISAGREE_TOKENS = {"disagree", "incorrect", "invalid", "wrong", "false", "error", "mistake"}

        self_action = self.action.strip().lower()
        other_action = other.action.strip().lower()

        if self_action in {"yes", "no"} and other_action in {"yes", "no"}:
            return self_action != other_action

        other_agrees = any(t in other_action for t in AGREE_TOKENS)
        other_disagrees = any(t in other_action for t in DISAGREE_TOKENS)

        if other_agrees and not other_disagrees:
            return False
        if other_disagrees and not other_agrees:
            return True

        return other.confidence < 0.6