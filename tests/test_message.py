import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.communication.message import StructuredMessage


def make_msg(role="solver", action="42", confidence=0.9, reasoning=None):
    return StructuredMessage(
        role=role,
        intent="solve",
        reasoning=reasoning or ["step 1", "step 2"],
        action=action,
        confidence=confidence,
        content="some content",
        round_id=0,
    )


def test_to_dict_has_required_keys():
    msg = make_msg()
    d = msg.to_dict()
    for key in ("role", "intent", "reasoning", "action", "confidence", "content", "round_id"):
        assert key in d


def test_to_json_is_valid_json():
    import json
    msg = make_msg()
    parsed = json.loads(msg.to_json())
    assert parsed["action"] == "42"


def test_from_dict_roundtrip():
    msg = make_msg()
    d = msg.to_dict()
    restored = StructuredMessage.from_dict(d)
    assert restored.action == msg.action
    assert restored.confidence == msg.confidence
    assert restored.reasoning == msg.reasoning


def test_get_reasoning_text_format():
    msg = make_msg(reasoning=["add 1 and 2", "result is 3"])
    text = msg.get_reasoning_text()
    assert "Step 1:" in text
    assert "Step 2:" in text
    assert "add 1 and 2" in text


def test_disagrees_with_different_action():
    msg_a = make_msg(action="42", confidence=0.9)
    msg_b = make_msg(action="100", confidence=0.9)
    assert msg_a.disagrees_with(msg_b) is True


def test_agrees_with_same_action():
    msg_a = make_msg(action="42", confidence=0.9)
    msg_b = make_msg(action="42", confidence=0.9)
    assert msg_a.disagrees_with(msg_b) is False


def test_disagrees_when_low_confidence():
    msg_a = make_msg(action="42", confidence=0.3)
    msg_b = make_msg(action="42", confidence=0.9)
    assert msg_a.disagrees_with(msg_b) is True


def test_confidence_clamped_in_parse():
    msg = make_msg(confidence=0.0)
    assert msg.confidence == 0.0
    msg2 = make_msg(confidence=1.0)
    assert msg2.confidence == 1.0