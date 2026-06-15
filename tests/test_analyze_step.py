"""The ``analyze`` Step, exercised offline through the harness.

These tests drive the real Step code (the Agent Contract's ``run``) through
the harness runner with the **mock tool handler**, so the world is held fixed.
The live model loop is stubbed to keep the tests deterministic and key-free,
while still proving trajectory shape and that analysis lands in final state
(ADR 0002).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from tooleval import EntryPointRegistry, MockToolHandler, Task, run_task

from prompts import build_analyze_prompt
from steps.analyze import ENTRY_POINT_NAME, AnalyzeStep

RESUME_PATH = "/fixtures/resume.pdf"
RESUME_TEXT = (
    "Jane Doe\n"
    "Software Engineer with 6 years of Python and AWS experience.\n"
    "Worked on ML pipelines at Startup Inc."
)
JOB_DETAILS = {
    "role": "Senior Software Engineer",
    "company": "Acme",
    "tech_stack": ["Python", "AWS"],
    "requirements": ["5+ years experience"],
    "nice_to_have": ["TensorFlow", "PyTorch"],
}
ANALYSIS_TEXT = (
    "### Strong Matches\n"
    "- 6 years of Python experience meets the 5+ year requirement.\n"
    "- AWS experience matches the required tech stack.\n\n"
    "### Gaps to Address\n"
    "- No mention of TensorFlow or PyTorch, which are listed as nice-to-have."
)


def _usage() -> SimpleNamespace:
    return SimpleNamespace(input_tokens=10, output_tokens=5)


def _response(content: list, stop_reason: str) -> SimpleNamespace:
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=_usage())


def _tool_use(block_id: str, name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=args)


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


class _FakeMessages:
    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses: list) -> None:
        self.messages = _FakeMessages(responses)


def _read_then_stop() -> _FakeClient:
    """A model that reads the resume once, then produces analysis."""
    return _FakeClient(
        [
            _response(
                [_tool_use("t1", "read_resume", {"filepath": RESUME_PATH})],
                "tool_use",
            ),
            _response([_text(ANALYSIS_TEXT)], "end_turn"),
        ]
    )


def _build_step(client: _FakeClient) -> AnalyzeStep:
    return AnalyzeStep(client=client)


def _analyze_task() -> Task:
    return Task(
        id="analyze-1",
        description=build_analyze_prompt(RESUME_PATH, JOB_DETAILS),
        entry_point=ENTRY_POINT_NAME,
        toolset=["read_resume"],
        required_tool_calls=["read_resume"],
        minimal_necessary_calls=1,
    )


def _mock_handler() -> MockToolHandler:
    return MockToolHandler.build(
        [("read_resume", {"filepath": RESUME_PATH}, RESUME_TEXT)]
    )


def test_analyze_runs_through_harness_and_yields_matches_and_gaps():
    step = _build_step(_read_then_stop())
    registry = EntryPointRegistry()
    registry.register(ENTRY_POINT_NAME, step.run)

    result = run_task(_analyze_task(), _mock_handler(), registry=registry)

    assert [s.tool_name for s in result.trajectory] == ["read_resume"]
    assert result.trajectory[0].result == RESUME_TEXT

    assert result.final_state["analysis"] == ANALYSIS_TEXT
    assert result.final_state["resume_text"] == RESUME_TEXT


def test_analyze_only_offers_read_resume_to_the_model():
    client = _read_then_stop()
    step = _build_step(client)

    step.run(_analyze_task(), _mock_handler())

    offered = {t["name"] for call in client.messages.calls for t in call["tools"]}
    assert offered == {"read_resume"}
    assert "web_search" not in offered
    assert "fetch_webpage" not in offered
    assert "save_output" not in offered


def test_analyze_with_no_resume_read_yields_none_resume_text():
    client = _FakeClient([_response([_text(ANALYSIS_TEXT)], "end_turn")])
    step = _build_step(client)

    result = step.run(_analyze_task(), _mock_handler())

    assert result.trajectory == []
    assert result.final_state["analysis"] == ANALYSIS_TEXT
    assert result.final_state["resume_text"] is None


def test_unknown_tool_name_is_rejected():
    class BadStep(AnalyzeStep):
        tool_names = ("read_resume", "nonexistent_tool")

    with pytest.raises(ValueError, match="unknown tools"):
        _ = BadStep(client=_read_then_stop()).tools
