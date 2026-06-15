"""The ``synthesize`` Step, exercised offline through the harness.

These tests drive the real Step code (the Agent Contract's ``run``) through the
harness runner with the **mock tool handler**, so the world is held fixed. The
live model loop is stubbed to keep the tests deterministic and key-free, while
still proving the restricted toolset, that the report is saved, and that the
saved report lands in final state (ADR 0002).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from tooleval import EntryPointRegistry, MockToolHandler, Task, run_task

from prompts import build_synthesize_prompt
from steps.synthesize import ENTRY_POINT_NAME, SynthesizeStep

JOB_DETAILS = {
    "role": "Senior Software Engineer",
    "company": "Acme",
    "tech_stack": ["Python", "AWS"],
}
COMPANY_BRIEFING = "Acme builds AI data pipelines. $50M Series B."
ANALYSIS = "### Strong Matches\n- Python\n\n### Gaps to Address\n- TensorFlow"

FILENAME = "Acme-Senior-Software-Engineer-2026-06-15.md"
REPORT = (
    "# Application Package: Senior Software Engineer at Acme\n"
    "**Candidate:** Jane Doe | **Generated:** 2026-06-15\n\n"
    "## Company Briefing\nAcme builds AI data pipelines.\n\n"
    "## Job Requirements Analysis\n### Strong Matches\n- Python\n"
    "### Gaps to Address\n- TensorFlow\n### Tech Stack Overview\n- Python, AWS\n\n"
    "## Tailored Resume Bullets\n- Built X, achieving Y.\n\n"
    "## Interview Prep\n### Top 10 Likely Questions\n**Q1: Why Acme?**\n> Because…"
)
SAVED_TO = f"Saved to output/{FILENAME}"


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


def _compile_then_save() -> _FakeClient:
    """A model that compiles the report and saves it, then ends its turn."""
    return _FakeClient(
        [
            _response(
                [_tool_use("t1", "save_output", {"filename": FILENAME, "content": REPORT})],
                "tool_use",
            ),
            _response([_text("Saved the application package.")], "end_turn"),
        ]
    )


def _build_step(client: _FakeClient) -> SynthesizeStep:
    return SynthesizeStep(client=client)


def _synthesize_task() -> Task:
    return Task(
        id="synthesize-1",
        description=build_synthesize_prompt(
            JOB_DETAILS, COMPANY_BRIEFING, ANALYSIS, "Jane Doe", "2026-06-15"
        ),
        entry_point=ENTRY_POINT_NAME,
        toolset=["save_output"],
        required_tool_calls=["save_output"],
        minimal_necessary_calls=1,
    )


def _mock_handler() -> MockToolHandler:
    return MockToolHandler.build(
        [("save_output", {"filename": FILENAME, "content": REPORT}, SAVED_TO)]
    )


def test_synthesize_runs_through_harness_and_saves_the_report():
    step = _build_step(_compile_then_save())
    registry = EntryPointRegistry()
    registry.register(ENTRY_POINT_NAME, step.run)

    result = run_task(_synthesize_task(), _mock_handler(), registry=registry)

    # The single observable world-boundary write is the save_output call.
    assert [s.tool_name for s in result.trajectory] == ["save_output"]
    assert result.trajectory[0].result == SAVED_TO

    assert result.final_state["filename"] == FILENAME
    assert result.final_state["report"] == REPORT
    assert result.final_state["saved_to"] == SAVED_TO


def test_synthesize_only_offers_save_output_to_the_model():
    client = _compile_then_save()
    step = _build_step(client)

    step.run(_synthesize_task(), _mock_handler())

    offered = {t["name"] for call in client.messages.calls for t in call["tools"]}
    assert offered == {"save_output"}
    assert "web_search" not in offered
    assert "fetch_webpage" not in offered
    assert "read_resume" not in offered


def test_synthesize_reports_when_nothing_was_saved():
    client = _FakeClient([_response([_text("I could not finish.")], "end_turn")])
    step = _build_step(client)

    result = step.run(_synthesize_task(), _mock_handler())

    assert result.trajectory == []
    assert result.final_state["filename"] is None
    assert result.final_state["report"] == ""
    assert "error" in result.final_state


def test_unknown_tool_name_is_rejected():
    class BadStep(SynthesizeStep):
        tool_names = ("save_output", "nonexistent_tool")

    with pytest.raises(ValueError, match="unknown tools"):
        _ = BadStep(client=_compile_then_save()).tools
