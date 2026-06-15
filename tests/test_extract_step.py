"""The ``extract`` Step, exercised offline through the harness.

These tests drive the real Step code (the Agent Contract's ``run``) through the
harness runner with the **mock tool handler**, so the world is held fixed. The
live pieces — the model that drives the tool loop and the ``extract_job_details``
cognition — are stubbed so the test is deterministic and needs no API key, while
still proving trajectory shape and that structured details land in final state.
The live path is covered separately and opt-in (ADR 0002).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from tooleval import EntryPointRegistry, MockToolHandler, Task, run_task

from prompts import build_extract_prompt
from steps.extract import ENTRY_POINT_NAME, ExtractStep

JOB_URL = "https://jobs.example.com/senior-swe"
POSTING_TEXT = (
    "# Senior Software Engineer at Acme\n"
    "We use Python and AWS. 5+ years experience required."
)
JOB_DETAILS = {
    "role": "Senior Software Engineer",
    "company": "Acme",
    "tech_stack": ["Python", "AWS"],
    "requirements": ["5+ years experience"],
}


def _usage() -> SimpleNamespace:
    return SimpleNamespace(input_tokens=10, output_tokens=5)


def _response(content: list, stop_reason: str) -> SimpleNamespace:
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=_usage())


def _tool_use(block_id: str, name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=args)


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


class _FakeMessages:
    """Replays a scripted queue of responses and records each create() call."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses: list) -> None:
        self.messages = _FakeMessages(responses)


def _fetch_then_stop() -> _FakeClient:
    """A model that fetches the posting once, then ends its turn."""
    return _FakeClient(
        [
            _response(
                [_tool_use("t1", "fetch_webpage", {"url": JOB_URL})],
                "tool_use",
            ),
            _response([_text("Fetched the posting.")], "end_turn"),
        ]
    )


def _build_step(client: _FakeClient) -> ExtractStep:
    return ExtractStep(client=client, extractor=lambda text: JOB_DETAILS)


def _extract_task() -> Task:
    return Task(
        id="extract-1",
        description=build_extract_prompt(JOB_URL),
        entry_point=ENTRY_POINT_NAME,
        toolset=["fetch_webpage"],
        required_tool_calls=["fetch_webpage"],
        minimal_necessary_calls=1,
    )


def _mock_handler() -> MockToolHandler:
    return MockToolHandler.build([("fetch_webpage", {"url": JOB_URL}, POSTING_TEXT)])


def test_extract_runs_through_harness_and_yields_structured_details():
    step = _build_step(_fetch_then_stop())
    registry = EntryPointRegistry()
    registry.register(ENTRY_POINT_NAME, step.run)

    result = run_task(_extract_task(), _mock_handler(), registry=registry)

    # Trajectory: exactly the one world-boundary fetch, serviced by the mock.
    assert [s.tool_name for s in result.trajectory] == ["fetch_webpage"]
    assert result.trajectory[0].result == POSTING_TEXT

    # The former extract_job_details is internal cognition: it produced the
    # structured details in final state without ever appearing in the trajectory.
    assert result.final_state["job_details"] == JOB_DETAILS
    assert result.final_state["source_text"] == POSTING_TEXT


def test_extract_only_offers_fetch_webpage_to_the_model():
    client = _fetch_then_stop()
    step = _build_step(client)

    step.run(_extract_task(), _mock_handler())

    # Restricted toolset is enforced at the seam: the model is only ever offered
    # fetch_webpage — never web_search, read_resume, or save_output.
    offered = {t["name"] for call in client.messages.calls for t in call["tools"]}
    assert offered == {"fetch_webpage"}


def test_extract_reports_when_nothing_was_fetched():
    # A model that never fetches (straight to end_turn) leaves no posting.
    client = _FakeClient([_response([_text("I have no URL.")], "end_turn")])
    step = _build_step(client)

    result = step.run(_extract_task(), _mock_handler())

    assert result.trajectory == []
    assert result.final_state["job_details"] is None
    assert "error" in result.final_state


def test_extract_cognition_runs_on_the_fetched_text():
    seen: list[str] = []

    def spy_extractor(text: str) -> dict:
        seen.append(text)
        return JOB_DETAILS

    step = ExtractStep(client=_fetch_then_stop(), extractor=spy_extractor)
    step.run(_extract_task(), _mock_handler())

    # Cognition received exactly the fetched posting text, not the URL or prompt.
    assert seen == [POSTING_TEXT]


def test_unknown_tool_name_is_rejected():
    class BadStep(ExtractStep):
        tool_names = ("fetch_webpage", "nonexistent_tool")

    with pytest.raises(ValueError, match="unknown tools"):
        _ = BadStep(client=_fetch_then_stop()).tools
