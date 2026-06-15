"""The ``research`` Step, exercised offline through the harness.

These tests drive the real Step code (the Agent Contract's ``run``) through the
harness runner with the **mock tool handler**, so the world is held fixed.
The live model loop is stubbed to keep the tests deterministic and key-free,
while still proving trajectory shape and that briefing material lands in final
state (ADR 0002).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from tooleval import EntryPointRegistry, MockToolHandler, Task, run_task

from prompts import build_research_prompt
from steps.research import ENTRY_POINT_NAME, ResearchStep

COMPANY = "Acme Corp"
SEARCH_QUERY = "Acme Corp recent news 2025 2026"
COMPANY_URL = "https://acme.example.com"

SEARCH_RESULT = (
    "**Acme Corp raises $50M Series B**\nhttps://news.example.com/acme\n"
    "Acme announced a $50M round led by Benchmark."
)
PAGE_TEXT = (
    "# Acme Corp\n"
    "We build AI-powered data pipelines. Founded 2020. HQ: San Francisco."
)
BRIEFING_TEXT = (
    "Acme Corp builds AI-powered data pipelines.\n\n"
    "Recent news: $50M Series B led by Benchmark.\n"
    "Culture: fast-moving, remote-friendly engineering team."
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


def _search_fetch_then_stop() -> _FakeClient:
    """A model that searches, fetches a page, then produces a briefing."""
    return _FakeClient(
        [
            _response(
                [_tool_use("t1", "web_search", {"query": SEARCH_QUERY})],
                "tool_use",
            ),
            _response(
                [_tool_use("t2", "fetch_webpage", {"url": COMPANY_URL})],
                "tool_use",
            ),
            _response([_text(BRIEFING_TEXT)], "end_turn"),
        ]
    )


def _build_step(client: _FakeClient) -> ResearchStep:
    return ResearchStep(client=client)


def _research_task() -> Task:
    return Task(
        id="research-1",
        description=build_research_prompt(COMPANY),
        entry_point=ENTRY_POINT_NAME,
        toolset=["web_search", "fetch_webpage"],
        required_tool_calls=["web_search"],
        minimal_necessary_calls=2,
    )


def _mock_handler() -> MockToolHandler:
    return MockToolHandler.build(
        [
            ("web_search", {"query": SEARCH_QUERY}, SEARCH_RESULT),
            ("fetch_webpage", {"url": COMPANY_URL}, PAGE_TEXT),
        ]
    )


def test_research_runs_through_harness_and_yields_company_briefing():
    step = _build_step(_search_fetch_then_stop())
    registry = EntryPointRegistry()
    registry.register(ENTRY_POINT_NAME, step.run)

    result = run_task(_research_task(), _mock_handler(), registry=registry)

    assert [s.tool_name for s in result.trajectory] == ["web_search", "fetch_webpage"]
    assert result.trajectory[0].result == SEARCH_RESULT
    assert result.trajectory[1].result == PAGE_TEXT

    assert result.final_state["company_briefing"] == BRIEFING_TEXT
    assert len(result.final_state["sources"]) == 2
    assert result.final_state["sources"][0]["tool"] == "web_search"
    assert result.final_state["sources"][1]["tool"] == "fetch_webpage"


def test_research_only_offers_web_search_and_fetch_webpage_to_the_model():
    client = _search_fetch_then_stop()
    step = _build_step(client)

    step.run(_research_task(), _mock_handler())

    offered = {t["name"] for call in client.messages.calls for t in call["tools"]}
    assert offered == {"web_search", "fetch_webpage"}
    assert "read_resume" not in offered
    assert "save_output" not in offered


def test_research_with_no_tool_calls_yields_empty_sources():
    client = _FakeClient([_response([_text(BRIEFING_TEXT)], "end_turn")])
    step = _build_step(client)

    result = step.run(_research_task(), _mock_handler())

    assert result.trajectory == []
    assert result.final_state["company_briefing"] == BRIEFING_TEXT
    assert result.final_state["sources"] == []


def test_research_sources_reflect_full_trajectory():
    step = _build_step(_search_fetch_then_stop())
    result = step.run(_research_task(), _mock_handler())

    src0 = result.final_state["sources"][0]
    assert src0["tool"] == "web_search"
    assert src0["args"] == {"query": SEARCH_QUERY}
    assert src0["result"] == SEARCH_RESULT

    src1 = result.final_state["sources"][1]
    assert src1["tool"] == "fetch_webpage"
    assert src1["args"] == {"url": COMPANY_URL}
    assert src1["result"] == PAGE_TEXT


def test_unknown_tool_name_is_rejected():
    class BadStep(ResearchStep):
        tool_names = ("web_search", "nonexistent_tool")

    with pytest.raises(ValueError, match="unknown tools"):
        _ = BadStep(client=_search_fetch_then_stop()).tools
