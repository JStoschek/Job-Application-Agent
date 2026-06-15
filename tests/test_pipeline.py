"""The orchestrated Pipeline, exercised offline through the harness.

The Pipeline composes the *same* Step code the single-step Tasks run (ADR 0001).
These tests build the four real Steps with stubbed model clients and a single
shared mock tool handler, then drive the Pipeline through the Agent Contract's
``run``. They prove the end-to-end behaviour without a live model or network:

- each Step's output threads into the next Step's prompt (the design payoff);
- the four Steps' trajectories merge into one ordered trajectory ending in the
  observable ``save_output`` write;
- the saved report and the threaded intermediate state surface in final state.
"""

from __future__ import annotations

from types import SimpleNamespace

from tooleval import EntryPointRegistry, MockToolHandler, Task, run_task

from agent import ENTRY_POINT_NAME, Pipeline
from prompts import build_user_prompt
from steps.analyze import AnalyzeStep
from steps.extract import ExtractStep
from steps.research import ResearchStep
from steps.synthesize import SynthesizeStep

JOB_URL = "https://jobs.acme.com/senior-swe"
RESUME_PATH = "/fixtures/resume.pdf"
CANDIDATE = "Jane Doe"

POSTING_TEXT = "# Senior Software Engineer at Acme\nWe use Python and AWS."
JOB_DETAILS = {
    "role": "Senior Software Engineer",
    "company": "Acme",
    "tech_stack": ["Python", "AWS"],
}
SEARCH_QUERY = "Acme company research"
SEARCH_RESULT = "**Acme raises $50M**\nhttps://news.example.com/acme\nSeries B."
BRIEFING = "Acme builds AI data pipelines. $50M Series B led by Benchmark."
RESUME_TEXT = "Jane Doe — 6 years Python and AWS."
ANALYSIS = "### Strong Matches\n- Python\n\n### Gaps to Address\n- TensorFlow"

FILENAME = "Acme-Senior-Software-Engineer-2026-06-15.md"
REPORT = (
    "# Application Package: Senior Software Engineer at Acme\n"
    "**Candidate:** Jane Doe | **Generated:** 2026-06-15\n\n"
    "## Company Briefing\n...\n## Job Requirements Analysis\n...\n"
    "## Tailored Resume Bullets\n...\n## Interview Prep\n..."
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


def _build_pipeline() -> tuple[Pipeline, dict[str, _FakeClient]]:
    """A Pipeline of real Steps driven by scripted fake clients."""
    extract_client = _FakeClient(
        [
            _response([_tool_use("e1", "fetch_webpage", {"url": JOB_URL})], "tool_use"),
            _response([_text("Fetched.")], "end_turn"),
        ]
    )
    research_client = _FakeClient(
        [
            _response([_tool_use("r1", "web_search", {"query": SEARCH_QUERY})], "tool_use"),
            _response([_text(BRIEFING)], "end_turn"),
        ]
    )
    analyze_client = _FakeClient(
        [
            _response([_tool_use("a1", "read_resume", {"filepath": RESUME_PATH})], "tool_use"),
            _response([_text(ANALYSIS)], "end_turn"),
        ]
    )
    synthesize_client = _FakeClient(
        [
            _response(
                [_tool_use("s1", "save_output", {"filename": FILENAME, "content": REPORT})],
                "tool_use",
            ),
            _response([_text("Saved.")], "end_turn"),
        ]
    )
    pipeline = Pipeline(
        extract=ExtractStep(client=extract_client, extractor=lambda text: JOB_DETAILS),
        research=ResearchStep(client=research_client),
        analyze=AnalyzeStep(client=analyze_client),
        synthesize=SynthesizeStep(client=synthesize_client),
    )
    clients = {
        "extract": extract_client,
        "research": research_client,
        "analyze": analyze_client,
        "synthesize": synthesize_client,
    }
    return pipeline, clients


def _mock_handler() -> MockToolHandler:
    return MockToolHandler.build(
        [
            ("fetch_webpage", {"url": JOB_URL}, POSTING_TEXT),
            ("web_search", {"query": SEARCH_QUERY}, SEARCH_RESULT),
            ("read_resume", {"filepath": RESUME_PATH}, RESUME_TEXT),
            ("save_output", {"filename": FILENAME, "content": REPORT}, SAVED_TO),
        ]
    )


def _pipeline_task() -> Task:
    return Task(
        id="pipeline-1",
        description=build_user_prompt(JOB_URL, RESUME_PATH, CANDIDATE),
        entry_point=ENTRY_POINT_NAME,
        toolset=["web_search", "fetch_webpage", "read_resume", "save_output"],
    )


def _first_user_message(client: _FakeClient) -> str:
    return client.messages.calls[0]["messages"][0]["content"]


def test_pipeline_runs_end_to_end_through_the_harness():
    pipeline, _ = _build_pipeline()
    registry = EntryPointRegistry()
    registry.register(ENTRY_POINT_NAME, pipeline.run)

    result = run_task(_pipeline_task(), _mock_handler(), registry=registry)

    # The four Steps' trajectories merge into one continuous, ordered run; the
    # final disk write is observable as the last step.
    assert [s.tool_name for s in result.trajectory] == [
        "fetch_webpage",
        "web_search",
        "read_resume",
        "save_output",
    ]
    assert [s.order for s in result.trajectory] == [0, 1, 2, 3]
    assert result.trajectory[-1].tool_name == "save_output"
    assert result.trajectory[-1].result == SAVED_TO


def test_pipeline_final_state_carries_report_and_threaded_state():
    pipeline, _ = _build_pipeline()

    result = pipeline.run(_pipeline_task(), _mock_handler())

    # The saved report (so existing filename/report/saved_to readers keep working)…
    assert result.final_state["filename"] == FILENAME
    assert result.final_state["report"] == REPORT
    assert result.final_state["saved_to"] == SAVED_TO
    # …plus the intermediate state threaded between Steps.
    assert result.final_state["job_details"] == JOB_DETAILS
    assert result.final_state["company_briefing"] == BRIEFING
    assert result.final_state["analysis"] == ANALYSIS


def test_pipeline_threads_each_steps_output_into_the_next():
    pipeline, clients = _build_pipeline()

    pipeline.run(_pipeline_task(), _mock_handler())

    # research is prompted with the company extract produced.
    assert "Acme" in _first_user_message(clients["research"])
    # analyze is prompted with the extracted job details.
    assert "Senior Software Engineer" in _first_user_message(clients["analyze"])
    # synthesize is prompted with the briefing and the analysis from upstream.
    synth_prompt = _first_user_message(clients["synthesize"])
    assert BRIEFING in synth_prompt
    assert ANALYSIS in synth_prompt
    assert CANDIDATE in synth_prompt


def test_pipeline_sums_token_usage_across_steps():
    pipeline, _ = _build_pipeline()

    result = pipeline.run(_pipeline_task(), _mock_handler())

    # Each Step's stubbed loop made two create() calls at 10 in / 5 out apiece;
    # four Steps → 8 calls → 80 in / 40 out.
    assert result.token_usage == {"input_tokens": 80, "output_tokens": 40}


def test_pipeline_runs_the_same_step_objects_it_was_given():
    pipeline, _ = _build_pipeline()
    assert pipeline.extract.name == "extract"
    assert pipeline.research.name == "research"
    assert pipeline.analyze.name == "analyze"
    assert pipeline.synthesize.name == "synthesize"
