"""The ``extract`` Step — the first focused Step carved from the monolith.

Given a job-posting URL, ``extract`` fetches the posting and returns structured
job details (role, company, requirements, tech stack, …). Its toolset is
restricted to ``fetch_webpage`` alone: it has no access to ``web_search``,
``read_resume``, or ``save_output``.

The former ``extract_job_details`` LLM "tool" is folded into this Step's own
internal cognition (ADR 0002): it is a live LLM call the Step makes itself in
:meth:`finalize`, not a member of the tool handler and never serviced by it.
The model the Step drives only ever sees ``fetch_webpage``; structuring the
fetched text happens after the loop returns.

Importing this module registers ``extract`` as a named entry point so the
harness can select and run it through the Agent Contract.
"""

from __future__ import annotations

from typing import Any, Callable

from tooleval import Task, TrajectoryStep, register_entry_point

from prompts import EXTRACT_SYSTEM_PROMPT
from steps.base import Step
from tools import extract_job_details

ENTRY_POINT_NAME = "job_search.extract"

# The cognition the Step folds in: raw posting text -> structured job details.
Extractor = Callable[[str], dict[str, Any]]


class ExtractStep(Step):
    """Fetch a posting and extract structured job details from it."""

    name = "extract"
    system_prompt = EXTRACT_SYSTEM_PROMPT
    tool_names = ("fetch_webpage",)

    def __init__(
        self, *, extractor: Extractor = extract_job_details, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        # Injectable so the Step can be exercised without a live model in tests;
        # the default is the real live-cognition call.
        self._extract = extractor

    def finalize(
        self,
        task: Task,
        trajectory: list[TrajectoryStep],
        final_response: str,
    ) -> dict[str, Any]:
        posting = _last_fetched_posting(trajectory)
        if posting is None:
            return {
                "job_details": None,
                "error": "extract: no posting was fetched",
            }
        # Internal cognition — always live, never the tool handler (ADR 0002).
        details = self._extract(posting)
        return {"job_details": details, "source_text": posting}


def _last_fetched_posting(trajectory: list[TrajectoryStep]) -> str | None:
    """The text of the most recent ``fetch_webpage`` call, if any."""
    for step in reversed(trajectory):
        if step.tool_name == "fetch_webpage":
            return step.result
    return None


# The registered entry point uses the default live extractor. Tests build their
# own ExtractStep instances (with a fake client / extractor) and register them
# in a throwaway registry, so this global registration stays live-only.
extract_step = ExtractStep()
register_entry_point(ENTRY_POINT_NAME)(extract_step.run)
