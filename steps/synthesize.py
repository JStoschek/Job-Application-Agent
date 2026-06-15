"""The ``synthesize`` Step — the final Step that compiles and saves the report.

By the time ``synthesize`` runs, the upstream Steps have done the gathering:
the extracted job details, the company briefing, and the resume analysis are
threaded into its instructions. ``synthesize`` does the remaining cognition —
generating tailored resume bullets in STAR format and the interview prep — then
compiles the full application-package report and persists it. Its toolset is
restricted to ``save_output`` alone: it touches the world only to write the
finished report to disk.

The model's ``save_output`` call is the single observable world-boundary write;
``finalize`` snapshots the saved report from the trajectory so a later run can
never erase what this one produced.

Importing this module registers ``synthesize`` as a named entry point so the
harness can select and run it through the Agent Contract.
"""

from __future__ import annotations

from typing import Any

from tooleval import Task, TrajectoryStep, register_entry_point

from prompts import SYNTHESIZE_SYSTEM_PROMPT
from steps.base import Step

ENTRY_POINT_NAME = "job_search.synthesize"


class SynthesizeStep(Step):
    """Generate bullets + interview prep, compile the report, and save it."""

    name = "synthesize"
    system_prompt = SYNTHESIZE_SYSTEM_PROMPT
    tool_names = ("save_output",)

    def finalize(
        self,
        task: Task,
        trajectory: list[TrajectoryStep],
        final_response: str,
    ) -> dict[str, Any]:
        saved = _last_saved_report(trajectory)
        if saved is None:
            return {
                "filename": None,
                "report": "",
                "saved_to": None,
                "error": "synthesize: save_output was never called",
            }
        return saved


def _last_saved_report(trajectory: list[TrajectoryStep]) -> dict[str, Any] | None:
    """Snapshot the report from the most recent ``save_output`` call, if any.

    Captured in-memory from the call args so a later run that overwrites the
    output file on disk can never erase what this run produced.
    """
    for step in reversed(trajectory):
        if step.tool_name == "save_output":
            return {
                "filename": step.args.get("filename"),
                "report": step.args.get("content", ""),
                "saved_to": step.result,
            }
    return None


synthesize_step = SynthesizeStep()
register_entry_point(ENTRY_POINT_NAME)(synthesize_step.run)
