"""The ``analyze`` Step — reasons over resume and job details.

Given a candidate's resume path and the extracted job details, ``analyze``
reads the resume and reasons over both to identify strong matches and gaps
to address. Its toolset is restricted to ``read_resume`` alone; it has no
access to ``web_search``, ``fetch_webpage``, or ``save_output``.

The model's final text response is the analysis; ``finalize`` packages it
alongside the resume text collected from the trajectory.

Importing this module registers ``analyze`` as a named entry point so the
harness can select and run it through the Agent Contract.
"""

from __future__ import annotations

from typing import Any

from tooleval import Task, TrajectoryStep, register_entry_point

from prompts import ANALYZE_SYSTEM_PROMPT
from steps.base import Step

ENTRY_POINT_NAME = "job_search.analyze"


class AnalyzeStep(Step):
    """Read the resume and reason over it against extracted job details."""

    name = "analyze"
    system_prompt = ANALYZE_SYSTEM_PROMPT
    tool_names = ("read_resume",)

    def finalize(
        self,
        task: Task,
        trajectory: list[TrajectoryStep],
        final_response: str,
    ) -> dict[str, Any]:
        resume_text = _last_resume_read(trajectory)
        return {
            "analysis": final_response,
            "resume_text": resume_text,
        }


def _last_resume_read(trajectory: list[TrajectoryStep]) -> str | None:
    """The text of the most recent ``read_resume`` call, if any."""
    for step in reversed(trajectory):
        if step.tool_name == "read_resume":
            return step.result
    return None


analyze_step = AnalyzeStep()
register_entry_point(ENTRY_POINT_NAME)(analyze_step.run)
