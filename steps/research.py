"""The ``research`` Step — gathers company-briefing material.

Given a company name and any extracted job context, ``research`` performs web
searches and page fetches to collect recent news, culture, products/funding,
and team information. Its toolset is restricted to ``web_search`` and
``fetch_webpage``; it has no access to ``read_resume`` or ``save_output``.

The model's final text response is the synthesized company briefing; ``finalize``
packages it alongside the raw sources collected from the trajectory.

Importing this module registers ``research`` as a named entry point so the
harness can select and run it through the Agent Contract.
"""

from __future__ import annotations

from typing import Any

from tooleval import Task, TrajectoryStep, register_entry_point

from prompts import RESEARCH_SYSTEM_PROMPT
from steps.base import Step

ENTRY_POINT_NAME = "job_search.research"


class ResearchStep(Step):
    """Search the web and fetch pages to build a company briefing."""

    name = "research"
    system_prompt = RESEARCH_SYSTEM_PROMPT
    tool_names = ("web_search", "fetch_webpage")

    def finalize(
        self,
        task: Task,
        trajectory: list[TrajectoryStep],
        final_response: str,
    ) -> dict[str, Any]:
        sources = [
            {"tool": s.tool_name, "args": s.args, "result": s.result}
            for s in trajectory
        ]
        return {
            "company_briefing": final_response,
            "sources": sources,
        }


research_step = ResearchStep()
register_entry_point(ENTRY_POINT_NAME)(research_step.run)
