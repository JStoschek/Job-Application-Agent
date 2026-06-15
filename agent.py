"""Job Application Research Agent — the orchestrated Pipeline entry point.

The monolithic tool-use loop is gone. The Pipeline is now an **orchestrator that
composes the focused Steps** — extract → research → analyze → synthesize —
threading each Step's output into the next (ADR 0001). It runs the *same Step
code* the single-step Tasks run; there is no parallel implementation here.

State threading is the design payoff:

- ``extract`` fetches the posting and returns structured ``job_details``;
- ``research`` takes the company (from ``job_details``) and returns a briefing;
- ``analyze`` takes the resume path and ``job_details`` and returns matches/gaps;
- ``synthesize`` takes all of the above, generates bullets + interview prep,
  compiles the report, and writes it with ``save_output``.

The Pipeline honours the Agent Contract (``run(task, tool_handler) ->
AgentResult``): it merges the Steps' trajectories into one ordered trajectory,
sums their token usage, and surfaces both the saved report and the threaded
intermediate state as its final state. ``extract_job_details`` is no longer a
tool — it is the ``extract`` Step's internal cognition (ADR 0002) — so the
Pipeline needs no special-case handler.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from tooleval import (
    AgentResult,
    Task,
    ToolHandler,
    TrajectoryStep,
    register_entry_point,
)

from prompts import (
    build_analyze_prompt,
    build_extract_prompt,
    build_research_prompt,
    build_synthesize_prompt,
    parse_pipeline_inputs,
)
from steps import analyze_step, extract_step, research_step, synthesize_step
from steps.base import Step

ENTRY_POINT_NAME = "job_search.pipeline"


class Pipeline:
    """Compose the focused Steps into the full application-package pipeline.

    The Steps are injectable so the orchestration can be exercised offline with
    stubbed Steps; the registered Pipeline uses the live Step singletons — the
    very same objects the single-step entry points expose.
    """

    def __init__(
        self,
        *,
        extract: Step = extract_step,
        research: Step = research_step,
        analyze: Step = analyze_step,
        synthesize: Step = synthesize_step,
    ) -> None:
        self.extract = extract
        self.research = research
        self.analyze = analyze
        self.synthesize = synthesize

    def run(self, task: Task, tool_handler: ToolHandler) -> AgentResult:
        """Run the four Steps in order, threading state between them.

        The same injected ``tool_handler`` services every Step's world-boundary
        tools, so the harness's mock (or live) world is shared across the whole
        run and the merged trajectory reads as one continuous run.
        """
        inputs = parse_pipeline_inputs(task.description)
        job_url = inputs.get("job_url", "")
        resume_path = inputs.get("resume_path", "")
        candidate_name = inputs.get("candidate_name")
        today = inputs.get("today") or date.today().isoformat()

        extract_result = self.extract.run(
            self._subtask(task, self.extract, build_extract_prompt(job_url)),
            tool_handler,
        )
        job_details = (extract_result.final_state or {}).get("job_details") or {}
        company = job_details.get("company") or ""

        research_result = self.research.run(
            self._subtask(
                task, self.research, build_research_prompt(company, job_details)
            ),
            tool_handler,
        )
        company_briefing = research_result.final_state.get("company_briefing", "")

        analyze_result = self.analyze.run(
            self._subtask(
                task, self.analyze, build_analyze_prompt(resume_path, job_details)
            ),
            tool_handler,
        )
        analysis = analyze_result.final_state.get("analysis", "")

        synthesize_result = self.synthesize.run(
            self._subtask(
                task,
                self.synthesize,
                build_synthesize_prompt(
                    job_details, company_briefing, analysis, candidate_name, today
                ),
            ),
            tool_handler,
        )

        steps = [extract_result, research_result, analyze_result, synthesize_result]

        # The Pipeline's final state is the saved report (so existing readers of
        # filename/report/saved_to keep working) plus the threaded intermediate
        # state, which makes the state flow between Steps inspectable.
        final_state: dict[str, Any] = dict(synthesize_result.final_state)
        final_state.update(
            {
                "job_details": job_details,
                "company_briefing": company_briefing,
                "analysis": analysis,
            }
        )

        return AgentResult(
            task_id=task.id,
            trajectory=_merge_trajectories(steps),
            final_response=synthesize_result.final_response,
            final_state=final_state,
            token_usage=_sum_token_usage(steps),
        )

    @staticmethod
    def _subtask(parent: Task, step: Step, description: str) -> Task:
        """A focused Task for one Step, inheriting the parent's identity.

        The Step's restricted toolset is recorded on the sub-task so the call is
        self-describing; the Step itself enforces the restriction at the seam.
        """
        return Task(
            id=f"{parent.id}:{step.name}",
            description=description,
            entry_point=f"job_search.{step.name}",
            toolset=list(step.tool_names),
        )


def _merge_trajectories(results: list[AgentResult]) -> list[TrajectoryStep]:
    """Concatenate the Steps' trajectories into one, re-numbering ``order``."""
    merged: list[TrajectoryStep] = []
    for result in results:
        for step in result.trajectory:
            merged.append(
                TrajectoryStep(
                    order=len(merged),
                    tool_name=step.tool_name,
                    args=step.args,
                    result=step.result,
                    timestamp=step.timestamp,
                )
            )
    return merged


def _sum_token_usage(results: list[AgentResult]) -> dict[str, int]:
    """Sum the per-Step token usage into the Pipeline's total."""
    total: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
    for result in results:
        for key, value in result.token_usage.items():
            total[key] = total.get(key, 0) + value
    return total


# The registered Pipeline composes the live Step singletons — the same Step code
# the single-step entry points run. Tests build their own Pipeline with stubbed
# Steps and a throwaway registry, so this global registration stays live-only.
pipeline = Pipeline()
register_entry_point(ENTRY_POINT_NAME)(pipeline.run)
