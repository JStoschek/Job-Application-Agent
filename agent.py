"""Job Application Research Agent — Contract-compliant pipeline entry point.

The monolithic tool-use loop is now reached only through the Agent Contract
(``run(task, tool_handler) -> AgentResult``). It accepts an injected tool
handler instead of touching the world directly, captures its full trajectory
and final state into an ``AgentResult``, and registers itself as a named entry
point so the harness can select and run it.

Behaviour is unchanged: passing the live tool handler still produces the
complete application-package report. This slice is purely the seam — the agent
is now observable and evaluable. Decomposition into Steps happens in later
slices (ADR 0001); for now the whole Pipeline is a single registered entry
point.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic
from tooleval import (
    AgentResult,
    RecordingHandler,
    Task,
    ToolHandler,
    register_entry_point,
)

from prompts import MAIN_SYSTEM_PROMPT
from tools import TOOL_DEFINITIONS, extract_job_details

ENTRY_POINT_NAME = "job_search.pipeline"
MODEL = "claude-sonnet-4-6"
MAX_ITERATIONS = 20
MAX_TOKENS = 16000


class _CognitionAwareHandler:
    """Wrap the injected world handler so the agent's own cognition never
    reaches it.

    ``extract_job_details`` is a live LLM call — the agent's mind, not a world
    boundary — so it always runs live and is never mocked (ADR 0002). Every
    other tool is a world-boundary call and is delegated to the injected
    handler, which the harness may point at mocks or real I/O.
    """

    def __init__(self, world_handler: ToolHandler) -> None:
        self._world = world_handler

    def __call__(self, tool_name: str, args: dict[str, Any]) -> Any:
        if tool_name == "extract_job_details":
            return json.dumps(extract_job_details(args["text"]), indent=2)
        return self._world(tool_name, args)


@register_entry_point(ENTRY_POINT_NAME)
def run_pipeline(task: Task, tool_handler: ToolHandler) -> AgentResult:
    """Run the full job-application pipeline once and return an ``AgentResult``.

    The Task's ``description`` is the user prompt (job URL, resume path, name).
    The injected ``tool_handler`` services the world-boundary tools; the
    recording wrapper captures every call as the trajectory.
    """
    client = anthropic.Anthropic()
    handler = RecordingHandler(_CognitionAwareHandler(tool_handler))

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": task.description}
    ]
    input_tokens = 0
    output_tokens = 0
    final_response = ""

    for _ in range(MAX_ITERATIONS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=MAIN_SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )
        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens

        for block in response.content:
            if block.type == "text" and block.text.strip():
                final_response = block.text

        if response.stop_reason != "tool_use":
            break

        # The assistant turn must be appended in full so its tool_use blocks
        # are paired with the tool_result blocks that follow.
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = handler(block.name, block.input)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
            )
        messages.append({"role": "user", "content": tool_results})

    return AgentResult(
        task_id=task.id,
        trajectory=handler.trajectory,
        final_response=final_response,
        final_state=_capture_final_state(handler.trajectory),
        token_usage={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    )


def _capture_final_state(trajectory) -> dict[str, Any]:
    """Snapshot the produced report from the trajectory.

    The report is captured in-memory from the ``save_output`` call so a later
    run that overwrites the output file on disk can never erase what this run
    produced. The runner deep-copies this snapshot before the result leaves it.
    """
    state: dict[str, Any] = {}
    for step in trajectory:
        if step.tool_name == "save_output":
            state = {
                "filename": step.args.get("filename"),
                "report": step.args.get("content", ""),
                "saved_to": step.result,
            }
    return state
