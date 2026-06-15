"""The reusable Step abstraction.

A Step is a focused mini-agent: a restricted tool subset, a focused system
prompt, and a tool-use loop that returns an ``AgentResult`` honoring the Agent
Contract (``run(task, tool_handler) -> AgentResult``). ``extract`` is the first
Step; ``research`` and ``analyze`` follow this same pattern, and the full
Pipeline composes Steps in order (ADR 0001).

A Step composes two kinds of work, kept deliberately distinct (ADR 0002):

- **World-boundary tool calls** — made by the model through the injected tool
  handler inside :meth:`Step._run_loop`. These are the only calls a mock handler
  ever services, and they form the observable trajectory. A Step is restricted
  to a subset of the world tools simply by which tool definitions it exposes to
  the model, so a Step can never call a tool it was not granted.
- **Internal cognition** — live LLM reasoning the Step does itself (e.g. the
  former ``extract_job_details``). This is the agent's mind, never a tool, and
  never routed through the handler; it always runs live. Subclasses fold it into
  :meth:`Step.finalize`.
"""

from __future__ import annotations

from typing import Any

import anthropic
from tooleval import (
    AgentResult,
    RecordingHandler,
    Task,
    ToolHandler,
    TrajectoryStep,
)

from tools import TOOL_DEFINITIONS

MODEL = "claude-sonnet-4-6"
MAX_ITERATIONS = 10
MAX_TOKENS = 8000


class Step:
    """Base class for a focused, independently-runnable Step.

    Subclasses set :attr:`name`, :attr:`system_prompt`, and :attr:`tool_names`
    (its restricted toolset) and may override :meth:`finalize` to fold internal
    cognition into the Step's final state. The bound :meth:`run` method is the
    Agent Contract callable registered as an entry point.
    """

    name: str = ""
    system_prompt: str = ""
    tool_names: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        client: anthropic.Anthropic | None = None,
        model: str = MODEL,
        max_iterations: int = MAX_ITERATIONS,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        # The client is created lazily so importing a Step (and registering its
        # entry point) never requires an API key; only running it does.
        self._client = client
        self.model = model
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    @property
    def tools(self) -> list[dict[str, Any]]:
        """The tool definitions this Step exposes — its restricted toolset.

        Restriction is enforced here: the model is only ever offered these
        tools, so a Step cannot call a world tool outside its subset.
        """
        wanted = set(self.tool_names)
        tools = [t for t in TOOL_DEFINITIONS if t["name"] in wanted]
        missing = wanted - {t["name"] for t in tools}
        if missing:
            raise ValueError(
                f"Step {self.name!r} names unknown tools: {sorted(missing)}"
            )
        return tools

    def run(self, task: Task, tool_handler: ToolHandler) -> AgentResult:
        """Run the Step once and return an ``AgentResult`` (the Agent Contract).

        The injected ``tool_handler`` services the Step's world-boundary tools;
        the recording wrapper captures every call as the trajectory. Internal
        cognition (live) turns the trajectory into the Step's final state.
        """
        handler = RecordingHandler(tool_handler)
        final_response, token_usage = self._run_loop(task, handler)
        final_state = self.finalize(task, handler.trajectory, final_response)
        return AgentResult(
            task_id=task.id,
            trajectory=handler.trajectory,
            final_response=final_response,
            final_state=final_state,
            token_usage=token_usage,
        )

    def finalize(
        self,
        task: Task,
        trajectory: list[TrajectoryStep],
        final_response: str,
    ) -> dict[str, Any]:
        """Internal cognition hook — derive the Step's final state.

        The default Step produces no derived state. Steps that do live reasoning
        of their own (like ``extract``) override this; it always runs live and
        is never routed through the tool handler (ADR 0002).
        """
        return {}

    def _run_loop(
        self, task: Task, handler: RecordingHandler
    ) -> tuple[str, dict[str, int]]:
        """The shared Anthropic tool-use loop, restricted to the Step's tools."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": task.description}
        ]
        input_tokens = 0
        output_tokens = 0
        final_response = ""

        for _ in range(self.max_iterations):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                tools=self.tools,
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

        return final_response, {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
