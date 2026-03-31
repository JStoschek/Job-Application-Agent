import anthropic
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from prompts import MAIN_SYSTEM_PROMPT, build_user_prompt
from tools import TOOL_DEFINITIONS, execute_tool

console = Console()
MAX_ITERATIONS = 20


def run_agent(job_url: str, resume_path: str, candidate_name: str) -> None:
    client = anthropic.Anthropic()

    messages = [
        {
            "role": "user",
            "content": build_user_prompt(job_url, resume_path, candidate_name),
        }
    ]

    console.print(
        Panel(
            f"[bold]Job:[/bold] {job_url}\n"
            f"[bold]Resume:[/bold] {resume_path}\n"
            f"[bold]Candidate:[/bold] {candidate_name}",
            title="[bold green]Job Application Agent — Starting Research[/bold green]",
            border_style="green",
        )
    )

    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1
        console.print(f"\n[dim]── Iteration {iteration}/{MAX_ITERATIONS} ──[/dim]")

        with console.status("[bold blue]Thinking...[/bold blue]"):
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=16000,
                system=MAIN_SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )

        # Print any text commentary from Claude
        for block in response.content:
            if block.type == "text" and block.text.strip():
                console.print(
                    Panel(
                        Markdown(block.text),
                        title="[cyan]Claude[/cyan]",
                        border_style="cyan",
                    )
                )

        if response.stop_reason == "end_turn":
            console.print(
                Panel(
                    "[bold green]Research complete. Check the output/ folder for your application package.[/bold green]",
                    border_style="green",
                )
            )
            break

        if response.stop_reason != "tool_use":
            console.print(
                f"[yellow]Unexpected stop reason: {response.stop_reason}[/yellow]"
            )
            break

        # Append full assistant response (must include tool_use blocks)
        messages.append({"role": "assistant", "content": response.content})

        # Execute each tool call and collect results
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            console.print(
                Panel(
                    f"[bold]{block.name}[/bold]\n[dim]{_format_input(block.input)}[/dim]",
                    title="[yellow]Tool Call[/yellow]",
                    border_style="yellow",
                )
            )

            with console.status(f"[bold yellow]Running {block.name}...[/bold yellow]"):
                result = execute_tool(block.name, block.input)

            preview = result[:200] + "..." if len(result) > 200 else result
            console.print(f"[dim green]  ↳ {preview}[/dim green]")

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
            )

        messages.append({"role": "user", "content": tool_results})

    else:
        console.print(
            f"[bold red]Reached max iterations ({MAX_ITERATIONS}) without completing.[/bold red]"
        )


def _format_input(tool_input: dict) -> str:
    parts = []
    for k, v in tool_input.items():
        v_str = str(v)
        if len(v_str) > 120:
            v_str = v_str[:120] + "..."
        parts.append(f"{k}: {v_str}")
    return "\n".join(parts)
