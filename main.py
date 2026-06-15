#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Job Application Research Agent — researches a job posting and generates a personalized application package.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --job-url "https://jobs.acme.com/engineer-123" --resume resume.pdf
  python main.py --job-url "https://jobs.acme.com/engineer-123" --resume resume.pdf --name "Jane Smith"
        """,
    )
    parser.add_argument("--job-url", required=True, help="URL of the job posting")
    parser.add_argument(
        "--resume", required=True, help="Path to your resume file (.pdf or .txt)"
    )
    parser.add_argument("--name", default=None, help="Your full name (optional, used in the report header)")

    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            "[bold red]Error: ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file or export it in your shell.[/bold red]"
        )
        sys.exit(1)

    resume_path = Path(args.resume)
    if not resume_path.exists():
        console.print(f"[bold red]Error: Resume file not found: {args.resume}[/bold red]")
        sys.exit(1)

    if resume_path.suffix.lower() not in {".pdf", ".txt"}:
        console.print(
            f"[bold red]Error: Resume must be a .pdf or .txt file, got: {resume_path.suffix}[/bold red]"
        )
        sys.exit(1)

    # Importing the agent module registers its entry point in the harness
    # registry; we then drive it through the Agent Contract exactly as the
    # eval harness does, but with the live tool handler.
    from agent import ENTRY_POINT_NAME
    from prompts import build_user_prompt
    from tools import live_tool_handler, WORLD_TOOLS
    from tooleval import Task, run_task

    task = Task(
        id="live-run",
        description=build_user_prompt(args.job_url, str(resume_path), args.name),
        entry_point=ENTRY_POINT_NAME,
        toolset=[*WORLD_TOOLS, "extract_job_details"],
    )

    console.print(
        Panel(
            f"[bold]Job:[/bold] {args.job_url}\n"
            f"[bold]Resume:[/bold] {resume_path}\n"
            f"[bold]Candidate:[/bold] {args.name or '—'}",
            title="[bold green]Job Application Agent — Starting Research[/bold green]",
            border_style="green",
        )
    )

    try:
        with console.status("[bold blue]Researching… (live tools)[/bold blue]"):
            result = run_task(task, live_tool_handler)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(0)
    except Exception:
        console.print("\n[bold red]Fatal error:[/bold red]")
        console.print_exception()
        sys.exit(1)

    _report_run(result)


def _report_run(result) -> None:
    calls = ", ".join(step.tool_name for step in result.trajectory) or "none"
    saved_to = result.final_state.get("saved_to")
    usage = result.token_usage

    body = (
        f"[bold]Tool calls ({len(result.trajectory)}):[/bold] {calls}\n"
        f"[bold]Tokens:[/bold] {usage.get('input_tokens', 0)} in / "
        f"{usage.get('output_tokens', 0)} out\n"
        f"[bold]Duration:[/bold] {result.duration_s:.1f}s"
    )
    if saved_to:
        body += f"\n[bold green]{saved_to}[/bold green]"
    else:
        body += "\n[bold yellow]No report was saved (the run did not call save_output).[/bold yellow]"

    console.print(
        Panel(
            body,
            title="[bold green]Research complete[/bold green]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
