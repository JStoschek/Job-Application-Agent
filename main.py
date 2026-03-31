#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Job Application Research Agent — researches a job posting and generates a personalized application package.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --job-url "https://jobs.acme.com/engineer-123" --resume resume.pdf --name "Jane Smith"
  python main.py --job-url "https://greenhouse.io/acme/jobs/456" --resume /path/to/resume.txt --name "John Doe"
        """,
    )
    parser.add_argument("--job-url", required=True, help="URL of the job posting")
    parser.add_argument(
        "--resume", required=True, help="Path to your resume file (.pdf or .txt)"
    )
    parser.add_argument("--name", required=True, help="Your full name")

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

    from agent import run_agent

    try:
        run_agent(
            job_url=args.job_url,
            resume_path=str(resume_path),
            candidate_name=args.name,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(0)
    except Exception:
        console.print("\n[bold red]Fatal error:[/bold red]")
        console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
