# Job Application Research Agent

A command-line AI agent that takes a job posting URL and your resume, then autonomously researches the company and role to produce a personalized **Application Package** — saved as a markdown file in `output/`.

## What It Does

Given a job URL and your resume, the agent:

1. Fetches and parses the job posting
2. Extracts structured details (role, requirements, tech stack, seniority)
3. Searches the web for recent company news, culture, and background
4. Reads your resume and identifies skill matches and gaps
5. Generates 5–8 tailored resume bullet suggestions (STAR format)
6. Generates 10 likely interview questions with suggested answers
7. Saves everything as a single markdown file in `output/`

## Installation

```bash
pip install -r requirements.txt
```

## Setup

Copy the example env file and add your Anthropic API key:

```bash
cp .env.example .env
```

Edit `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Get a key at [console.anthropic.com](https://console.anthropic.com).

## Usage

```bash
# Without name
python main.py --job-url "https://jobs.example.com/some-role" --resume resume.pdf

# With name (included in the report header)
python main.py --job-url "https://jobs.example.com/some-role" --resume resume.pdf --name "Your Name"
```

**Arguments:**

| Flag | Required | Description |
|------|----------|-------------|
| `--job-url` | Yes | URL of the job posting |
| `--resume` | Yes | Path to your resume (`.pdf` or `.txt`) |
| `--name` | No | Your full name (optional, used in the report header) |

## Output

Reports are saved to `output/` as `{Company}-{Role}-{Date}.md`, for example:

```
output/Stripe-Software-Engineer-2026-03-31.md
```

Each report contains:

```
# Application Package: Software Engineer at Stripe
Candidate: Jane Smith | Generated: 2026-03-31

## Company Briefing
## Job Requirements Analysis
  ### Strong Matches
  ### Gaps to Address
  ### Tech Stack Overview
## Tailored Resume Bullets
## Interview Prep (10 Q&A)
```

## How the Agentic Loop Works

The agent uses the [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) with a manual tool-use loop — no framework, just the raw API.

1. An initial user message is sent to `claude-sonnet-4-6` with 5 tools available
2. Claude decides which tool to call and returns a `tool_use` block
3. The agent executes the tool locally and appends a `tool_result` to the conversation
4. This repeats until Claude returns `stop_reason: "end_turn"` — meaning it's done
5. A safety limit of 20 iterations prevents runaway loops

All tool calls and results are printed to the terminal in real time using [Rich](https://github.com/Textualize/rich).

## Known Limitations

- **Paywalled job boards**: Some sites (LinkedIn, Workday) require login. The agent will proceed with whatever text it can fetch and note the limitation.
- **Image-only PDFs**: Scanned resume PDFs with no embedded text cannot be parsed. Use a text-based PDF or `.txt` instead.
- **DuckDuckGo rate limits**: The search tool uses DuckDuckGo's unofficial API, which may occasionally throttle requests. The agent retries automatically.
