# Job Application Research Agent

A command-line AI agent that takes a **job posting URL** and **your resume**, then
autonomously researches the company and role and produces a personalized
**Application Package** — a company briefing, a resume match/gap analysis,
tailored STAR-format resume bullets, and interview prep — saved as a single
markdown file.

It's a genuinely useful tool. But the reason it's worth reading is *how it's
built*: it was deliberately re-architected from a single monolithic tool-use
loop into an **orchestrator that composes focused, independently-testable
Steps**, each honoring a formal **Agent Contract**. That re-architecture is the
point, and the rest of this README is about it.

> Built on the raw [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
> — no agent frameworks. The companion eval harness that scores this agent lives
> at [JStoschek/tool-eval](https://github.com/JStoschek/tool-eval).

---

## The engineering story

The first version was the obvious one: a single `while` loop that handed Claude
five tools and let it run until `end_turn`. It worked, but it was a black box —
you could only ask *"did the whole thing succeed?"* You couldn't isolate which
part of the reasoning was weak, couldn't restrict what each phase was allowed to
touch, and couldn't test any of it without hitting the live network.

So it was rebuilt around two architectural decisions (documented as ADRs in the
[ToolEval repo](https://github.com/JStoschek/tool-eval/tree/main/docs/adr)):

**1. Agent Contract + orchestrated decomposition.** Every runnable unit — each
Step *and* the full Pipeline — implements one interface:

```python
run(task: Task, tool_handler: ToolHandler) -> AgentResult
```

The tool handler is **injected**, not hard-wired. That single seam is what makes
the agent both testable and evaluable: swap in a mock handler that returns canned
fixtures and the entire pipeline runs offline and deterministically; swap in the
live handler and it hits the real web and disk. The Pipeline runs the *same Step
code* the single-Step entry points run — there is no parallel "test
implementation" that could drift from production.

**2. Mock the world, not the mind.** Each Step cleanly separates two kinds of
work:

- **World-boundary tool calls** (`fetch_webpage`, `web_search`, `read_resume`,
  `save_output`) — every side effect goes through the injected handler and is
  recorded into the trajectory. These are the *only* calls a mock ever services.
- **Internal cognition** — the LLM reasoning a Step does itself (e.g. turning raw
  posting text into structured job details). This is the agent's *mind*; it is
  never a tool, never mocked, and always runs live.

The payoff: you can replay an agent's entire run from fixtures and trust that
what you're testing is the real decision-making, not a stubbed copy of it.

---

## Architecture

The Pipeline is an orchestrator that threads state through four focused Steps,
each a mini-agent with a **restricted toolset** — it is only ever offered the
tools it needs, so it *cannot* call anything outside its subset.

```
                       ┌─────────────────────────────────────────────┐
  job URL  ─────────▶  │ extract    tools: fetch_webpage             │
                       │            → structured job_details          │
                       └───────────────────────┬─────────────────────┘
                                                │ job_details
                       ┌────────────────────────▼────────────────────┐
                       │ research   tools: web_search, fetch_webpage  │
                       │            → company_briefing                │
                       └────────────────────────┬────────────────────┘
                                                 │
  resume  ──────────▶  ┌────────────────────────▼────────────────────┐
                       │ analyze    tools: read_resume                │
                       │            → matches & gaps analysis         │
                       └────────────────────────┬────────────────────┘
                                                 │
                       ┌────────────────────────▼────────────────────┐
                       │ synthesize tools: save_output               │
                       │  bullets + interview prep → compiled report  │
                       └────────────────────────┬────────────────────┘
                                                 │
                                                 ▼
                              output/{Company}-{Role}-{Date}.md
```

Each Step's output is threaded into the next: `extract`'s `job_details` feeds
`research` and `analyze`; everything converges in `synthesize`, which generates
the tailored bullets and interview questions, compiles the full report, and is
the *only* Step permitted to write to disk. The Pipeline merges the Steps'
trajectories into one ordered run and sums their token usage, so a full run reads
as a single continuous trajectory.

| File | Responsibility |
| --- | --- |
| [`agent.py`](agent.py) | The `Pipeline` orchestrator — composes Steps, threads state, merges trajectories |
| [`steps/base.py`](steps/base.py) | The reusable `Step` abstraction (restricted toolset + tool-use loop + `finalize` cognition hook) |
| [`steps/extract.py`](steps/extract.py) · [`research.py`](steps/research.py) · [`analyze.py`](steps/analyze.py) · [`synthesize.py`](steps/synthesize.py) | The four focused Steps |
| [`tools.py`](tools.py) | The four world-boundary tools + the live tool handler |
| [`prompts.py`](prompts.py) | Per-Step system prompts and prompt builders |
| [`main.py`](main.py) | CLI entry point — drives the Pipeline through the Contract with the live handler |

Both the Pipeline and every individual Step are registered as named entry points,
so the [ToolEval harness](https://github.com/JStoschek/tool-eval) can select and
score them at any granularity — a single Step in isolation, or the whole
Pipeline end to end.

---

## Quickstart

```bash
# 1. Install the eval harness (sibling repo) + this agent's dependencies
git clone https://github.com/JStoschek/tool-eval.git
git clone https://github.com/JStoschek/job-application-agent.git JobPostingAgent
cd JobPostingAgent
pip install -e ../ToolEval        # provides the Agent Contract types & registry
pip install -r requirements.txt

# 2. Add your Anthropic API key
cp .env.example .env              # then edit .env and set ANTHROPIC_API_KEY

# 3. Run it
python main.py --job-url "https://jobs.example.com/some-role" --resume resume.pdf
```

Get an API key at [console.anthropic.com](https://console.anthropic.com).

### Usage

```bash
# Minimal
python main.py --job-url "<URL>" --resume resume.pdf

# With your name in the report header
python main.py --job-url "<URL>" --resume resume.pdf --name "Jane Smith"
```

| Flag | Required | Description |
| --- | --- | --- |
| `--job-url` | Yes | URL of the job posting |
| `--resume` | Yes | Path to your resume (`.pdf` or `.txt`) |
| `--name` | No | Your full name (used in the report header) |

The run prints its trajectory, token usage, and duration to the terminal via
[Rich](https://github.com/Textualize/rich).

---

## Output

Reports are written to `output/` as `{Company}-{Role}-{Date}.md`, e.g.
`output/Stripe-Software-Engineer-2026-06-15.md`. Each contains:

```
# Application Package: Software Engineer at Stripe
Candidate: Jane Smith | Generated: 2026-06-15

## Company Briefing
## Job Requirements Analysis
  ### Strong Matches
  ### Gaps to Address
  ### Tech Stack Overview
## Tailored Resume Bullets        (5–8, STAR format, quantified)
## Interview Prep                 (10 likely Q&A)
```

See [`output/`](output/) for real generated examples.

---

## Testing

The injected-handler seam means the whole agent runs offline against fixtures —
no API key, no network. Every Step and the full Pipeline have their own tests:

```bash
pytest
```

`tests/test_pipeline.py` builds a `Pipeline` from stubbed Steps and a throwaway
registry to verify the orchestration (state threading, trajectory merging, token
summing) without touching the live model — exactly the testability the
re-architecture was for.

---

## Known limitations

- **Paywalled job boards** (LinkedIn, Workday): the agent proceeds with whatever
  text it can fetch and notes the limitation rather than failing.
- **Image-only PDFs**: scanned resumes with no embedded text can't be parsed —
  use a text-based PDF or a `.txt` file.
- **DuckDuckGo rate limits**: the search tool uses DuckDuckGo's unofficial API
  and may throttle; it retries with backoff automatically.
</content>
