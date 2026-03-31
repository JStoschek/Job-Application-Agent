import json
import os
import re
import time

import anthropic
import markdownify
import pdfplumber
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from ddgs.exceptions import DDGSException

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

TOOL_DEFINITIONS = [
    {
        "name": "web_search",
        "description": (
            "Search the web using DuckDuckGo. Returns top 5 results as title + URL + snippet. "
            "Use for company research, recent news, culture, funding, and background info."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query string"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_webpage",
        "description": (
            "Fetch and parse a webpage URL. Returns cleaned markdown text, truncated to 4000 "
            "characters. Use for job postings, company About pages, and news articles."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL including https://"}
            },
            "required": ["url"],
        },
    },
    {
        "name": "read_resume",
        "description": "Read a resume file (PDF or .txt). Returns the full text content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Absolute or relative path to the resume file",
                }
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "save_output",
        "description": (
            "Save the final markdown report to the output/ directory. "
            "Call this LAST after all research and analysis is complete."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename like 'Acme-Engineer-2026-03-31.md'",
                },
                "content": {
                    "type": "string",
                    "description": "Full markdown content of the report",
                },
            },
            "required": ["filename", "content"],
        },
    },
    {
        "name": "extract_job_details",
        "description": (
            "Extract structured job details from raw job posting text. "
            "Returns JSON with role, company, requirements, tech_stack, "
            "responsibilities, nice_to_have, location, seniority, and salary_range."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Raw text of the job posting",
                }
            },
            "required": ["text"],
        },
    },
]


def web_search(query: str) -> str:
    for attempt in range(3):
        try:
            results = DDGS().text(query, max_results=5)
            if not results:
                return "No results found."
            lines = []
            for r in results:
                lines.append(f"**{r['title']}**\n{r['href']}\n{r['body']}")
            return "\n\n---\n\n".join(lines)
        except DDGSException as e:
            if attempt < 2:
                time.sleep(2**attempt)
                continue
            return f"Search failed after retries: {e}"
        except Exception as e:
            return f"Search error: {e}"
    return "Search failed after retries."


def fetch_webpage(url: str) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        md = markdownify.markdownify(str(soup), heading_style="ATX")
        md = re.sub(r"\n{3,}", "\n\n", md)
        return md[:4000]
    except requests.RequestException as e:
        return f"Error fetching URL: {e}"
    except Exception as e:
        return f"Error parsing page: {e}"


def read_resume(filepath: str) -> str:
    path = filepath.strip()
    if path.lower().endswith(".pdf"):
        try:
            with pdfplumber.open(path) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            text = "\n\n".join(pages)
            if not text.strip():
                return "Error: PDF appears to be image-only or has no extractable text."
            return text
        except Exception as e:
            return f"Error reading PDF: {e}"
    else:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"


def save_output(filename: str, content: str) -> str:
    os.makedirs("output", exist_ok=True)
    filepath = os.path.join("output", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Saved to {filepath}"


def extract_job_details(text: str) -> dict:
    from prompts import SUB_AGENT_SYSTEM_PROMPT

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=SUB_AGENT_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract job details from this posting and return valid JSON only.\n\n"
                    f"{text[:8000]}"
                ),
            }
        ],
    )
    text_block = next(b for b in response.content if b.type == "text")
    raw = text_block.text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


def execute_tool(tool_name: str, tool_input: dict) -> str:
    dispatch = {
        "web_search": lambda i: web_search(i["query"]),
        "fetch_webpage": lambda i: fetch_webpage(i["url"]),
        "read_resume": lambda i: read_resume(i["filepath"]),
        "save_output": lambda i: save_output(i["filename"], i["content"]),
        "extract_job_details": lambda i: json.dumps(
            extract_job_details(i["text"]), indent=2
        ),
    }
    if tool_name not in dispatch:
        return f"Unknown tool: {tool_name}"
    try:
        return dispatch[tool_name](tool_input)
    except Exception as e:
        return f"Tool execution error in {tool_name}: {type(e).__name__}: {e}"
