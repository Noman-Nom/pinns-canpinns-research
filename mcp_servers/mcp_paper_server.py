"""
MCP Paper Research Server for PINN / CAN-PINN Research
Provides tools to search and read academic papers via:
  - Semantic Scholar API  (free, no key needed)
  - ArXiv API             (free, no key needed)
  - URL fetcher           (reads any paper/webpage)

Install dependencies:
    pip install mcp httpx beautifulsoup4 arxiv

Run:
    python mcp_paper_server.py
"""

import json
import re
import asyncio
from typing import Any

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

server = Server("paper-research")

# ── helpers ───────────────────────────────────────────────────────────────────

def _clean(text: str, max_chars: int = 4000) -> str:
    """Strip excess whitespace and truncate."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n... [truncated at {max_chars} chars]"
    return text


def _strip_html(html: str) -> str:
    """Very lightweight HTML stripper — no extra deps needed."""
    # remove script / style blocks
    html = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", " ", html,
                  flags=re.DOTALL | re.IGNORECASE)
    # remove all other tags
    html = re.sub(r"<[^>]+>", " ", html)
    # decode common HTML entities
    for entity, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                          ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " ")]:
        html = html.replace(entity, char)
    return html


# ── tool list ─────────────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_semantic_scholar",
            description=(
                "Search Semantic Scholar for academic papers. "
                "Returns title, authors, year, abstract, citation count, and URL. "
                "Best for finding peer-reviewed papers on any topic."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. 'hybrid PINN adaptive sampling Allen-Cahn'"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results (default 5, max 10)",
                        "default": 5
                    },
                    "year_from": {
                        "type": "integer",
                        "description": "Filter papers from this year onwards, e.g. 2020"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="search_arxiv",
            description=(
                "Search ArXiv preprint server for papers. "
                "Returns title, authors, published date, abstract, and PDF link. "
                "Best for the very latest research (preprints before journal publication)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. 'physics informed neural network uncertainty weighting'"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results (default 5, max 10)",
                        "default": 5
                    },
                    "category": {
                        "type": "string",
                        "description": "ArXiv category filter, e.g. 'cs.LG', 'math.NA', 'physics.comp-ph'",
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_paper_details",
            description=(
                "Get full details of a specific paper from Semantic Scholar using its DOI, "
                "ArXiv ID, or Semantic Scholar paper ID. "
                "Returns full abstract, references, and citation context."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": (
                            "Paper identifier. Can be:\n"
                            "  - DOI:  '10.1016/j.jcp.2018.10.045'\n"
                            "  - ArXiv ID: 'arxiv:1711.10561'\n"
                            "  - Semantic Scholar ID: '204bbf...' (from search results)"
                        )
                    }
                },
                "required": ["paper_id"]
            }
        ),
        Tool(
            name="fetch_url",
            description=(
                "Fetch and read the text content of any URL — paper pages, "
                "journal abstracts, or ArXiv HTML pages. "
                "Strips HTML tags and returns readable text."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL to fetch, e.g. 'https://arxiv.org/abs/1711.10561'"
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to return (default 4000)",
                        "default": 4000
                    }
                },
                "required": ["url"]
            }
        ),
        Tool(
            name="find_pinn_references",
            description=(
                "Convenience tool: automatically searches for papers relevant to a PINN topic. "
                "Searches both Semantic Scholar and ArXiv and combines results. "
                "Use this to quickly find references for supervisor meetings."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": (
                            "PINN research topic, e.g.:\n"
                            "  'adaptive sampling PINN'\n"
                            "  'uncertainty weighting loss function PINN'\n"
                            "  'L-BFGS physics informed neural network'\n"
                            "  'Allen-Cahn PINN phase field'\n"
                            "  'CAN-PINN finite difference automatic differentiation'"
                        )
                    }
                },
                "required": ["topic"]
            }
        ),
    ]


# ── tool handlers ─────────────────────────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:

    # ── 1. Semantic Scholar search ────────────────────────────────────────────
    if name == "search_semantic_scholar":
        query     = arguments["query"]
        limit     = min(int(arguments.get("limit", 5)), 10)
        year_from = arguments.get("year_from")

        params: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,year,abstract,citationCount,externalIds,url"
        }
        if year_from:
            params["year"] = f"{year_from}-"

        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params=params
            )

        if resp.status_code != 200:
            return [TextContent(type="text",
                                text=f"Semantic Scholar error {resp.status_code}: {resp.text[:300]}")]

        data = resp.json()
        papers = data.get("data", [])

        if not papers:
            return [TextContent(type="text", text="No papers found.")]

        lines = [f"Found {len(papers)} papers from Semantic Scholar:\n"]
        for i, p in enumerate(papers, 1):
            authors = ", ".join(a["name"] for a in p.get("authors", [])[:3])
            if len(p.get("authors", [])) > 3:
                authors += " et al."
            doi = p.get("externalIds", {}).get("DOI", "")
            arxiv_id = p.get("externalIds", {}).get("ArXiv", "")
            url = p.get("url", "")
            abstract = _clean(p.get("abstract") or "No abstract available.", 500)

            lines.append(
                f"[{i}] {p.get('title', 'Unknown title')}\n"
                f"    Authors: {authors}\n"
                f"    Year: {p.get('year', 'N/A')} | Citations: {p.get('citationCount', 0)}\n"
                f"    DOI: {doi}\n"
                f"    ArXiv: {arxiv_id}\n"
                f"    URL: {url}\n"
                f"    Abstract: {abstract}\n"
            )

        return [TextContent(type="text", text="\n".join(lines))]

    # ── 2. ArXiv search ───────────────────────────────────────────────────────
    elif name == "search_arxiv":
        query    = arguments["query"]
        limit    = min(int(arguments.get("limit", 5)), 10)
        category = arguments.get("category", "")

        search_query = query
        if category:
            search_query = f"cat:{category} AND ({query})"

        params = {
            "search_query": f"all:{search_query}",
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }

        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get("https://export.arxiv.org/api/query", params=params)

        if resp.status_code != 200:
            return [TextContent(type="text",
                                text=f"ArXiv error {resp.status_code}: {resp.text[:300]}")]

        # parse Atom XML manually (no extra dep)
        xml = resp.text
        entries = re.findall(r"<entry>(.*?)</entry>", xml, re.DOTALL)

        if not entries:
            return [TextContent(type="text", text="No ArXiv papers found.")]

        def tag(entry: str, t: str) -> str:
            m = re.search(fr"<{t}[^>]*>(.*?)</{t}>", entry, re.DOTALL)
            return m.group(1).strip() if m else ""

        lines = [f"Found {len(entries)} papers from ArXiv:\n"]
        for i, entry in enumerate(entries, 1):
            title   = _clean(tag(entry, "title"), 200)
            summary = _clean(tag(entry, "summary"), 500)
            published = tag(entry, "published")[:10]
            arxiv_url = re.search(r"<id>(.*?)</id>", entry)
            url = arxiv_url.group(1).strip() if arxiv_url else ""
            pdf_url = url.replace("abs", "pdf") if url else ""
            authors_raw = re.findall(r"<name>(.*?)</name>", entry)
            authors = ", ".join(authors_raw[:3])
            if len(authors_raw) > 3:
                authors += " et al."

            lines.append(
                f"[{i}] {title}\n"
                f"    Authors: {authors}\n"
                f"    Published: {published}\n"
                f"    Abstract: {summary}\n"
                f"    URL: {url}\n"
                f"    PDF: {pdf_url}\n"
            )

        return [TextContent(type="text", text="\n".join(lines))]

    # ── 3. Get paper details ──────────────────────────────────────────────────
    elif name == "get_paper_details":
        paper_id = arguments["paper_id"].strip()

        # normalise arxiv IDs
        if paper_id.lower().startswith("arxiv:"):
            paper_id = "arXiv:" + paper_id[6:]

        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
        params = {
            "fields": (
                "title,authors,year,abstract,citationCount,"
                "externalIds,url,references,tldr"
            )
        }

        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, params=params)

        if resp.status_code != 200:
            return [TextContent(type="text",
                                text=f"Error {resp.status_code}: {resp.text[:300]}")]

        p = resp.json()
        authors = ", ".join(a["name"] for a in p.get("authors", [])[:5])
        doi = p.get("externalIds", {}).get("DOI", "N/A")
        arxiv_id = p.get("externalIds", {}).get("ArXiv", "N/A")
        tldr = p.get("tldr") or {}
        tldr_text = tldr.get("text", "Not available") if isinstance(tldr, dict) else "Not available"
        abstract = _clean(p.get("abstract") or "Not available.", 1500)

        refs = p.get("references", [])[:5]
        ref_lines = []
        for r in refs:
            ref_lines.append(f"  - {r.get('title', 'Unknown')} ({r.get('year', '?')})")

        result = (
            f"PAPER DETAILS\n"
            f"{'='*60}\n"
            f"Title:      {p.get('title', 'Unknown')}\n"
            f"Authors:    {authors}\n"
            f"Year:       {p.get('year', 'N/A')}\n"
            f"Citations:  {p.get('citationCount', 0)}\n"
            f"DOI:        {doi}\n"
            f"ArXiv:      {arxiv_id}\n"
            f"URL:        {p.get('url', 'N/A')}\n\n"
            f"TL;DR:\n{tldr_text}\n\n"
            f"Abstract:\n{abstract}\n\n"
            f"Top References:\n" + ("\n".join(ref_lines) if ref_lines else "  None available")
        )

        return [TextContent(type="text", text=result)]

    # ── 4. Fetch URL ──────────────────────────────────────────────────────────
    elif name == "fetch_url":
        url       = arguments["url"]
        max_chars = int(arguments.get("max_chars", 4000))

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; PaperResearchBot/1.0; "
                "+https://github.com/research)"
            )
        }

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)

        if resp.status_code != 200:
            return [TextContent(type="text",
                                text=f"HTTP {resp.status_code} for {url}")]

        content_type = resp.headers.get("content-type", "")
        if "html" in content_type:
            text = _strip_html(resp.text)
        else:
            text = resp.text

        text = _clean(text, max_chars)
        return [TextContent(type="text",
                            text=f"Content from {url}:\n\n{text}")]

    # ── 5. find_pinn_references (convenience) ────────────────────────────────
    elif name == "find_pinn_references":
        topic = arguments["topic"]

        # run SS and ArXiv searches in parallel
        ss_args  = {"query": topic, "limit": 4}
        arx_args = {"query": topic, "limit": 3}

        ss_result, arx_result = await asyncio.gather(
            call_tool("search_semantic_scholar", ss_args),
            call_tool("search_arxiv", arx_args),
        )

        combined = (
            f"REFERENCES FOR: '{topic}'\n"
            f"{'='*60}\n\n"
            f"── Semantic Scholar (peer-reviewed) ──\n"
            f"{ss_result[0].text}\n\n"
            f"── ArXiv (preprints) ──\n"
            f"{arx_result[0].text}"
        )
        return [TextContent(type="text", text=combined)]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ── entry point ───────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream,
                         server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
