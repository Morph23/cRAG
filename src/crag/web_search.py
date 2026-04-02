
from __future__ import annotations

import re
from typing import List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config as cfg


_REWRITE_SYSTEM = (
    "You are a search query optimizer. "
    "Given a question or statement, extract the key concepts and rewrite them as "
    "a short, effective search query (3-7 keywords). "
    "Output ONLY the search query, nothing else."
)


def rewrite_query(query: str, model: str = cfg.QUERY_REWRITE_MODEL) -> str:
    try:
        if "claude" in model.lower():
            import anthropic
            client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
            resp = client.messages.create(
                model=model,
                max_tokens=64,
                system=_REWRITE_SYSTEM,
                messages=[{"role": "user", "content": query}],
            )
            return resp.content[0].text.strip()
        else:
            import openai
            client = openai.OpenAI(api_key=cfg.OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _REWRITE_SYSTEM},
                    {"role": "user",   "content": query},
                ],
                max_tokens=64,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
    except Exception:
        return query


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

_WIKIPEDIA_PRIORITY_DOMAINS = {"en.wikipedia.org", "wikipedia.org"}


def _url_priority(url: str) -> int:
    domain = urlparse(url).netloc.lstrip("www.")
    return 0 if domain in _WIKIPEDIA_PRIORITY_DOMAINS else 1


def fetch_page_text(url: str, timeout: int = 10) -> str:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        return ""


def _search_tavily(keywords: str, num_results: int) -> List[str]:
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=cfg.TAVILY_API_KEY)
        resp = client.search(
            query=keywords,
            search_depth="basic",
            max_results=num_results,
            include_raw_content=True,
        )
        results = []
        for r in resp.get("results", []):
            content = r.get("raw_content") or r.get("content") or ""
            if content:
                results.append(content)
        return results[:num_results]
    except Exception:
        return []


def _search_serp(keywords: str, num_results: int) -> List[str]:
    try:
        params = {
            "q":       keywords,
            "num":     num_results,
            "api_key": cfg.SERP_API_KEY,
            "engine":  "google",
        }
        resp = requests.get("https://serpapi.com/search", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        urls = [r["link"] for r in data.get("organic_results", [])[:num_results]]
        urls.sort(key=_url_priority)
        return [t for url in urls if (t := fetch_page_text(url))][:num_results]
    except Exception:
        return []


def _search_duckduckgo(keywords: str, num_results: int) -> List[str]:
    try:
        from duckduckgo_search import DDGS
        raw = DDGS().text(keywords, max_results=num_results) or []
        texts = []
        for r in raw:
            body = r.get("body", "")
            if body:
                texts.append(body)
        return texts[:num_results]
    except Exception:
        return []


def web_search(
    query: str,
    num_results: int = cfg.WEB_SEARCH_RESULTS,
    rewrite: bool = True,
) -> List[str]:
    keywords = rewrite_query(query) if rewrite else query

    results: List[str] = []

    if cfg.TAVILY_API_KEY:
        results = _search_tavily(keywords, num_results)

    if not results and cfg.SERP_API_KEY:
        results = _search_serp(keywords, num_results)

    if not results:
        results = _search_duckduckgo(keywords, num_results)

    return results
