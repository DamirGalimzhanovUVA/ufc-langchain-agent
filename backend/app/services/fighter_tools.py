import json
import logging
from typing import Any
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool, StructuredTool
from tavily import TavilyClient

from services.fighter_stats_provider import FighterStatsProvider

logger = logging.getLogger("uvicorn.error")


class TavilyNewsClient:
    def __init__(
        self, api_key: str | None, timeout_seconds: float = 10
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def search_news(self, query: str) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY must be set")

        payload = {
            "query": query,
            "topic": "general",
            "search_depth": "basic",
            "max_results": 5,
            "include_answer": False,
            "include_raw_content": False,
        }
        body = json.dumps(payload).encode()
        request = Request(
            "https://api.tavily.com/search",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        logger.info("Tavily request payload: %s", payload)
        with urlopen(request, timeout=self.timeout_seconds) as response:
            response_payload = json.load(response)

        results = [
            {
                "title": result.get("title"),
                "url": result.get("url"),
                "published_date": result.get("published_date"),
                "content": result.get("content"),
            }
            for result in response_payload.get("results", [])
            if isinstance(result, dict)
        ]
        logger.info(
            "Tavily search results for query %r: %s",
            query,
            results,
        )
        return {"query": query, "results": results}


class WikipediaClient:
    api_url = "https://en.wikipedia.org/w/api.php"

    def __init__(self, timeout_seconds: float = 10) -> None:
        self.timeout_seconds = timeout_seconds

    def get_fighter_page(self, fighter_name: str) -> dict[str, Any]:
        query = urlencode(
            {
                "action": "query",
                "prop": "extracts|info",
                "inprop": "url",
                "exintro": "1",
                "explaintext": "1",
                "exchars": "1200",
                "redirects": "1",
                "titles": fighter_name,
                "format": "json",
                "formatversion": "2",
            }
        )
        request = Request(
            f"{self.api_url}?{query}",
            headers={"User-Agent": "ufc-langchain-chat/1.0"},
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.load(response)

        pages = payload.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            return {
                "fighter": fighter_name,
                "found": False,
                "summary": None,
                "url": None,
            }

        page = pages[0]
        summary = page.get("extract", "")
        if len(summary) > 1200:
            summary = f"{summary[:1197]}..."

        return {
            "fighter": page.get("title", fighter_name),
            "found": True,
            "summary": summary,
            "url": page.get("fullurl"),
        }


def extract_fight_description(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    article_body = soup.select_one("main#content article #zephr-anchor")
    if article_body is None:
        article_body = soup.select_one("main#content article")
    if article_body is None:
        raise ValueError("MMA Fighting article body was not found")

    paragraphs: list[str] = []
    for component in article_body.select(
        ".duet--article--article-body-component"
    ):
        if component.find(["h2", "h3"]):
            break

        paragraph = component.select_one(
            "p.duet--article--standard-paragraph"
        )
        if paragraph is not None:
            paragraphs.append(paragraph.get_text(" ", strip=True))

    if not paragraphs:
        raise ValueError("MMA Fighting article description was not found")

    return "\n\n".join(paragraphs)


def get_fight_description(
    query: str,
    tavily_client: TavilyClient | None,
    timeout_seconds: float = 10,
) -> dict[str, Any]:
    if tavily_client is None:
        raise RuntimeError("TAVILY_API_KEY must be set")

    response = tavily_client.search(
        query=query,
        search_depth="advanced",
        include_domains=["mmafighting.com"],
    )
    article_url = next(
        (
            result.get("url")
            for result in response.get("results", [])
            if isinstance(result, dict)
            and isinstance(result.get("url"), str)
            and (
                urlparse(result["url"]).hostname == "mmafighting.com"
                or urlparse(result["url"]).hostname == "www.mmafighting.com"
            )
        ),
        None,
    )
    if article_url is None:
        return {
            "query": query,
            "found": False,
            "url": None,
            "description": None,
        }

    request = Request(
        article_url,
        headers={"User-Agent": "ufc-langchain-chat/1.0"},
    )
    with urlopen(request, timeout=timeout_seconds) as article_response:
        html = article_response.read().decode("utf-8")

    return {
        "query": query,
        "found": True,
        "url": article_url,
        "description": extract_fight_description(html),
    }


def create_fighter_tools(
    news_client: TavilyNewsClient,
    stats_provider: FighterStatsProvider,
    wikipedia_client: WikipediaClient,
    tavily_client: TavilyClient | None = None,
) -> list[BaseTool]:
    def search_fighter_news(query: str) -> dict[str, Any]:
        """Search MMA news using a focused, standalone semantic query."""
        arguments = {"query": query}
        logger.info("Tool call: name=%s arguments=%s", "fighter_news", arguments)
        result = news_client.search_news(query)
        logger.info("Tool result: name=%s result=%s", "fighter_news", result)
        return result

    def get_fighter_stats(fighter_name: str) -> dict[str, Any]:
        """Get current career and fight statistics for a named MMA fighter."""
        arguments = {"fighter_name": fighter_name}
        logger.info("Tool call: name=%s arguments=%s", "fighter_stats", arguments)
        result = {
            "fighter": fighter_name,
            "stats": stats_provider.get_fighter_stats(fighter_name),
        }
        logger.info("Tool result: name=%s result=%s", "fighter_stats", result)
        return result

    def get_fighter_wikipedia(fighter_name: str) -> dict[str, Any]:
        """Get a named MMA fighter's biography from their Wikipedia page."""
        arguments = {"fighter_name": fighter_name}
        logger.info(
            "Tool call: name=%s arguments=%s", "fighter_wikipedia", arguments
        )
        result = wikipedia_client.get_fighter_page(fighter_name)
        logger.info(
            "Tool result: name=%s result=%s", "fighter_wikipedia", result
        )
        return result

    def find_fight_description(query: str) -> dict[str, Any]:
        """Find and scrape an MMA Fighting live blog's fight description."""
        arguments = {"query": query}
        logger.info(
            "Tool call: name=%s arguments=%s",
            "fight_description",
            arguments,
        )
        result = get_fight_description(query, tavily_client)
        logger.info(
            "Tool result: name=%s result=%s",
            "fight_description",
            result,
        )
        return result

    return [
        StructuredTool.from_function(
            search_fighter_news,
            name="fighter_news",
            description=(
                "Search MMA news for recent events, statements, interviews, "
                "announcements, or current coverage. Generate a focused, "
                "standalone search query that preserves every person and topic "
                "from the user's question."
            ),
        ),
        StructuredTool.from_function(
            get_fighter_stats,
            name="fighter_stats",
            description=(
                "Retrieve structured career and fight statistics for an MMA "
                "fighter. Use this for records, measurements, and performance data."
            ),
        ),
        StructuredTool.from_function(
            get_fighter_wikipedia,
            name="fighter_wikipedia",
            description=(
                "Retrieve the introduction and URL from an MMA fighter's "
                "Wikipedia page. Use this for biography and background."
            ),
        ),
        StructuredTool.from_function(
            find_fight_description,
            name="fight_description",
            description=(
                "Find an MMA Fighting live-blog page for a specific fight and "
                "retrieve the matchup description from the main article. Pass "
                "a focused query containing both fighter names and 'live blog'."
            ),
        ),
    ]
