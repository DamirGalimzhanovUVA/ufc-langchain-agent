import json
import logging
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from langchain_core.tools import BaseTool, StructuredTool

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
            "topic": "news",
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


def create_fighter_tools(
    news_client: TavilyNewsClient,
    stats_provider: FighterStatsProvider,
    wikipedia_client: WikipediaClient,
) -> list[BaseTool]:
    def search_fighter_news(query: str) -> dict[str, Any]:
        """Search MMA news using a focused, standalone semantic query."""
        return news_client.search_news(query)

    def get_fighter_stats(fighter_name: str) -> dict[str, Any]:
        """Get current career and fight statistics for a named MMA fighter."""
        return {
            "fighter": fighter_name,
            "stats": stats_provider.get_fighter_stats(fighter_name),
        }

    def get_fighter_wikipedia(fighter_name: str) -> dict[str, Any]:
        """Get a named MMA fighter's biography from their Wikipedia page."""
        return wikipedia_client.get_fighter_page(fighter_name)

    return [
        StructuredTool.from_function(
            search_fighter_news,
            name="fighter_news",
            description=(
                "Search MMA news for recent events, statements, interviews, "
                "announcements, or current coverage. Generate a focused, "
                "standalone search query that preserves every person and topic "
                "from the user's question. If previous results were not relevant, "
                "refine the query to target the missing information."
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
    ]
