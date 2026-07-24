import io
import json
from unittest.mock import Mock

import pytest

import services.fighter_tools as fighter_tools_module
from services.fighter_tools import (
    TavilyNewsClient,
    WikipediaClient,
    create_fighter_tools,
)


class JsonResponse(io.BytesIO):
    def __enter__(self) -> "JsonResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_tavily_news_client_searches_model_generated_query_without_day_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = {
        "results": [
            {
                "title": "Fight announced",
                "url": "https://news.example/fight",
                "published_date": "2026-07-22",
                "content": "Article excerpt describing the reported statement.",
            }
        ]
    }
    urlopen = Mock(return_value=JsonResponse(json.dumps(response).encode()))
    monkeypatch.setattr(fighter_tools_module, "urlopen", urlopen)
    logger = Mock()
    monkeypatch.setattr(fighter_tools_module, "logger", logger)
    client = TavilyNewsClient("tavily-key")
    query = "What did Makhachev say about Ian Garry's fighting style?"

    result = client.search_news(query)

    assert result == {
        "query": query,
        "results": [
            {
                "title": "Fight announced",
                "url": "https://news.example/fight",
                "published_date": "2026-07-22",
                "content": "Article excerpt describing the reported statement.",
            }
        ],
    }
    request = urlopen.call_args.args[0]
    assert request.full_url == "https://api.tavily.com/search"
    assert request.get_header("Authorization") == "Bearer tavily-key"
    body = json.loads(request.data)
    assert "api_key" not in body
    assert body["query"] == query
    assert body["topic"] == "news"
    assert "days" not in body
    assert body["max_results"] == 5
    logger.info.assert_called_once_with(
        "Tavily request payload: %s",
        {
            "query": query,
            "topic": "news",
            "search_depth": "basic",
            "max_results": 5,
            "include_answer": False,
            "include_raw_content": False,
        },
    )


def test_tavily_news_client_requires_api_key() -> None:
    client = TavilyNewsClient(None)

    with pytest.raises(RuntimeError, match="TAVILY_API_KEY must be set"):
        client.search_news("Max Holloway latest news")


def test_wikipedia_client_returns_page_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = {
        "query": {
            "pages": [
                {
                    "title": "Georges St-Pierre",
                    "extract": "Canadian former mixed martial artist.",
                    "fullurl": "https://en.wikipedia.org/wiki/Georges_St-Pierre",
                }
            ]
        }
    }
    urlopen = Mock(return_value=JsonResponse(json.dumps(response).encode()))
    monkeypatch.setattr(fighter_tools_module, "urlopen", urlopen)

    result = WikipediaClient().get_fighter_page("Georges St-Pierre")

    assert result == {
        "fighter": "Georges St-Pierre",
        "found": True,
        "summary": "Canadian former mixed martial artist.",
        "url": "https://en.wikipedia.org/wiki/Georges_St-Pierre",
    }
    request = urlopen.call_args.args[0]
    assert "titles=Georges+St-Pierre" in request.full_url
    assert "action=query" in request.full_url


def test_tools_delegate_to_injected_clients_and_provider() -> None:
    news_client = Mock()
    news_client.search_news.return_value = {"results": []}
    stats_provider = Mock()
    stats_provider.get_fighter_stats.return_value = {"wins": 30}
    wikipedia_client = Mock()
    wikipedia_client.get_fighter_page.return_value = {"found": True}
    tools = create_fighter_tools(
        news_client, stats_provider, wikipedia_client
    )
    tools_by_name = {tool.name: tool for tool in tools}

    news_result = tools_by_name["fighter_news"].invoke(
        {"query": "What did José Aldo say about retirement?"}
    )
    stats_result = tools_by_name["fighter_stats"].invoke(
        {"fighter_name": "José Aldo"}
    )
    wikipedia_result = tools_by_name["fighter_wikipedia"].invoke(
        {"fighter_name": "José Aldo"}
    )

    assert news_result == {"results": []}
    news_client.search_news.assert_called_once_with(
        "What did José Aldo say about retirement?"
    )
    assert stats_result == {
        "fighter": "José Aldo",
        "stats": {"wins": 30},
    }
    assert wikipedia_result == {"found": True}
