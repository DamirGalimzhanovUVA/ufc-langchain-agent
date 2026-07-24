import io
import json
from pathlib import Path
from unittest.mock import Mock, call

import pytest

import services.fighter_tools as fighter_tools_module
from services.fighter_tools import (
    TavilyNewsClient,
    WikipediaClient,
    create_fighter_tools,
    extract_fight_description,
    get_fight_description,
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
    assert body["topic"] == "general"
    assert "days" not in body
    assert body["max_results"] == 5
    logger.info.assert_has_calls(
        [
            call(
                "Tavily request payload: %s",
                {
                    "query": query,
                    "topic": "general",
                    "search_depth": "basic",
                    "max_results": 5,
                    "include_answer": False,
                    "include_raw_content": False,
                },
            ),
            call(
                "Tavily search results for query %r: %s",
                query,
                [
                    {
                        "title": "Fight announced",
                        "url": "https://news.example/fight",
                        "published_date": "2026-07-22",
                        "content": (
                            "Article excerpt describing the reported statement."
                        ),
                    }
                ],
            ),
        ]
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


def test_extract_fight_description_uses_main_article_intro() -> None:
    asset_path = (
        Path(__file__).parents[1]
        / "assets"
        / "ufc-white-house-live-blog-alex-pereira-vs-ciryl-gane.html"
    )

    description = extract_fight_description(asset_path.read_text())

    assert description.startswith("This is the UFC White House live blog")
    assert "Alex" in description
    assert "Pereira vs. Ciryl Gane" in description
    assert "Pereira (13-3) is currently No. 3" in description
    assert "Standing in Pereira’s way is Gane (13-2, 1 NC)" in description
    assert "Check out the UFC White House live blog" not in description
    assert "Round 1" not in description


def test_get_fight_description_searches_and_scrapes_mmafighting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    html = """
    <main id="content">
      <article>
        <div id="zephr-anchor">
          <div class="duet--article--article-body-component">
            <p class="duet--article--standard-paragraph">
              Fighter One meets Fighter Two in the main event.
            </p>
          </div>
          <div class="duet--article--article-body-component"><h2>Round 1</h2></div>
          <div class="duet--article--article-body-component">
            <p class="duet--article--standard-paragraph">Live updates.</p>
          </div>
        </div>
      </article>
    </main>
    """
    article_url = (
        "https://www.mmafighting.com/ufc/123/fighter-one-vs-fighter-two"
    )
    search_response = Mock()
    search_response.json.return_value = {
        "results": [
            {"url": "https://example.com/unrelated"},
            {"url": article_url},
        ]
    }
    post = Mock(return_value=search_response)
    article_response = Mock()
    article_response.text = html
    get = Mock(return_value=article_response)
    monkeypatch.setattr(fighter_tools_module.requests, "post", post)
    monkeypatch.setattr(fighter_tools_module.requests, "get", get)
    query = "Fighter One vs Fighter Two live blog"

    result = get_fight_description(query, "tavily-key")

    assert result == {
        "query": query,
        "found": True,
        "url": article_url,
        "description": "Fighter One meets Fighter Two in the main event.",
    }
    post.assert_called_once_with(
        "https://api.tavily.com/search",
        json={
            "query": query,
            "search_depth": "advanced",
            "include_domains": ["mmafighting.com"],
        },
        headers={"Authorization": "Bearer tavily-key"},
        timeout=10,
    )
    search_response.raise_for_status.assert_called_once_with()
    get.assert_called_once_with(
        article_url,
        headers={"User-Agent": "ufc-langchain-chat/1.0"},
        timeout=10,
    )
    article_response.raise_for_status.assert_called_once_with()


def test_get_fight_description_returns_not_found_without_article_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_response = Mock()
    search_response.json.return_value = {
        "results": [{"url": "https://example.com/unrelated"}]
    }
    post = Mock(return_value=search_response)
    get = Mock()
    monkeypatch.setattr(fighter_tools_module.requests, "post", post)
    monkeypatch.setattr(fighter_tools_module.requests, "get", get)

    result = get_fight_description("missing fight live blog", "tavily-key")

    assert result == {
        "query": "missing fight live blog",
        "found": False,
        "url": None,
        "description": None,
    }
    get.assert_not_called()


def test_get_fight_description_requires_api_key_client() -> None:
    with pytest.raises(RuntimeError, match="TAVILY_API_KEY must be set"):
        get_fight_description("Fighter One vs Fighter Two live blog", None)


def test_tools_delegate_to_injected_clients_and_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = Mock()
    monkeypatch.setattr(fighter_tools_module, "logger", logger)
    news_client = Mock()
    news_client.search_news.return_value = {"results": []}
    stats_provider = Mock()
    stats_provider.get_fighter_stats.return_value = {"wins": 30}
    wikipedia_client = Mock()
    wikipedia_client.get_fighter_page.return_value = {"found": True}
    search_response = Mock()
    search_response.json.return_value = {"results": []}
    post = Mock(return_value=search_response)
    monkeypatch.setattr(fighter_tools_module.requests, "post", post)
    tools = create_fighter_tools(
        news_client, stats_provider, wikipedia_client, "tavily-key"
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
    fight_description_result = tools_by_name["fight_description"].invoke(
        {"query": "José Aldo vs Conor McGregor live blog"}
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
    assert fight_description_result == {
        "query": "José Aldo vs Conor McGregor live blog",
        "found": False,
        "url": None,
        "description": None,
    }
    post.assert_called_once_with(
        "https://api.tavily.com/search",
        json={
            "query": "José Aldo vs Conor McGregor live blog",
            "search_depth": "advanced",
            "include_domains": ["mmafighting.com"],
        },
        headers={"Authorization": "Bearer tavily-key"},
        timeout=10,
    )
    logger.info.assert_has_calls(
        [
            call(
                "Tool call: name=%s arguments=%s",
                "fighter_news",
                {
                    "query": (
                        "What did José Aldo say about retirement?"
                    )
                },
            ),
            call(
                "Tool result: name=%s result=%s",
                "fighter_news",
                {"results": []},
            ),
            call(
                "Tool call: name=%s arguments=%s",
                "fighter_stats",
                {"fighter_name": "José Aldo"},
            ),
            call(
                "Tool result: name=%s result=%s",
                "fighter_stats",
                {
                    "fighter": "José Aldo",
                    "stats": {"wins": 30},
                },
            ),
            call(
                "Tool call: name=%s arguments=%s",
                "fighter_wikipedia",
                {"fighter_name": "José Aldo"},
            ),
            call(
                "Tool result: name=%s result=%s",
                "fighter_wikipedia",
                {"found": True},
            ),
            call(
                "Tool call: name=%s arguments=%s",
                "fight_description",
                {"query": "José Aldo vs Conor McGregor live blog"},
            ),
            call(
                "Tool result: name=%s result=%s",
                "fight_description",
                {
                    "query": "José Aldo vs Conor McGregor live blog",
                    "found": False,
                    "url": None,
                    "description": None,
                },
            ),
        ]
    )
