import io
from unittest.mock import Mock

import pytest

import services.fighter_stats_provider as provider_module
from services.fighter_stats_provider import HttpFighterStatsProvider


class JsonResponse(io.BytesIO):
    def __enter__(self) -> "JsonResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_http_stats_provider_requests_configured_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    urlopen = Mock(return_value=JsonResponse(b'{"wins": 22, "losses": 3}'))
    monkeypatch.setattr(provider_module, "urlopen", urlopen)
    provider = HttpFighterStatsProvider(
        "https://stats.example/fighters", api_key="secret"
    )

    result = provider.get_fighter_stats("Valentina Shevchenko")

    assert result == {"wins": 22, "losses": 3}
    request = urlopen.call_args.args[0]
    assert request.full_url == (
        "https://stats.example/fighters?"
        "fighter_name=Valentina+Shevchenko"
    )
    assert request.get_header("Authorization") == "Bearer secret"
    urlopen.assert_called_once_with(request, timeout=10)


def test_http_stats_provider_supports_fighter_name_url_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    urlopen = Mock(return_value=JsonResponse(b'{"wins": 12}'))
    monkeypatch.setattr(provider_module, "urlopen", urlopen)
    provider = HttpFighterStatsProvider(
        "https://stats.example/fighters/{fighter_name}"
    )

    provider.get_fighter_stats("Tom Aspinall")

    request = urlopen.call_args.args[0]
    assert request.full_url == "https://stats.example/fighters/Tom%20Aspinall"


def test_http_stats_provider_requires_api_url() -> None:
    provider = HttpFighterStatsProvider("")

    with pytest.raises(RuntimeError, match="FIGHTER_STATS_API_URL must be set"):
        provider.get_fighter_stats("Alex Pereira")
