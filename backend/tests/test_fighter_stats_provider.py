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


def test_http_stats_provider_requests_cito_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    urlopen = Mock(return_value=JsonResponse(b'{"wins": 22, "losses": 3}'))
    monkeypatch.setattr(provider_module, "urlopen", urlopen)
    provider = HttpFighterStatsProvider(api_key="secret")

    result = provider.get_fighter_stats("Valentina Shevchenko")

    assert result == {"wins": 22, "losses": 3}
    request = urlopen.call_args.args[0]
    assert request.full_url == (
        "https://api.citoapi.com/api/v1/ufc/fighters/"
        "valentina-shevchenko/stats"
    )
    assert request.get_header("X-api-key") == "secret"
    urlopen.assert_called_once_with(request, timeout=10)


@pytest.mark.parametrize(
    ("fighter_name", "slug"),
    [
        ("*Dricucs* Du Plessis", "dricucs-du-plessis"),
        ("Benoit Saint Denis", "benoit-saint-denis"),
        ("Loneer Kavanagh", "loneer-kavanagh"),
    ],
)
def test_http_stats_provider_builds_fighter_slug(
    monkeypatch: pytest.MonkeyPatch,
    fighter_name: str,
    slug: str,
) -> None:
    urlopen = Mock(return_value=JsonResponse(b'{"wins": 12}'))
    monkeypatch.setattr(provider_module, "urlopen", urlopen)
    provider = HttpFighterStatsProvider()

    provider.get_fighter_stats(fighter_name)

    request = urlopen.call_args.args[0]
    assert request.full_url == (
        f"https://api.citoapi.com/api/v1/ufc/fighters/{slug}/stats"
    )
