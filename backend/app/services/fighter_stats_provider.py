import json
from typing import Any, Protocol
from urllib.parse import quote
from urllib.request import Request, urlopen


class FighterStatsProvider(Protocol):
    def get_fighter_stats(self, fighter_name: str) -> dict[str, Any]:
        """Retrieve statistics for a fighter."""


class HttpFighterStatsProvider:
    api_url = "https://api.citoapi.com/api/v1/ufc/fighters/{slug}/stats"

    def __init__(
        self,
        api_key: str | None = None,
        timeout_seconds: float = 10,
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def get_fighter_stats(self, fighter_name: str) -> dict[str, Any]:
        slug = "-".join(fighter_name.replace("*", "").lower().split())
        url = self.api_url.format(slug=quote(slug, safe="-"))

        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        request = Request(url, headers=headers)
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.load(response)

        if not isinstance(payload, dict):
            raise ValueError("The fighter stats API must return a JSON object")

        return payload
