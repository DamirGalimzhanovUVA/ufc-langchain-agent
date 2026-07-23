import json
from typing import Any, Protocol
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


class FighterStatsProvider(Protocol):
    def get_fighter_stats(self, fighter_name: str) -> dict[str, Any]:
        """Retrieve statistics for a fighter."""


class HttpFighterStatsProvider:
    def __init__(
        self,
        api_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 10,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def get_fighter_stats(self, fighter_name: str) -> dict[str, Any]:
        if not self.api_url:
            raise RuntimeError("FIGHTER_STATS_API_URL must be set")

        if "{fighter_name}" in self.api_url:
            url = self.api_url.format(fighter_name=quote(fighter_name))
        else:
            separator = "&" if "?" in self.api_url else "?"
            url = f"{self.api_url}{separator}{urlencode({'fighter_name': fighter_name})}"

        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = Request(url, headers=headers)
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.load(response)

        if not isinstance(payload, dict):
            raise ValueError("The fighter stats API must return a JSON object")

        return payload
