#!/usr/bin/env python3
"""PIT-safe, quota-aware REST client for the optional StatsHawk CI bridge.

The ChatGPT-connected StatsHawk MCP remains the primary acquisition path for
interactive research. This client exists so GitHub Actions can use a user's
own StatsHawk REST key when one is explicitly configured.

Safety:
- API keys are read from an environment variable and are never logged.
- Only relative /v1 paths are accepted.
- 401/403/429 quota failures are fail-closed.
- 429 rate-limit failures may be retried with bounded backoff.
- 5xx transport failures may be retried; deterministic 4xx failures are not.
- Response quota headers are preserved as metadata for audit, never inferred
  as PIT availability timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any, Mapping
from urllib.parse import urljoin

import requests


class StatsHawkRestError(RuntimeError):
    """Base error for REST client failures."""


class StatsHawkAuthError(StatsHawkRestError):
    """Authentication or permission failure."""


class StatsHawkQuotaExceeded(StatsHawkRestError):
    """Monthly quota is exhausted; never retry."""


class StatsHawkRateLimited(StatsHawkRestError):
    """Per-second rate limit remains after bounded retries."""


class StatsHawkHttpError(StatsHawkRestError):
    """Non-retryable or exhausted HTTP failure."""


@dataclass(frozen=True)
class StatsHawkResponse:
    payload: Any
    status_code: int
    quota_limit: int | None
    quota_remaining: int | None
    quota_reset: str | None
    request_weight: int | None


def _header_int(headers: Mapping[str, str], name: str) -> int | None:
    value = headers.get(name)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _retry_after(headers: Mapping[str, str], default_seconds: float) -> float:
    raw = headers.get("Retry-After")
    if raw in (None, ""):
        return default_seconds
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default_seconds
    return max(0.0, min(value, 30.0))


class StatsHawkRestClient:
    """Small REST client with explicit failure and retry semantics."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = "https://api.statshawk.ai",
        timeout_seconds: float = 20.0,
        max_retries: int = 2,
        backoff_base_seconds: float = 1.0,
        session: requests.Session | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.getenv("STATSHAWK_API_KEY")
        if not key:
            raise StatsHawkAuthError("statshawk_api_key_missing")
        if not isinstance(key, str) or not key.strip():
            raise StatsHawkAuthError("statshawk_api_key_invalid")
        if not base_url.startswith(("https://", "http://")):
            raise ValueError("base_url_must_be_http")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds_must_be_positive")
        if max_retries < 0 or max_retries > 5:
            raise ValueError("max_retries_out_of_range")

        self._api_key = key.strip()
        self._base_url = base_url.rstrip("/") + "/"
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._backoff_base = max(0.0, backoff_base_seconds)
        self._session = session or requests.Session()

    @staticmethod
    def _validate_path(path: str) -> str:
        if not isinstance(path, str) or not path.startswith("/v1/"):
            raise ValueError("path_must_start_with_/v1/")
        if "://" in path:
            raise ValueError("absolute_url_not_allowed")
        return path

    def get(self, path: str, *, params: Mapping[str, Any] | None = None) -> StatsHawkResponse:
        path = self._validate_path(path)
        url = urljoin(self._base_url, path.lstrip("/"))
        headers = {
            "Accept": "application/json",
            "X-API-Key": self._api_key,
            "User-Agent": "Baseball-Prediction-System/statshawk-rest-bridge",
        }

        for attempt in range(self._max_retries + 1):
            try:
                response = self._session.get(
                    url,
                    params=dict(params or {}),
                    headers=headers,
                    timeout=self._timeout,
                )
            except requests.RequestException as exc:
                if attempt >= self._max_retries:
                    raise StatsHawkHttpError("transport_error") from exc
                time.sleep(self._backoff_base * (2**attempt))
                continue

            status = response.status_code
            try:
                payload = response.json()
            except ValueError as exc:
                payload = None
                if status < 200 or status >= 300:
                    payload = None
                else:
                    raise StatsHawkHttpError("response_not_json") from exc

            if 200 <= status < 300:
                return StatsHawkResponse(
                    payload=payload,
                    status_code=status,
                    quota_limit=_header_int(response.headers, "X-Account-Quota-Limit"),
                    quota_remaining=_header_int(response.headers, "X-Account-Quota-Remaining"),
                    quota_reset=response.headers.get("X-Account-Quota-Reset"),
                    request_weight=_header_int(response.headers, "x-statshawk-weight"),
                )

            error_code = None
            if isinstance(payload, dict):
                error = payload.get("error")
                if isinstance(error, dict):
                    error_code = error.get("code")

            if status in {401, 403}:
                raise StatsHawkAuthError(str(error_code or f"http_{status}"))

            if status == 429:
                if error_code == "QUOTA_EXCEEDED":
                    raise StatsHawkQuotaExceeded("quota_exceeded")
                if error_code == "RATE_LIMIT_EXCEEDED":
                    if attempt >= self._max_retries:
                        raise StatsHawkRateLimited("rate_limit_exceeded")
                    time.sleep(
                        _retry_after(
                            response.headers,
                            self._backoff_base * (2**attempt),
                        )
                    )
                    continue
                raise StatsHawkHttpError(str(error_code or "http_429"))

            if 500 <= status < 600 and attempt < self._max_retries:
                time.sleep(
                    _retry_after(
                        response.headers,
                        self._backoff_base * (2**attempt),
                    )
                )
                continue

            raise StatsHawkHttpError(str(error_code or f"http_{status}"))

        raise AssertionError("unreachable")

    def probe_mlb_capabilities(self) -> StatsHawkResponse:
        """Low-cost authenticated connectivity probe; no game data is pulled."""
        return self.get("/v1/competitions/mlb/capabilities")
