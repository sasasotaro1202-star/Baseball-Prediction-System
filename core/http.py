"""Bounded, retrying HTTP helpers for Baseball data acquisition.

Design goals:
- Never wait indefinitely on a remote server.
- Distinguish connect timeout from read timeout.
- Retry transient HTTP responses and network/read timeouts with short bounded backoff.
- Never honor an unbounded server-provided Retry-After delay.
- Keep failure explicit so PIT-critical callers can fail closed instead of inventing data.

This module is intentionally dependency-light and uses requests.Session.
"""
from __future__ import annotations

import time
from typing import Any, Mapping

import requests
from requests.adapters import HTTPAdapter


DEFAULT_CONNECT_TIMEOUT = 8.0
DEFAULT_READ_TIMEOUT = 45.0
DEFAULT_RETRIES = 4
MAX_BACKOFF_SECONDS = 8.0
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


def session(*, user_agent: str | None = None) -> requests.Session:
    """Create a session with HTTP-level retries fully controlled by this module.

    urllib3 adapter retries are disabled so a caller never accidentally gets a
    second opaque retry layer or a long server-controlled Retry-After sleep.
    """
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=0))
    s.mount("http://", HTTPAdapter(max_retries=0))
    s.headers.update(
        {
            "User-Agent": user_agent
            or "Mozilla/5.0 (sports-forecast-platform; +github-actions)"
        }
    )
    return s


def _timeouts(timeout: int | float | tuple[float, float] | None) -> tuple[float, float]:
    if timeout is None:
        return DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT
    if isinstance(timeout, tuple):
        if len(timeout) != 2:
            raise ValueError("timeout tuple must be (connect_timeout, read_timeout)")
        connect, read = float(timeout[0]), float(timeout[1])
    else:
        connect, read = DEFAULT_CONNECT_TIMEOUT, float(timeout)
    if connect <= 0 or read <= 0:
        raise ValueError("HTTP timeouts must be strictly positive")
    return connect, read


def _backoff(attempt: int) -> float:
    # Deterministic bounded backoff: no jitter so CI/replay behaviour remains auditable.
    return min(MAX_BACKOFF_SECONDS, 0.75 * (2**attempt))


def request(
    sess: requests.Session,
    url: str,
    *,
    method: str = "GET",
    params: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: int | float | tuple[float, float] | None = None,
    retries: int = DEFAULT_RETRIES,
) -> requests.Response:
    """Perform a bounded HTTP request with retryable-error recovery.

    A request has at most retries network/HTTP attempts. Connect and read
    timeouts are explicit. 4xx errors other than the retryable subset are
    surfaced immediately. Retry-After is never allowed to create an
    unbounded sleep.
    """
    attempts = max(1, int(retries))
    timeout_pair = _timeouts(timeout)
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            response = sess.request(
                method.upper(),
                url,
                params=params,
                headers=dict(headers or {}),
                timeout=timeout_pair,
            )
            if response.status_code in RETRYABLE_STATUS and attempt + 1 < attempts:
                last_error = requests.HTTPError(
                    f"HTTP {response.status_code} for {url}",
                    response=response,
                )
                time.sleep(_backoff(attempt))
                continue
            response.raise_for_status()
            return response
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                break
            time.sleep(_backoff(attempt))
        except requests.HTTPError as exc:
            last_error = exc
            status = getattr(exc.response, "status_code", None)
            if attempt + 1 >= attempts or status not in RETRYABLE_STATUS:
                break
            time.sleep(_backoff(attempt))

    raise RuntimeError(
        f"bounded HTTP request failed after {attempts} attempts: "
        f"{method.upper()} {url}: {last_error}"
    ) from last_error


def get_json(
    sess: requests.Session,
    url: str,
    params: Mapping[str, Any] | None = None,
    timeout: int | float | tuple[float, float] | None = None,
    *,
    headers: Mapping[str, str] | None = None,
    retries: int = DEFAULT_RETRIES,
):
    return request(
        sess,
        url,
        params=params,
        headers=headers,
        timeout=timeout,
        retries=retries,
    ).json()


def get_text(
    sess: requests.Session,
    url: str,
    params: Mapping[str, Any] | None = None,
    timeout: int | float | tuple[float, float] | None = None,
    *,
    headers: Mapping[str, str] | None = None,
    retries: int = DEFAULT_RETRIES,
) -> str:
    return request(
        sess,
        url,
        params=params,
        headers=headers,
        timeout=timeout,
        retries=retries,
    ).text
