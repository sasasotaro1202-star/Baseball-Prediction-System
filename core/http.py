"""Bounded, retrying HTTP helpers for Baseball data acquisition.

Design goals:
- Never wait indefinitely on a remote server.
- Distinguish connect timeout from read timeout.
- Retry transient HTTP responses and network/read timeouts with short bounded backoff.
- Bound the entire request/retry budget, not only each socket read.
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
DEFAULT_TOTAL_TIMEOUT = 180.0
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


def _total_timeout(timeout: int | float | None) -> float:
    if timeout is None:
        return DEFAULT_TOTAL_TIMEOUT
    value = float(timeout)
    if value <= 0:
        raise ValueError("total HTTP timeout must be strictly positive")
    return value


def _attempt_timeout(
    timeout_pair: tuple[float, float],
    remaining: float,
) -> tuple[float, float]:
    """Scale connect/read timeouts to fit inside the remaining total budget."""
    if remaining >= sum(timeout_pair):
        return timeout_pair
    if remaining <= 0:
        raise ValueError("no total HTTP timeout budget remains")
    total = sum(timeout_pair)
    return (
        remaining * timeout_pair[0] / total,
        remaining * timeout_pair[1] / total,
    )


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
    total_timeout: int | float | None = None,
) -> requests.Response:
    """Perform a bounded HTTP request with retryable-error recovery.

    A request has at most retries network/HTTP attempts and at most
    total_timeout wall-clock budget (including backoff). Connect and read
    timeouts are explicit. 4xx errors other than the retryable subset are
    surfaced immediately. Retry-After is never allowed to create an
    unbounded sleep.
    """
    attempts = max(1, int(retries))
    timeout_pair = _timeouts(timeout)
    total_budget = _total_timeout(total_timeout)
    deadline = time.monotonic() + total_budget
    last_error: Exception | None = None
    attempts_made = 0

    for attempt in range(attempts):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        attempts_made += 1
        attempt_timeout = _attempt_timeout(timeout_pair, remaining)

        try:
            response = sess.request(
                method.upper(),
                url,
                params=params,
                headers=dict(headers or {}),
                timeout=attempt_timeout,
            )
            if response.status_code in RETRYABLE_STATUS and attempt + 1 < attempts:
                last_error = requests.HTTPError(
                    f"HTTP {response.status_code} for {url}",
                    response=response,
                )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(_backoff(attempt), remaining))
                continue
            response.raise_for_status()
            return response
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(_backoff(attempt), remaining))
        except requests.HTTPError as exc:
            last_error = exc
            status = getattr(exc.response, "status_code", None)
            if attempt + 1 >= attempts or status not in RETRYABLE_STATUS:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(_backoff(attempt), remaining))

    raise RuntimeError(
        f"bounded HTTP request failed after {attempts_made}/{attempts} attempts "
        f"within {total_budget:.1f}s: {method.upper()} {url}: {last_error}"
    ) from last_error


def get_json(
    sess: requests.Session,
    url: str,
    params: Mapping[str, Any] | None = None,
    timeout: int | float | tuple[float, float] | None = None,
    *,
    headers: Mapping[str, str] | None = None,
    retries: int = DEFAULT_RETRIES,
    total_timeout: int | float | None = None,
):
    return request(
        sess,
        url,
        params=params,
        headers=headers,
        timeout=timeout,
        retries=retries,
        total_timeout=total_timeout,
    ).json()


def get_text(
    sess: requests.Session,
    url: str,
    params: Mapping[str, Any] | None = None,
    timeout: int | float | tuple[float, float] | None = None,
    *,
    headers: Mapping[str, str] | None = None,
    retries: int = DEFAULT_RETRIES,
    total_timeout: int | float | None = None,
) -> str:
    return request(
        sess,
        url,
        params=params,
        headers=headers,
        timeout=timeout,
        retries=retries,
        total_timeout=total_timeout,
    ).text
