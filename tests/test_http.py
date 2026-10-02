from __future__ import annotations

import requests
import pytest

import core.http as http


class FakeResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code
        self.headers = {"Retry-After": "300"}
        self.content = b"{}"
        self.text = "{}"
        self.apparent_encoding = "utf-8"
        self.encoding = "utf-8"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_request_retries_read_timeout_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr(http.time, "sleep", sleeps.append)
    sess = FakeSession([
        requests.ReadTimeout("slow read"),
        FakeResponse(200),
    ])

    response = http.request(
        sess,
        "https://example.test/data",
        timeout=(8, 45),
        retries=2,
    )

    assert response.status_code == 200
    assert len(sess.calls) == 2
    assert sess.calls[0][2]["timeout"] == (8.0, 45.0)
    assert sleeps == [0.75]


def test_request_retries_transient_status_without_honoring_retry_after(monkeypatch):
    sleeps = []
    monkeypatch.setattr(http.time, "sleep", sleeps.append)
    sess = FakeSession([
        FakeResponse(503),
        FakeResponse(200),
    ])

    response = http.request(
        sess,
        "https://example.test/data",
        timeout=20,
        retries=2,
    )

    assert response.status_code == 200
    assert len(sess.calls) == 2
    assert sleeps == [0.75]


def test_request_does_not_retry_non_transient_client_error(monkeypatch):
    sleeps = []
    monkeypatch.setattr(http.time, "sleep", sleeps.append)
    sess = FakeSession([FakeResponse(404)])

    with pytest.raises(RuntimeError, match="bounded HTTP request failed"):
        http.request(sess, "https://example.test/missing", timeout=10, retries=4)

    assert len(sess.calls) == 1
    assert sleeps == []


def test_timeout_validation_is_strict():
    with pytest.raises(ValueError):
        http.request(FakeSession([FakeResponse(200)]), "https://example.test", timeout=(8, 0))
