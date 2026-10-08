import requests
import pytest

from research.statshawk_rest_client import (
    StatsHawkAuthError,
    StatsHawkHttpError,
    StatsHawkQuotaExceeded,
    StatsHawkRateLimited,
    StatsHawkRestClient,
)


class FakeResponse:
    def __init__(self, status_code, payload, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, *, params, headers, timeout):
        self.calls.append(
            {"url": url, "params": params, "headers": headers, "timeout": timeout}
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_missing_api_key_fails_closed(monkeypatch):
    monkeypatch.delenv("STATSHAWK_API_KEY", raising=False)
    with pytest.raises(StatsHawkAuthError, match="statshawk_api_key_missing"):
        StatsHawkRestClient()


def test_probe_uses_v1_capabilities_and_preserves_quota_metadata():
    session = FakeSession(
        [
            FakeResponse(
                200,
                {"competition": "mlb"},
                {
                    "X-Account-Quota-Limit": "5000",
                    "X-Account-Quota-Remaining": "4975",
                    "X-Account-Quota-Reset": "2026-11-01T00:00:00Z",
                    "x-statshawk-weight": "1",
                },
            )
        ]
    )
    client = StatsHawkRestClient("secret-value", session=session)
    result = client.probe_mlb_capabilities()

    assert result.payload == {"competition": "mlb"}
    assert result.quota_limit == 5000
    assert result.quota_remaining == 4975
    assert result.request_weight == 1
    assert session.calls[0]["url"] == (
        "https://api.statshawk.ai/v1/competitions/mlb/capabilities"
    )
    assert session.calls[0]["headers"]["X-API-Key"] == "secret-value"


def test_api_key_is_not_in_error():
    session = FakeSession(
        [FakeResponse(401, {"error": {"code": "UNAUTHORIZED"}})]
    )
    client = StatsHawkRestClient("super-secret-key", session=session)
    with pytest.raises(StatsHawkAuthError) as exc:
        client.get("/v1/competitions/mlb/capabilities")
    assert "super-secret-key" not in str(exc.value)


def test_quota_exceeded_is_never_retried(monkeypatch):
    monkeypatch.setattr(
        "research.statshawk_rest_client.time.sleep",
        lambda _: pytest.fail("quota failure must not sleep"),
    )
    session = FakeSession(
        [FakeResponse(429, {"error": {"code": "QUOTA_EXCEEDED"}})]
    )
    client = StatsHawkRestClient("secret", session=session)
    with pytest.raises(StatsHawkQuotaExceeded, match="quota_exceeded"):
        client.get("/v1/competitions/mlb/capabilities")
    assert len(session.calls) == 1


def test_rate_limit_retries_bounded(monkeypatch):
    sleeps = []
    monkeypatch.setattr(
        "research.statshawk_rest_client.time.sleep",
        lambda value: sleeps.append(value),
    )
    session = FakeSession(
        [
            FakeResponse(
                429,
                {"error": {"code": "RATE_LIMIT_EXCEEDED"}},
                {"Retry-After": "0"},
            ),
            FakeResponse(
                200,
                {"competition": "mlb"},
                {"X-Account-Quota-Remaining": "4999"},
            ),
        ]
    )
    client = StatsHawkRestClient("secret", session=session)
    result = client.get("/v1/competitions/mlb/capabilities")

    assert result.status_code == 200
    assert len(session.calls) == 2
    assert sleeps == [0.0]


def test_5xx_retries_then_fails_without_masking(monkeypatch):
    monkeypatch.setattr(
        "research.statshawk_rest_client.time.sleep",
        lambda _: None,
    )
    session = FakeSession(
        [
            FakeResponse(503, {"error": {"code": "UNAVAILABLE"}}),
            FakeResponse(503, {"error": {"code": "UNAVAILABLE"}}),
        ]
    )
    client = StatsHawkRestClient("secret", max_retries=1, session=session)
    with pytest.raises(StatsHawkHttpError, match="UNAVAILABLE"):
        client.get("/v1/competitions/mlb/capabilities")
    assert len(session.calls) == 2


def test_deterministic_4xx_does_not_retry():
    session = FakeSession(
        [FakeResponse(400, {"error": {"code": "BAD_REQUEST"}})]
    )
    client = StatsHawkRestClient("secret", session=session)
    with pytest.raises(StatsHawkHttpError, match="BAD_REQUEST"):
        client.get("/v1/competitions/mlb/capabilities")
    assert len(session.calls) == 1


@pytest.mark.parametrize(
    "path",
    [
        "v1/competitions/mlb/capabilities",
        "/competitions/mlb/capabilities",
        "https://example.com/v1/competitions/mlb/capabilities",
    ],
)
def test_only_relative_v1_paths_are_allowed(path):
    session = FakeSession([])
    client = StatsHawkRestClient("secret", session=session)
    with pytest.raises(ValueError):
        client.get(path)


def test_transport_error_retries_and_then_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "research.statshawk_rest_client.time.sleep",
        lambda _: None,
    )
    session = FakeSession(
        [
            requests.RequestException("temporary"),
            requests.RequestException("still-down"),
        ]
    )
    client = StatsHawkRestClient("secret", max_retries=1, session=session)
    with pytest.raises(StatsHawkHttpError, match="transport_error"):
        client.get("/v1/competitions/mlb/capabilities")
    assert len(session.calls) == 2
