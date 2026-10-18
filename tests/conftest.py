import pytest


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must never reach the internet; rate lookups fall back to the constant."""

    def offline(*_a: object, **_k: object) -> None:
        raise ConnectionError("network disabled in tests")

    monkeypatch.setattr("volscan.data.rates._fetch_series", offline)
    monkeypatch.setattr("requests.sessions.Session.request", offline)
