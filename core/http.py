"""requests ラッパー。

重要な既知の障害モード: 一部サイト (football-data.co.uk 等) は一時的に落ちている間
`503 Retry-After: 300+` を返す。urllib3 の Retry はデフォルトで Retry-After ヘッダーを
尊重するため、respect_retry_after_header を外さないと 1 リクエストで数分〜数十分ブロック
され、GitHub Actions のジョブタイムアウトで失敗する。これが Actions で頻発した失敗の
主因だったため、必ず respect_retry_after_header=False とし、独自の短い backoff で制御する。
"""
from __future__ import annotations
import requests
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry
except ImportError:  # pragma: no cover
    from requests.packages.urllib3.util.retry import Retry


def session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1.0,
        backoff_max=8.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=False,  # サーバー指定の長時間待機を無視する (必須)
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": "Mozilla/5.0 (sports-forecast-platform; +github-actions)"})
    return s


def get_json(sess: requests.Session, url: str, params: dict | None = None, timeout: int = 20):
    from core.parallel import throttle_for
    from urllib.parse import urlparse
    throttle_for(urlparse(url).netloc).wait()
    r = sess.get(url, params=params, timeout=(5, timeout))
    r.raise_for_status()
    return r.json()


def get_text(sess: requests.Session, url: str, params: dict | None = None, timeout: int = 20) -> str:
    from core.parallel import throttle_for
    from urllib.parse import urlparse
    throttle_for(urlparse(url).netloc).wait()
    r = sess.get(url, params=params, timeout=(5, timeout))
    r.raise_for_status()
    return r.text
