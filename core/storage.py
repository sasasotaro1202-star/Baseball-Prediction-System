"""Parquet 保存。pyarrow が無い環境でも CSV にフォールバックして必ず動く。"""
from __future__ import annotations
import pandas as pd
from core import config

try:
    import pyarrow  # noqa: F401
    _HAS_PARQUET = True
except ImportError:
    _HAS_PARQUET = False

_EXT = "parquet" if _HAS_PARQUET else "csv"
_DATE_COLS = ("date", "predicted_at", "saved_at", "captured_at")


def _dir(layer: str, dataset: str):
    return config.path("data", layer, dataset)


def _write_one(df: pd.DataFrame, fp) -> None:
    if _HAS_PARQUET:
        df.to_parquet(fp, index=False)
    else:
        df.to_csv(fp, index=False)


def _read_one(fp) -> pd.DataFrame:
    if _HAS_PARQUET:
        df = pd.read_parquet(fp)
    else:
        df = pd.read_csv(fp)
        # CSV フォールバック時は既知の列名を datetime に復元
        for col in _DATE_COLS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def write(df: pd.DataFrame, layer: str, dataset: str, mode: str = "overwrite",
          partition: str = "-") -> None:
    d = _dir(layer, dataset)
    fp = d / f"{partition}.{_EXT}"
    if mode == "append" and fp.exists():
        try:
            old = _read_one(fp)
            df = pd.concat([old, df], ignore_index=True)
        except Exception:
            pass
    _write_one(df, fp)


def read(layer: str, dataset: str) -> pd.DataFrame:
    d = _dir(layer, dataset)
    if not d.exists():
        return pd.DataFrame()
    files = list(d.glob(f"*.{_EXT}"))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            frames.append(_read_one(f))
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def row_count(layer: str, dataset: str) -> int:
    return len(read(layer, dataset))


def datasets(layer: str) -> list[str]:
    d = config.path("data", layer)
    return [p.name for p in d.iterdir() if p.is_dir()] if d.exists() else []


def bytes_on_disk(layer: str) -> int:
    d = config.path("data", layer)
    if not d.exists():
        return 0
    return sum(f.stat().st_size for f in d.rglob(f"*.{_EXT}"))
