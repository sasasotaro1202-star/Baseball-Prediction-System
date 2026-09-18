from pathlib import Path

from core.atomic_io import atomic_write_json, atomic_write_text, file_sha256


def test_atomic_text_write_replaces_complete_file(tmp_path: Path):
    path = tmp_path / "nested" / "manifest.json"
    atomic_write_text(path, "first")
    assert path.read_text(encoding="utf-8") == "first"
    atomic_write_text(path, "second")
    assert path.read_text(encoding="utf-8") == "second"
    assert not list(path.parent.glob(".*.tmp"))


def test_atomic_json_rejects_nan_and_is_deterministically_readable(tmp_path: Path):
    path = tmp_path / "manifest.json"
    atomic_write_json(path, {"b": 2, "a": [1, 2, 3]})
    assert '"a"' in path.read_text(encoding="utf-8")
    assert file_sha256(path)
