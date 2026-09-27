"""Universal adapter registry for baseball research sources.

Registration alone never implies data collection readiness. This registry records
the concrete repository adapter/normalizer contract where one exists and lets the
coverage audit distinguish IMPLEMENTED from UNWIRED without auto-promoting PIT/OOS.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from data.source_registry import SOURCES


@dataclass(frozen=True)
class AdapterSpec:
    source_id: str
    module_path: str
    symbol: str
    kind: str
    collection_ready: bool
    notes: str


# Only adapters whose implementation is actually present in this repository are
# listed. Absence from this table is intentionally equivalent to UNWIRED.
ADAPTERS: tuple[AdapterSpec, ...] = (
    AdapterSpec(
        "iblj_official_stats",
        "data.japan_independent_schedule",
        "discover_ibl_j",
        "collector",
        True,
        "Official Shikoku Island League 2026 schedule/results discovery; PIT/starter validation remains separate.",
    ),
    AdapterSpec(
        "bcl_official_stats",
        "data.japan_independent_schedule",
        "discover_bcl",
        "collector",
        True,
        "Official Route-Inn BC League 2026 schedule/results discovery; PIT/starter validation remains separate.",
    ),
    AdapterSpec("espn_mlb","data.espn_baseball_schedule","fetch_espn_scoreboard","collector",True,"Public ESPN MLB scoreboard/game discovery; field-level PIT remains separate."),
    AdapterSpec("espn_college_baseball","data.espn_baseball_schedule","fetch_espn_scoreboard","collector",True,"Public ESPN college baseball scoreboard/game discovery; field-level PIT remains separate."),
    AdapterSpec("espn_international","data.espn_baseball_schedule","fetch_espn_scoreboard","collector",True,"Public ESPN international baseball discovery; competition coverage is slug-dependent."),
    AdapterSpec(
        "kbo_official_stats",
        "data.kbo_public_schedule",
        "fetch_kbo_schedule",
        "collector",
        True,
        "Public KBO daily schedule discovery; starter announcement/PIT/OOS remain separate gates.",
    ),
    AdapterSpec(
        "cpbl_rebas",
        "data.cpbl_public_schedule",
        "discover_cpbl",
        "collector",
        True,
        "Public CPBL schedule discovery; page-rendering/starter/PIT/OOS remain separate gates.",
    ),
    AdapterSpec(
        "statcast",
        "data.mlb_statcast",
        "fetch_statcast",
        "collector",
        True,
        "Public Baseball Savant CSV collector; PIT/OOS remain separately unverified.",
    ),
    AdapterSpec(
        "npb_public_spaia_pbp",
        "data.npb_pbp_adapter",
        "normalize_pbp_frame",
        "normalizer",
        False,
        "Normalizes a supplied public NPB PBP corpus; acquisition/PIT/OOS are separate gates.",
    ),
    AdapterSpec(
        "npb_hawkeye_npbplus",
        "research.npb_tracking_source",
        "normalize_tracking_frame",
        "normalizer",
        False,
        "Normalizes approved tracking payloads; direct NPB+ / Hawk-Eye collection is not claimed.",
    ),
    AdapterSpec(
        "asian_games_baseball",
        "research.asian_games_baseball",
        "fetch_schedule",
        "collector",
        True,
        "Official organizer/BFJ schedule evidence only; prediction eligibility remains research-only.",
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _module_file(module_path: str) -> Path:
    return _repo_root() / (module_path.replace(".", "/") + ".py")


def adapter_specs() -> tuple[AdapterSpec, ...]:
    return ADAPTERS


def adapter_spec(source_id: str) -> AdapterSpec | None:
    for spec in ADAPTERS:
        if spec.source_id == source_id:
            return spec
    return None


def adapter_present(source_id: str) -> bool:
    """Return true only when both the module file and declared symbol exist."""
    spec = adapter_spec(source_id)
    if spec is None or not _module_file(spec.module_path).is_file():
        return False
    try:
        module = import_module(spec.module_path)
        return callable(getattr(module, spec.symbol))
    except Exception:
        return False


def adapter_metadata(source_id: str) -> dict[str, Any]:
    spec = adapter_spec(source_id)
    if spec is None:
        return {
            "mapped": False,
            "implemented": False,
            "module_path": None,
            "symbol": None,
            "kind": None,
            "collection_ready": False,
            "notes": "No concrete adapter contract is registered.",
        }
    file_present = _module_file(spec.module_path).is_file()
    implemented = adapter_present(source_id)
    return asdict(spec) | {
        "mapped": True,
        "implemented": implemented,
        "module_file_present": file_present,
        "module_file": str(_module_file(spec.module_path).relative_to(_repo_root())),
    }


def audit_adapters() -> dict[str, Any]:
    registered_ids = {s.source_id for s in SOURCES}
    mapped_ids = {s.source_id for s in ADAPTERS}
    unknown_mappings = sorted(mapped_ids - registered_ids)
    rows: list[dict[str, Any]] = []
    for source in SOURCES:
        meta = adapter_metadata(source.source_id)
        rows.append(
            {
                "source_id": source.source_id,
                "adapter_state": "IMPLEMENTED" if meta["implemented"] else "UNWIRED",
                "mapped": bool(meta["mapped"]),
                "module_path": meta["module_path"],
                "symbol": meta["symbol"],
                "kind": meta["kind"],
                "collection_ready": bool(meta["collection_ready"]),
                "notes": meta["notes"],
            }
        )
    return {
        "source_count": len(rows),
        "mapped_count": sum(r["mapped"] for r in rows),
        "implemented_count": sum(r["adapter_state"] == "IMPLEMENTED" for r in rows),
        "unwired_count": sum(r["adapter_state"] == "UNWIRED" for r in rows),
        "collection_ready_count": sum(r["collection_ready"] for r in rows),
        "unknown_adapter_mappings": unknown_mappings,
        "sources": rows,
        "promotion_rule": (
            "IMPLEMENTED adapter presence is not PIT/OOS/production evidence; "
            "PIT and chronological OOS must pass independently before advancement."
        ),
    }


def main() -> int:
    import json

    print(json.dumps(audit_adapters(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
