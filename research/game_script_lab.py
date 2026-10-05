from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ColumnMap:
    game_id: str
    date: str
    home: str
    away: str
    home_score: str
    away_score: str
    sequence: str
    serial: str | None
    half: str | None


def _norm(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9一-龯ぁ-んァ-ン]+",
        "",
        str(value or "").strip().lower(),
    )


def _first_col(columns: Iterable[str], names: Iterable[str]) -> str | None:
    mapping = {_norm(column): str(column) for column in columns}
    return next(
        (mapping[_norm(name)] for name in names if _norm(name) in mapping),
        None,
    )


def resolve_columns(frame: pd.DataFrame) -> ColumnMap:
    specs = {
        "game_id": ["game_id", "GameID"],
        "date": ["game_date", "GameDate", "date"],
        "home": ["home_team_name", "H_NameS", "home"],
        "away": ["away_team_name", "V_NameS", "away"],
        "home_score": [
            "home_total_runs",
            "H_R",
            "HScore",
            "home_score",
        ],
        "away_score": [
            "away_total_runs",
            "V_R",
            "VScore",
            "away_score",
        ],
        "sequence": [
            "PlayInfo_SeqNo",
            "play_id",
            "ID",
            "page",
            "row_order",
        ],
    }
    resolved = {
        key: _first_col(frame.columns, names)
        for key, names in specs.items()
    }
    missing = [key for key, value in resolved.items() if not value]
    if missing:
        raise ValueError(
            "PBP schema missing required columns: " + ",".join(missing)
        )

    return ColumnMap(
        **resolved,
        serial=_first_col(
            frame.columns,
            [
                "fiveDigitSerialNumber",
                "FiveDigitSerialNumber",
                "serialNumber",
                "serial",
                "play_serial",
            ],
        ),
        half=_first_col(
            frame.columns,
            ["TB", "half", "Half", "half_inning"],
        ),
    )


def _parse_state(
    row: pd.Series,
    columns: ColumnMap,
) -> tuple[int | None, str | None]:
    if columns.serial:
        raw = str(row.get(columns.serial, "")).strip()
        match = re.search(r"(\d{1,2}).{0,1}([TBtb12])", raw)
        if match:
            return (
                int(match.group(1)),
                "T"
                if match.group(2).upper() in {"T", "1"}
                else "B",
            )

        compact = re.sub(r"[^0-9A-Za-z]", "", raw)
        if (
            len(compact) >= 3
            and compact[:2].isdigit()
            and compact[2] in "TBtb12"
        ):
            return (
                int(compact[:2]),
                "T" if compact[2].upper() in {"T", "1"} else "B",
            )

    if columns.half:
        raw = str(row.get(columns.half, "")).strip()
        match = re.search(r"(\d{1,2}).*([TBtb12])", raw)
        if match:
            return (
                int(match.group(1)),
                "T"
                if match.group(2).upper() in {"T", "1"}
                else "B",
            )

        match = re.search(r"([TBtb12])", raw)
        if match:
            return (
                None,
                "T" if match.group(1).upper() in {"T", "1"} else "B",
            )

    return None, None


def load_pbp(data_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    aggregate = data_dir / "npb_multi_source_games_all.csv"
    if aggregate.exists():
        paths = [aggregate]
    else:
        season_dir = data_dir / "npb_games"
        paths = (
            sorted(season_dir.glob("*_multi_source_pbp.csv"))
            if season_dir.exists()
            else []
        )
        if not paths:
            paths = sorted(data_dir.glob("*_pbp.csv"))

    if not paths:
        raise FileNotFoundError("No NPB PBP CSV found.")

    frames: list[pd.DataFrame] = []
    used: list[Path] = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        if not frame.empty:
            frames.append(frame)
            used.append(path)

    if not frames:
        raise RuntimeError("PBP files are empty.")

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    ), used


def build_half_innings(
    raw: pd.DataFrame,
    columns: ColumnMap,
) -> pd.DataFrame:
    work = pd.DataFrame(
        {
            "_gid": raw[columns.game_id].astype(str),
            "_date": pd.to_datetime(
                raw[columns.date],
                errors="coerce",
                utc=True,
            ),
            "_seq": pd.to_numeric(
                raw[columns.sequence],
                errors="coerce",
            ),
            "_home": raw[columns.home].astype(str).str.strip(),
            "_away": raw[columns.away].astype(str).str.strip(),
            "_hs": pd.to_numeric(
                raw[columns.home_score],
                errors="coerce",
            ),
            "_as": pd.to_numeric(
                raw[columns.away_score],
                errors="coerce",
            ),
        },
        index=raw.index,
    )
    work = work.dropna(
        subset=["_gid", "_date", "_seq"]
    ).sort_values(
        ["_date", "_gid", "_seq"],
        kind="mergesort",
    )

    rows: list[dict[str, Any]] = []

    for game_id, game in work.groupby("_gid", sort=False):
        groups: list[
            tuple[tuple[int, str], list[tuple[Any, ...]]]
        ] = []
        current: tuple[int, str] | None = None
        bucket: list[tuple[Any, ...]] = []

        for _, row in game.iterrows():
            inning, half = _parse_state(row, columns)
            home_score = (
                float(row["_hs"])
                if np.isfinite(row["_hs"])
                else np.nan
            )
            away_score = (
                float(row["_as"])
                if np.isfinite(row["_as"])
                else np.nan
            )
            if inning is None or half is None:
                continue
            if not (
                np.isfinite(home_score)
                and np.isfinite(away_score)
            ):
                continue

            key = (inning, half)
            if current is not None and key != current:
                groups.append((current, bucket))
                bucket = []
            current = key
            bucket.append(
                (
                    row["_date"],
                    row["_home"],
                    row["_away"],
                    home_score,
                    away_score,
                )
            )

        if bucket:
            assert current is not None
            groups.append((current, bucket))

        previous_home = 0.0
        previous_away = 0.0

        for (inning, half), bucket_rows in groups:
            last = bucket_rows[-1]
            end_home = last[3]
            end_away = last[4]

            if (
                end_home < previous_home
                or end_away < previous_away
            ):
                raise ValueError(
                    f"negative score delta in "
                    f"game={game_id} inning={inning}{half}"
                )

            rows.append(
                {
                    "game_id": str(game_id),
                    "date": last[0],
                    "home": last[1],
                    "away": last[2],
                    "inning": int(inning),
                    "half": half,
                    "start_home_score": previous_home,
                    "start_away_score": previous_away,
                    "end_home_score": end_home,
                    "end_away_score": end_away,
                }
            )

            previous_home = end_home
            previous_away = end_away

    if not rows:
        raise ValueError(
            "No usable inning/half states. Refusing silent fallback."
        )

    half = pd.DataFrame(rows)
    half["home_runs"] = (
        half["end_home_score"] - half["start_home_score"]
    )
    half["away_runs"] = (
        half["end_away_score"] - half["start_away_score"]
    )
    half["score_diff_home"] = (
        half["start_home_score"] - half["start_away_score"]
    )
    half["offense_team"] = np.where(
        half.half.eq("B"),
        half.home,
        half.away,
    )
    half["defense_team"] = np.where(
        half.half.eq("B"),
        half.away,
        half.home,
    )
    half["runs"] = np.where(
        half.half.eq("B"),
        half.home_runs,
        half.away_runs,
    )
    half["score_diff_for"] = np.where(
        half.half.eq("B"),
        half.score_diff_home,
        -half.score_diff_home,
    )
    return half


@dataclass
class TransitionModel:
    league_mean: float
    inning_half: dict[str, float]
    offense: dict[str, float]
    defense: dict[str, float]
    score_state: dict[str, float]
    fit_game_ids: tuple[str, ...]

    def rate(
        self,
        offense_team: str,
        defense_team: str,
        inning: int,
        half: str,
        score_diff_for: int,
    ) -> float:
        base = max(
            self.inning_half.get(
                f"{inning}:{half}",
                self.league_mean,
            ),
            0.03,
        )
        league_mean = max(self.league_mean, 0.03)

        offense_term = math.log(
            max(
                self.offense.get(
                    offense_team,
                    league_mean,
                ),
                0.03,
            )
            / league_mean
        )
        defense_term = math.log(
            max(
                self.defense.get(
                    defense_team,
                    league_mean,
                ),
                0.03,
            )
            / league_mean
        )
        state_key = str(
            int(np.clip(score_diff_for, -3, 3))
        )
        state_term = math.log(
            max(
                self.score_state.get(
                    state_key,
                    league_mean,
                ),
                0.03,
            )
            / league_mean
        )

        return float(
            np.clip(
                base
                * math.exp(
                    0.60 * offense_term
                    + 0.55 * defense_term
                    + 0.20 * state_term
                ),
                0.03,
                4.0,
            )
        )


def fit_transition_model(
    halves: pd.DataFrame,
    train_games: set[str],
    prior_strength: float = 12.0,
) -> TransitionModel:
    train = halves[
        halves.game_id.astype(str).isin(train_games)
    ].copy()
    if train.empty:
        raise ValueError("No prior-game rows for model fit.")

    league_mean = float(train.runs.mean())

    def smooth(group: pd.DataFrame) -> float:
        return float(
            (
                group.runs.sum()
                + prior_strength * league_mean
            )
            / (len(group) + prior_strength)
        )

    inning_half = {
        f"{int(key[0])}:{key[1]}": smooth(group)
        for key, group in train.groupby(
            ["inning", "half"]
        )
    }
    offense = {
        str(key): smooth(group)
        for key, group in train.groupby("offense_team")
    }
    defense = {
        str(key): smooth(group)
        for key, group in train.groupby("defense_team")
    }

    train["_score_bucket"] = np.clip(
        train.score_diff_for.round().astype(int),
        -3,
        3,
    )
    score_state = {
        str(int(key)): smooth(group)
        for key, group in train.groupby("_score_bucket")
    }

    return TransitionModel(
        league_mean=league_mean,
        inning_half=inning_half,
        offense=offense,
        defense=defense,
        score_state=score_state,
        fit_game_ids=tuple(sorted(train_games)),
    )


def _chronological_games(
    halves: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for game_id, game in halves.groupby(
        "game_id",
        sort=False,
    ):
        ordered = game.assign(
            _half_order=game["half"].map(
                {"T": 0, "B": 1}
            ).fillna(9)
        ).sort_values(
            ["inning", "_half_order"],
            kind="mergesort",
        )
        last = ordered.iloc[-1]
        rows.append(
            {
                "game_id": str(game_id),
                "date": last.date,
                "home": last.home,
                "away": last.away,
                "actual_home": float(
                    last.end_home_score
                ),
                "actual_away": float(
                    last.end_away_score
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["date", "game_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def game_table(halves: pd.DataFrame) -> pd.DataFrame:
    result = _chronological_games(halves)
    result["actual_class"] = np.select(
        [
            result.actual_home > result.actual_away,
            result.actual_home == result.actual_away,
        ],
        [0, 1],
        default=2,
    ).astype(int)
    return result


def baseline_probs(
    games: pd.DataFrame,
    before: int,
    prior_strength: float = 12.0,
) -> np.ndarray:
    labels = games.iloc[:before].actual_class.to_numpy(
        dtype=int
    )
    if not len(labels):
        return np.ones(3) / 3.0

    counts = np.bincount(
        labels,
        minlength=3,
    ).astype(float)
    return (
        counts + prior_strength / 3.0
    ) / (
        counts.sum() + prior_strength
    )


def simulate(
    model: TransitionModel,
    home: str,
    away: str,
    simulations: int,
    seed: int,
    max_innings: int = 12,
) -> dict[str, Any]:
    if simulations < 1:
        raise ValueError("simulations must be >= 1")
    if max_innings < 9:
        raise ValueError(
            "max_innings must be >= 9"
        )

    rng = np.random.default_rng(seed)
    home_runs = np.zeros(
        simulations,
        dtype=np.int16,
    )
    away_runs = np.zeros(
        simulations,
        dtype=np.int16,
    )
    active = np.ones(
        simulations,
        dtype=bool,
    )

    for inning in range(1, max_innings + 1):
        for half in ("T", "B"):
            active_idx = np.flatnonzero(active)
            if len(active_idx) == 0:
                break

            if half == "T":
                offense_team = away
                defense_team = home
                score_diff = away_runs - home_runs
            else:
                offense_team = home
                defense_team = away
                score_diff = home_runs - away_runs

            base_rate = model.rate(
                offense_team,
                defense_team,
                inning,
                half,
                0,
            )
            score_bucket = np.clip(
                score_diff,
                -3,
                3,
            ).astype(int)
            rates = np.full(
                len(active_idx),
                base_rate,
                dtype=float,
            )
            league_mean = max(
                model.league_mean,
                0.03,
            )

            for bucket in range(-3, 4):
                local = (
                    score_bucket[active_idx]
                    == bucket
                )
                if not local.any():
                    continue

                state_mean = model.score_state.get(
                    str(bucket),
                    model.league_mean,
                )
                state_adjustment = math.exp(
                    0.20
                    * math.log(
                        max(state_mean, 0.03)
                        / league_mean
                    )
                )
                rates[local] = np.clip(
                    base_rate * state_adjustment,
                    0.03,
                    4.0,
                )

            scored = rng.poisson(rates).astype(
                np.int16
            )

            if half == "T":
                away_runs[active_idx] += scored
                if inning >= 9:
                    active &= ~(
                        home_runs > away_runs
                    )
            else:
                home_runs[active_idx] += scored
                if inning >= 9:
                    active &= (
                        home_runs == away_runs
                    )

    result_class = np.empty(
        simulations,
        dtype=int,
    )
    result_class[home_runs > away_runs] = 0
    result_class[
        home_runs == away_runs
    ] = 1
    result_class[
        home_runs < away_runs
    ] = 2

    probabilities = (
        np.bincount(
            result_class,
            minlength=3,
        )
        / simulations
    )

    score_counts: dict[str, int] = {}
    for home_score, away_score in zip(
        home_runs.tolist(),
        away_runs.tolist(),
    ):
        key = f"{home_score}-{away_score}"
        score_counts[key] = (
            score_counts.get(key, 0) + 1
        )

    top4 = sorted(
        score_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[:4]

    clipped = np.clip(
        probabilities,
        1e-9,
        1.0,
    )
    entropy = float(
        -np.sum(
            clipped * np.log(clipped)
        )
    )

    return {
        "proba": probabilities.tolist(),
        "home_mean": float(home_runs.mean()),
        "away_mean": float(away_runs.mean()),
        "low_prob": float(
            np.mean(
                (home_runs + away_runs) <= 6
            )
        ),
        "top4": [
            {
                "score": score,
                "prob": float(
                    count / simulations
                ),
            }
            for score, count in top4
        ],
        "entropy": entropy,
    }


def logloss(
    y: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    if not len(y):
        return float("nan")
    return float(
        -np.mean(
            np.log(
                np.clip(
                    probabilities[
                        np.arange(len(y)),
                        y,
                    ],
                    1e-9,
                    1.0,
                )
            )
        )
    )


def brier(
    y: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    if not len(y):
        return float("nan")
    return float(
        np.mean(
            np.sum(
                (
                    probabilities
                    - np.eye(
                        probabilities.shape[1]
                    )[y]
                )
                ** 2,
                axis=1,
            )
        )
    )


def ece(
    y: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    if not len(y):
        return float("nan")

    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (
        predicted == y
    ).astype(float)
    value = 0.0

    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        if index < bins - 1:
            mask = (
                (confidence >= low)
                & (confidence < high)
            )
        else:
            mask = (
                (confidence >= low)
                & (confidence <= high)
            )
        if mask.any():
            value += float(
                mask.mean()
            ) * abs(
                float(correct[mask].mean())
                - float(confidence[mask].mean())
            )

    return float(value)


def _sha256_files(paths: list[Path]) -> str:
    digest = "".join(
        hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths
    )
    return hashlib.sha256(
        digest.encode("utf-8")
    ).hexdigest()


def _atomic_json(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_cycle(
    data_dir: Path,
    out_dir: Path,
    config: dict[str, Any],
    git_commit: str,
) -> dict[str, Any]:
    raw, input_files = load_pbp(data_dir)
    columns = resolve_columns(raw)
    halves = build_half_innings(
        raw,
        columns,
    )
    games = game_table(halves)

    min_train = int(
        config.get(
            "min_train_games",
            90,
        )
    )
    min_oos = int(
        config.get(
            "min_oos_games",
            60,
        )
    )
    retrain_every = int(
        config.get(
            "retrain_every_games",
            60,
        )
    )
    simulations = int(
        os.getenv(
            "GAME_SCRIPT_SIMULATIONS",
            config.get(
                "simulations_per_game",
                2500,
            ),
        )
    )
    max_innings = int(
        config.get(
            "regulation_innings",
            12,
        )
    )

    if len(games) < min_train + min_oos:
        raise RuntimeError(
            "Insufficient chronological OOS games: "
            f"{len(games)}"
        )

    data_hash = _sha256_files(
        input_files
    )
    fingerprint_payload = {
        "schema": "game-script-lab-v1",
        "git_commit": git_commit,
        "data_hash": data_hash,
        "columns": asdict(columns),
        "config": config,
        "simulations": simulations,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    predictions_path = (
        out_dir
        / "game_script_oos.jsonl"
    )
    checkpoint_path = (
        out_dir
        / "checkpoint.json"
    )

    previous: list[dict[str, Any]] = []
    try:
        checkpoint = (
            json.loads(
                checkpoint_path.read_text(
                    encoding="utf-8"
                )
            )
            if checkpoint_path.exists()
            else {}
        )
        if (
            checkpoint.get(
                "fingerprint"
            )
            == fingerprint
            and predictions_path.exists()
        ):
            previous = [
                json.loads(line)
                for line in predictions_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
    except Exception:
        previous = []

    by_game = {
        str(row["game_id"]): row
        for row in previous
        if row.get("fingerprint")
        == fingerprint
    }
    model: TransitionModel | None = None

    with predictions_path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        for index, game in games.iterrows():
            game_id = str(game.game_id)
            if (
                index < min_train
                or game_id in by_game
            ):
                continue

            if (
                model is None
                or (
                    index - min_train
                )
                % max(1, retrain_every)
                == 0
            ):
                train_game_ids = set(
                    games.iloc[:index]
                    .game_id
                    .astype(str)
                )
                if game_id in train_game_ids:
                    raise RuntimeError(
                        "PIT violation: current "
                        "game entered training"
                    )
                model = fit_transition_model(
                    halves,
                    train_game_ids,
                    float(
                        config.get(
                            "prior_strength",
                            12.0,
                        )
                    ),
                )

            baseline = baseline_probs(
                games,
                index,
                float(
                    config.get(
                        "outcome_prior_strength",
                        12.0,
                    )
                ),
            )
            simulation = simulate(
                model,
                str(game.home),
                str(game.away),
                simulations,
                int(
                    config.get(
                        "seed",
                        42,
                    )
                ) + index,
                max_innings,
            )
            candidate = np.asarray(
                simulation["proba"],
                dtype=float,
            )

            record = {
                "schema_version": (
                    "game-script-oos-v1"
                ),
                "status": "EXECUTED",
                "research_only": True,
                "game_id": game_id,
                "game_time": pd.Timestamp(
                    game.date
                ).isoformat(),
                "prediction_cutoff": (
                    "HISTORICAL_PREGAME_"
                    "CUTOFF_UNVERIFIABLE"
                ),
                "model_version": (
                    "game-script-"
                    "state-transition-v1"
                ),
                "git_commit": git_commit,
                "dataset_hash": data_hash,
                "fingerprint": fingerprint,
                "fit_game_id_count": len(
                    model.fit_game_ids
                ),
                "pit_status": (
                    "PASS_FOR_"
                    "TRANSITION_LEARNING"
                ),
                "production_pregame_pit": (
                    "UNVERIFIABLE"
                ),
                "baseline_proba": (
                    baseline.tolist()
                ),
                "candidate_proba": (
                    candidate.tolist()
                ),
                "candidate_entropy": (
                    simulation["entropy"]
                ),
                "predictability_score": float(
                    max(
                        0.0,
                        1.0
                        - simulation["entropy"]
                        / math.log(3),
                    )
                ),
                "simulations": simulations,
                "sim_home_mean": (
                    simulation["home_mean"]
                ),
                "sim_away_mean": (
                    simulation["away_mean"]
                ),
                "low_prob": (
                    simulation["low_prob"]
                ),
                "top4": simulation["top4"],
                "actual_home": float(
                    game.actual_home
                ),
                "actual_away": float(
                    game.actual_away
                ),
                "actual_class": int(
                    game.actual_class
                ),
            }

            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
            by_game[game_id] = record

            checkpoint_every = int(
                config.get(
                    "checkpoint_every_games",
                    10,
                )
            )
            if (
                checkpoint_every > 0
                and len(by_game)
                % checkpoint_every
                == 0
            ):
                _atomic_json(
                    checkpoint_path,
                    {
                        "schema_version": (
                            "game-script-"
                            "checkpoint-v1"
                        ),
                        "fingerprint": fingerprint,
                        "completed_game_ids": sorted(
                            by_game
                        ),
                        "updated_at_utc": (
                            pd.Timestamp.utcnow()
                            .isoformat()
                        ),
                        "status": "CHECKPOINTED",
                    },
                )

    rows = sorted(
        by_game.values(),
        key=lambda row: (
            row["game_time"],
            row["game_id"],
        ),
    )
    if not rows:
        raise RuntimeError(
            "No OOS rows were generated."
        )

    y = np.asarray(
        [row["actual_class"] for row in rows],
        dtype=int,
    )
    baseline_probabilities = np.asarray(
        [
            row["baseline_proba"]
            for row in rows
        ],
        dtype=float,
    )
    candidate_probabilities = np.asarray(
        [
            row["candidate_proba"]
            for row in rows
        ],
        dtype=float,
    )

    candidate_home_mae = float(
        np.mean(
            [
                abs(
                    row["sim_home_mean"]
                    - row["actual_home"]
                )
                for row in rows
            ]
        )
    )
    candidate_away_mae = float(
        np.mean(
            [
                abs(
                    row["sim_away_mean"]
                    - row["actual_away"]
                )
                for row in rows
            ]
        )
    )
    candidate_total_mae = float(
        np.mean(
            [
                abs(
                    (
                        row["sim_home_mean"]
                        + row["sim_away_mean"]
                    )
                    - (
                        row["actual_home"]
                        + row["actual_away"]
                    )
                )
                for row in rows
            ]
        )
    )
    low_high_accuracy = float(
        np.mean(
            [
                (
                    row["low_prob"] >= 0.5
                )
                == (
                    (
                        row["actual_home"]
                        + row["actual_away"]
                    )
                    <= 6
                )
                for row in rows
            ]
        )
    )

    baseline_metrics = {
        "LogLoss": logloss(
            y,
            baseline_probabilities,
        ),
        "Brier": brier(
            y,
            baseline_probabilities,
        ),
        "Accuracy": float(
            np.mean(
                baseline_probabilities.argmax(
                    axis=1
                )
                == y
            )
        ),
        "ECE": ece(
            y,
            baseline_probabilities,
        ),
    }
    candidate_metrics = {
        "LogLoss": logloss(
            y,
            candidate_probabilities,
        ),
        "Brier": brier(
            y,
            candidate_probabilities,
        ),
        "Accuracy": float(
            np.mean(
                candidate_probabilities.argmax(
                    axis=1
                )
                == y
            )
        ),
        "ECE": ece(
            y,
            candidate_probabilities,
        ),
    }

    summary = {
        "schema_version": (
            "game-script-cycle-v1"
        ),
        "status": "EXECUTED",
        "research_only": True,
        "evidence_level": (
            "E4_LOCAL_OOS"
        ),
        "production_changed": False,
        "promotion_by_workflow": False,
        "decision": (
            "HOLD_PENDING_FULL_SYSTEM_GATE"
        ),
        "git_commit": git_commit,
        "dataset_hash": data_hash,
        "fingerprint": fingerprint,
        "game_rows": int(len(games)),
        "half_inning_rows": int(len(halves)),
        "oos_rows": int(len(rows)),
        "input_files": [
            str(path)
            for path in input_files
        ],
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta": {
            "LogLoss_candidate_minus_baseline": (
                candidate_metrics["LogLoss"]
                - baseline_metrics["LogLoss"]
            ),
            "Brier_candidate_minus_baseline": (
                candidate_metrics["Brier"]
                - baseline_metrics["Brier"]
            ),
            "Accuracy_candidate_minus_baseline": (
                candidate_metrics["Accuracy"]
                - baseline_metrics["Accuracy"]
            ),
            "ECE_candidate_minus_baseline": (
                candidate_metrics["ECE"]
                - baseline_metrics["ECE"]
            ),
        },
        "pit_audit": {
            "status": (
                "PASS_FOR_TRANSITION_LEARNING"
            ),
            "chronological_split": True,
            "same_game_in_training": False,
            "future_event_features_used": False,
            "production_pregame_cutoff": (
                "UNVERIFIABLE"
            ),
        },
        "score_metrics": {
            "candidate_home_run_mae": (
                candidate_home_mae
            ),
            "candidate_away_run_mae": (
                candidate_away_mae
            ),
            "candidate_total_run_mae": (
                candidate_total_mae
            ),
            "low_high_accuracy": (
                low_high_accuracy
            ),
        },
        "generated_at_utc": (
            pd.Timestamp.utcnow().isoformat()
        ),
    }

    _atomic_json(
        out_dir / "game_script_summary.json",
        summary,
    )
    _atomic_json(
        checkpoint_path,
        {
            "schema_version": (
                "game-script-checkpoint-v1"
            ),
            "fingerprint": fingerprint,
            "completed_game_ids": sorted(
                by_game
            ),
            "updated_at_utc": (
                pd.Timestamp.utcnow()
                .isoformat()
            ),
            "status": "COMPLETE",
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="data",
    )
    parser.add_argument(
        "--output-dir",
        default="results/game_script_lab",
    )
    parser.add_argument(
        "--config",
        default="config/game_script_lab.json",
    )
    parser.add_argument(
        "--git-commit",
        default=os.getenv(
            "GITHUB_SHA",
            "unknown",
        ),
    )
    args = parser.parse_args()
    config = json.loads(
        Path(args.config).read_text(
            encoding="utf-8"
        )
    )
    result = run_cycle(
        Path(args.data_dir),
        Path(args.output_dir),
        config,
        args.git_commit,
    )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
