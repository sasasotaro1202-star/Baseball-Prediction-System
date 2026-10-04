    eligible, reasons = eligibility_gate(
        availability=availability,
        required_data_ok=bool(row.get("required_data_ok", True)),
        feature_complete=bool(row.get("feature_complete", True)),
        model_available=bool(row.get("model_available", True)),
        calibration_available=bool(row.get("calibration_available", True)),
        production=production,
    )
    if not eligible:
        return {"eligible": False, "reasons": reasons, "event_id": availability.event_id}

    raw = _validate_final_probabilities(probability_fn(row), availability.league)
    calibrated = calibrate_fn(raw) if calibrate_fn is not None else raw
    probabilities = _validate_final_probabilities(calibrated, availability.league)

    now = datetime.now(timezone.utc)
    if now < _ts(availability.prediction_cutoff):
        raise ValueError("prediction cannot be created before its declared cutoff")

    derived = _derive_score_outputs(row)
    _assert_generated_outputs_are_authoritative(row, derived)
    score_candidates = _validate_score_candidates(derived.get("score_candidates", row.get("score_candidates")))

    total_line = row.get("total_runs_line")
    if total_line is not None:
        total_line = float(total_line)
        if not math.isfinite(total_line) or total_line < 0 or abs(total_line * 2 - round(total_line * 2)) > 1e-9:
            raise ValueError("total_runs_line must be finite, non-negative, integer or half-point")

    supplied_low = row.get("low_probability")
    supplied_high = row.get("high_probability")
    low = _optional_probability(derived.get("low_probability", supplied_low), "low_probability")
    high = _optional_probability(derived.get("high_probability", supplied_high), "high_probability")
    if (low is None) != (high is None):
        raise ValueError("low_probability and high_probability must be supplied together")
    if low is not None and abs(low + high - 1.0) > 1e-8:
        raise ValueError("low/high probabilities must sum to 1")
    if row.get("home_run_lambda") is not None and row.get("away_run_lambda") is not None and (not score_candidates or low is None):
        raise ValueError("run-distribution inputs require complete score and Low/High outputs")

    pid = make_prediction_id(availability.event_id, availability.prediction_cutoff, model_version, git_commit)
    record = PredictionRecord(
        prediction_id=pid, event_id=availability.event_id, league=availability.league,
        prediction_cutoff=availability.prediction_cutoff, prediction_created_at=now.isoformat(),
        home_team=availability.home_team, away_team=availability.away_team,
        home_starter=availability.home_starter, away_starter=availability.away_starter,
        probabilities=probabilities, score_candidates=score_candidates,
        low_probability=low, high_probability=high, total_runs_line=total_line,
        confidence=_optional_probability(row.get("confidence"), "confidence"),
        volatility=_optional_probability(row.get("volatility"), "volatility"),
        model_version=model_version, feature_version=feature_version,
        calibration_version=calibration_version, git_commit=git_commit,
        data_snapshot_id=data_snapshot_id,
        competition_key=(
            str(row.get("competition_key") or "").strip()
            or str(row.get("competition") or "").strip()
            or str(availability.league).strip()
        ),
        competition_stage=(
            str(row.get("competition_stage") or "").strip()
            or None
        ),
        season_type=(
            str(row.get("season_type") or "").strip()
            or None
        ),
        game_class=(
            str(row.get("game_class") or "").strip()
            or None
        ),
        competition_classification_status=(
            str(row.get("competition_classification_status") or "").strip()
            or None
        ),
        competition_metadata_source=(
            str(row.get("competition_metadata_source") or "").strip()
            or None
        ),
        competition_metadata_source_field=(
            str(row.get("competition_metadata_source_field") or "").strip()
            or None
        ),
        competition_metadata_source_value=(
            str(row.get("competition_metadata_source_value") or "").strip()
            or None
        ),
        prediction_set=(
            [str(x) for x in row.get("prediction_set")]
            if row.get("prediction_set") is not None else None
        ),
        prediction_set_alpha=(
            float(row.get("prediction_set_alpha"))
            if row.get("prediction_set_alpha") is not None else None
        ),
        prediction_set_method=(
            str(row.get("prediction_set_method") or "").strip()
            or None
        ),
        prediction_set_action=(
            str(row.get("prediction_set_action") or "").strip()
            or None
        ),
        prediction_intelligence=(
            dict(row.get("prediction_intelligence"))
            if isinstance(row.get("prediction_intelligence"), Mapping)
            else row.get("prediction_intelligence")
        ),
    )
    append_prediction(record, log_path)
    return {"eligible": True, "prediction": record}


def run_production_prediction(**kwargs: Any) -> dict[str, Any]:
    """Explicit production entry point that can never fall back to research mode."""
    kwargs["production"] = True
    return run_prediction(**kwargs)