        if checkpoint_valid:
            try:
                existing = pd.read_csv(ck)
            except Exception:
                existing = pd.DataFrame()
                checkpoint_valid = False
        elif ck.exists():
            print(f"[{league}] ignoring stale checkpoint (version mismatch)")
            existing = pd.DataFrame()

        if checkpoint_valid and not existing.empty:
            if "input_fingerprint" not in existing.columns:
                print(f"[{league}] ignoring stale checkpoint (input fingerprints missing)")
                existing = pd.DataFrame()
            else:
                existing_ids = existing["game_id"].astype(str)
                fingerprint_ok = True
                for game_id, fingerprint in existing[["game_id", "input_fingerprint"]].astype(str).itertuples(index=False):
                    if input_fingerprints.get(game_id) != fingerprint:
                        fingerprint_ok = False
                        break
                if not fingerprint_ok:
                    print(f"[{league}] ignoring stale checkpoint (input fingerprint mismatch)")
                    existing = pd.DataFrame()

        completed_ids = set(existing.get("game_id", pd.Series(dtype=str)).astype(str)) if not existing.empty else set()
        all_rows = existing.to_dict("records") if not existing.empty else []
        start = max(MIN_TRAIN, int(len(X) * 0.25))
        next_heartbeat = time.monotonic()
        total_blocks = int(math.ceil(max(0, len(X) - start) / max(RETRAIN_EVERY, 1)))
        block_number = 0
        for bstart in range(start, len(X), RETRAIN_EVERY):
            block_number += 1
            bend = min(len(X), bstart + RETRAIN_EVERY)
            block_ids = set(meta.iloc[bstart:bend]["game_id"].astype(str))
            now_mono = time.monotonic()
            if now_mono >= next_heartbeat:
                elapsed = max(0.0, time.time() - self.started_at)
                remaining = max(0.0, self.time_budget_sec - elapsed)
                print(
                    f"[{league} HEARTBEAT] block={block_number}/{total_blocks} "
                    f"range={bstart}:{bend} completed={len(completed_ids)}/{max(1, len(X)-start)} "
                    f"elapsed={elapsed:.0f}s budget_remaining={remaining:.0f}s",
                    flush=True,
                )
                next_heartbeat = now_mono + self.heartbeat_sec
            if block_ids and block_ids.issubset(completed_ids):
                print(f"[{league}] resume skip block {bstart}:{bend} ({len(block_ids)} games already checkpointed)")
                continue
            if time.time() - self.started_at >= self.time_budget_sec:
                self.audit.append({"type":"time_budget","league":league,"bstart":int(bstart),"budget_sec":self.time_budget_sec})
                print(f"[{league}] time budget reached; stopping walk-forward with resumable checkpoint")
                raise TimeoutError(
                    f"walk-forward incomplete: {league} time budget reached before block {bstart}:{bend}"
                )
            try:
                print(f"[{league} HEARTBEAT] fitting block={block_number}/{total_blocks} train={bstart} eval={bend-bstart}", flush=True)
                fitted, val_scores, best_name = self.fit_ensemble(X.iloc[:bstart], y[:bstart], league)
                if not fitted: raise RuntimeError("ensemble fitting failed")
                name = "Ensemble(" + "+".join(x[2] for x in fitted) + ")"
                p = self.ensemble_proba(fitted, X.iloc[bstart:bend], league)
                score_fit = self.fit_score_ensemble(X.iloc[:bstart], games.iloc[:bstart]["home_score"].astype(float).values, games.iloc[:bstart]["away_score"].astype(float).values, league)
            except TimeoutError:
                raise
            except Exception as e:
                print(f"[{league}] block {bstart}: model failure {e}")
                self.audit.append({
                    "type": "walkforward_block_failure",
                    "league": league,
                    "bstart": int(bstart),
                    "bend": int(bend),
                    "error": f"{type(e).__name__}: {e}",
                })
                raise RuntimeError(
                    f"walk-forward incomplete: {league} block {bstart}:{bend} failed"
                ) from e
            block_rows = []
            for j, idx in enumerate(range(bstart, bend)):
                r = meta.iloc[idx]
                if str(r["game_id"]) in completed_ids:
                    continue
                prob = p[j]
                pred = int(np.argmax(prob))
                actual = int(y[idx])
                target = np.zeros(len(prob)); target[actual] = 1
                ll = float(-math.log(max(prob[actual], 1e-12)))