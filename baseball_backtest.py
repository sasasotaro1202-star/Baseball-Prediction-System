        """Normalize public NPB PBP releases through the canonical PIT-safe adapter."""
        from data.npb_pbp_adapter import normalize_pbp_frame
        return normalize_pbp_frame(raw, data_dir=self.data_dir)

    def load_npb_pbp(self) -> pd.DataFrame:
        """Load all available NPB seasons from the massive multi-season collector.

        Preferred input is data/npb_multi_source_games_all.csv. If it is absent,
        season partitions under data/npb_games/*.csv are combined. Legacy monthly
        PBP files remain a fallback, but are never combined with an overlapping
        multi-source season file.
        """
        aggregate = self.data_dir / "npb_multi_source_games_all.csv"
        season_files = sorted((self.data_dir / "npb_games").glob("*_multi_source_pbp.csv")) if (self.data_dir / "npb_games").exists() else []
        cached_pbp_dir = self.data_dir / "pbp"
        cached_pbp_files = (
            sorted(cached_pbp_dir.glob("*_pbp.csv"))
            if cached_pbp_dir.exists()
            else []
        )
        if aggregate.exists():
            files = [aggregate]
        elif season_files:
            files = season_files
        elif cached_pbp_files:
            # Shared PIT-screening cache used by the Game-Script and daily
            # research lanes. Those workflows validate each asset against the
            # upstream release manifest before model fitting.
            files = cached_pbp_files
        else:
            files = sorted(self.data_dir.glob("*_multi_source_pbp.csv"))
            files += sorted(self.data_dir.glob("*_pbp.csv"))
            files += sorted(ROOT.glob("*_pbp.csv"))
            files = [f for f in dict.fromkeys(files) if f.name not in {aggregate.name, "2026_multi_source_pbp.csv"}]
        if not files:
            raise FileNotFoundError("NPB data not found. Run npb_multi_source_massive_resumable.py first.")
        chunks=[]
        for f in files:
            if time.time()-self.started_at >= self.time_budget_sec:
                raise TimeoutError("time budget reached during NPB loading")
            try:
                df=pd.read_csv(f,low_memory=False)
                df.columns=[str(c).strip() for c in df.columns]
                if "game_id" not in df.columns:
                    continue
                chunks.append(df)
                print(f"[NPB LOAD] {f} rows={len(df)}")
            except Exception as e:
                self.audit.append({"type":"npb_load_error","file":str(f),"error":str(e)})
                print(f"[NPB SKIP] {f}: {e}")
        if not chunks: raise RuntimeError("No readable NPB data files.")
        raw=pd.concat(chunks,ignore_index=True,sort=False)
        out=self._normalize_npb_pbp(raw)
        if "game_id" in out:
            # Preserve the full play-by-play chronology. Collapsing to one row per
            # game here destroys score reconstruction and starter evidence because
            # aggregate_npb_games needs all plays inside each game.