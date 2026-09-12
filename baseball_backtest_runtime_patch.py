#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Idempotent runtime hardening for NPB collector/backtest integration."""
from __future__ import annotations
import re
from pathlib import Path
P=Path('baseball_backtest.py')
s=P.read_text(encoding='utf-8')
if '# BACKTEST_RUNTIME_HARDENING_V4' in s:
    print('[BACKTEST PATCH] V4 already applied'); raise SystemExit(0)

# Preserve all prior V3 hardening, then replace the score layer with an
# adaptive model that explicitly uses prior-game run environment and validates
# the blend chronologically. This addresses the historical failure mode where
# MLB score lambdas collapsed toward a nearly constant league average.
anchor='    def fit_score_ensemble(self, X: pd.DataFrame, y_home: np.ndarray, y_away: np.ndarray, league: str):\n'
start=s.find(anchor)
if start < 0: raise RuntimeError('score ensemble anchor missing')
end=s.find('    def run_walkforward(self, games: pd.DataFrame, league: str) -> pd.DataFrame:\n', start)
if end < 0: raise RuntimeError('run_walkforward anchor missing')

replacement=r'''    def _score_prior(self, X: pd.DataFrame, league: str) -> Tuple[np.ndarray, np.ndarray]:
        """Pregame structural run prior from strictly historical rolling features.

        The feature builder stores home offense/defense as h_* and away offense/
        defense as a_*. These values are created before the target game is
        updated, so this prior cannot use the target score. Multiple windows are
        combined with shrinkage toward the league environment.
        """
        n=len(X)
        if n == 0: return np.array([]), np.array([])
        def col(name, default):
            if name in X: return pd.to_numeric(X[name],errors='coerce').fillna(default).to_numpy(float)
            return np.full(n,float(default))
        env=col('expected_env', 8.5 if league=='MLB' else 7.0)
        # Offensive and defensive rolling means are the strongest transparent
        # run-level priors available even when player-level data are sparse.
        h_off=np.nanmean(np.vstack([col('h_gf_3',env/2),col('h_gf_5',env/2),col('h_gf_10',env/2),col('h_gf_20',env/2)]),axis=0)
        h_def=np.nanmean(np.vstack([col('a_ga_3',env/2),col('a_ga_5',env/2),col('a_ga_10',env/2),col('a_ga_20',env/2)]),axis=0)
        a_off=np.nanmean(np.vstack([col('a_gf_3',env/2),col('a_gf_5',env/2),col('a_gf_10',env/2),col('a_gf_20',env/2)]),axis=0)
        a_def=np.nanmean(np.vstack([col('h_ga_3',env/2),col('h_ga_5',env/2),col('h_ga_10',env/2),col('h_ga_20',env/2)]),axis=0)
        h=(0.48*h_off+0.42*h_def+0.10*(env/2.0))
        a=(0.48*a_off+0.42*a_def+0.10*(env/2.0))
        # Home advantage is learned only as a bounded prior, not from target
        # outcomes. Starter quality adjusts expected scoring directionally.
        h += 0.025*col('home_adv',1.0)*env
        fip_h=col('hs_fip',4.0); fip_a=col('as_fip',4.0)
        h += np.clip((fip_a-4.0)*0.10,-0.55,0.55)
        a += np.clip((fip_h-4.0)*0.10,-0.55,0.55)
        # Batting power/contact and bullpen fatigue provide small, bounded
        # corrections; the score regressors still carry the main weight.
        h += np.clip(col('offense_power_gap_10',0.0)*2.5,-0.45,0.45)
        a -= np.clip(col('offense_power_gap_10',0.0)*2.5,-0.45,0.45)
        h -= np.clip(col('bullpen_fatigue_diff',0.0)*0.015,-0.30,0.30)
        a += np.clip(col('bullpen_fatigue_diff',0.0)*0.015,-0.30,0.30)
        floor=2.35 if league=='NPB' else 4.15
        h=np.clip(0.78*h+0.22*(env/2.0),0.25,12.0)
        a=np.clip(0.78*a+0.22*(env/2.0),0.25,12.0)
        return h,a

    @staticmethod
    def _nb_nll(y: np.ndarray, mu: np.ndarray, alpha: float) -> float:
        """Negative-binomial NLL with Var(Y)=mu+alpha*mu^2."""
        mu=np.clip(np.asarray(mu,float),1e-4,20.0); y=np.asarray(y,float); alpha=max(float(alpha),1e-6)
        r=1.0/alpha; p=r/(r+mu)
        return float(np.mean(-(np.array([math.lgamma(v+r)-math.lgamma(r)-math.lgamma(v+1.0) for v in y]) + r*np.log(p) + y*np.log(1.0-p))))

    def fit_score_ensemble(self, X: pd.DataFrame, y_home: np.ndarray, y_away: np.ndarray, league: str):
        """Chronological score ensemble with adaptive structural prior and overdispersion.

        Candidate regressors are scored on multiple chronological windows. The
        final mixture also learns how much to trust the structural prior, rather
        than forcing all games toward a constant league mean.
        """
        if len(X) < max(80, MIN_TRAIN // 2): return None
        splits=self._validation_splits(len(X))
        specs=[
            ('Poisson',lambda:PoissonRegressor(alpha=0.05,max_iter=1500)),
            ('HistPoisson',lambda:HistGradientBoostingRegressor(loss='poisson',max_iter=260,learning_rate=0.025,max_leaf_nodes=15,l2_regularization=0.8,random_state=42)),
            ('RFReg',lambda:RandomForestRegressor(n_estimators=240,min_samples_leaf=3,max_features=0.70,random_state=42,n_jobs=-1)),
            ('ExtraTreesReg',lambda:ExtraTreesRegressor(n_estimators=240,min_samples_leaf=3,max_features=0.75,random_state=42,n_jobs=-1)),
        ]
        prior_h,prior_a=self._score_prior(X,league)
        scored=[]
        for name,factory in specs:
            losses=[]
            for cut,val in splits:
                if time.time()-self.started_at>=self.time_budget_sec: break
                try:
                    mh=factory(); ma=factory()
                    self._fit_model(mh,X.iloc[:cut],y_home[:cut],self._sample_weights(cut),league)
                    self._fit_model(ma,X.iloc[:cut],y_away[:cut],self._sample_weights(cut),league)
                    ph=np.clip(mh.predict(X.iloc[cut:cut+val]),0.15,15.0); pa=np.clip(ma.predict(X.iloc[cut:cut+val]),0.15,15.0)
                    nh=self._nb_nll(y_home[cut:cut+val],ph,max(0.02,float(np.var(y_home[:cut])-np.mean(y_home[:cut]))/max(np.mean(y_home[:cut])**2,0.25)))
                    na=self._nb_nll(y_away[cut:cut+val],pa,max(0.02,float(np.var(y_away[:cut])-np.mean(y_away[:cut]))/max(np.mean(y_away[:cut])**2,0.25)))
                    losses.append((nh+na)/2.0)
                except Exception: continue
            if losses: scored.append((float(np.mean(losses)),name,factory))
        if not scored: return None
        scored.sort(key=lambda z:z[0]); top=scored[:3]
        fitted=[]; inv=np.asarray([1.0/max(x[0],1e-5) for x in top]); inv/=inv.sum()
        for (loss,name,factory),w in zip(top,inv):
            mh=factory(); ma=factory()
            self._fit_model(mh,X,y_home,self._sample_weights(len(X)),league)
            self._fit_model(ma,X,y_away,self._sample_weights(len(X)),league)
            fitted.append((name,mh,ma))
        # Learn prior-vs-model blend on the latest chronological validation
        # window. Candidate alpha values are deliberately bounded to prevent
        # the transparent prior from overpowering the learned models.
        alpha=0.20
        if splits:
            cut,val=splits[-1]
            try:
                mhm=[]; mam=[]
                for loss,name,factory in top:
                    mh=factory(); ma=factory(); self._fit_model(mh,X.iloc[:cut],y_home[:cut],self._sample_weights(cut),league); self._fit_model(ma,X.iloc[:cut],y_away[:cut],self._sample_weights(cut),league)
                    mhm.append(mh); mam.append(ma)
                baseh=np.zeros(val); basea=np.zeros(val)
                for w,mh,ma in zip(inv,mhm,mam):
                    baseh += float(w)*np.clip(mh.predict(X.iloc[cut:cut+val]),0.15,15.0); basea += float(w)*np.clip(ma.predict(X.iloc[cut:cut+val]),0.15,15.0)
                ph0,pa0=self._score_prior(X.iloc[cut:cut+val],league)
                best=(float('inf'),alpha)
                for al in np.linspace(0.0,0.55,23):
                    hh=(1-al)*baseh+al*ph0; aa=(1-al)*basea+al*pa0
                    loss=(self._nb_nll(y_home[cut:cut+val],hh,0.08)+self._nb_nll(y_away[cut:cut+val],aa,0.08))/2.0
                    if loss<best[0]: best=(loss,float(al))
                alpha=best[1]
            except Exception as e:
                self.audit.append({'type':'score_prior_calibration_error','error':str(e)})
        mu_h=float(np.mean(y_home)); mu_a=float(np.mean(y_away))
        alpha_h=max(0.03,float((np.var(y_home)-mu_h)/max(mu_h*mu_h,0.25)))
        alpha_a=max(0.03,float((np.var(y_away)-mu_a)/max(mu_a*mu_a,0.25)))
        return {'models':fitted,'weights':inv,'scores':{n:float(l) for l,n,_ in scored},'prior_blend':float(alpha),'dispersion_home':alpha_h,'dispersion_away':alpha_a}

    def predict_scores(self, fitted, xrow: pd.DataFrame, league: str) -> Tuple[float,float]:
        if fitted is None:
            return (2.35,2.35) if league=='NPB' else (4.55,4.55)
        lh=la=0.0
        for w,(name,mh,ma) in zip(fitted['weights'],fitted['models']):
            lh += float(w)*float(np.clip(mh.predict(xrow)[0],0.15,15.0))
            la += float(w)*float(np.clip(ma.predict(xrow)[0],0.15,15.0))
        try:
            ph,pa=self._score_prior(xrow,league)
            al=float(np.clip(fitted.get('prior_blend',0.20),0.0,0.55))
            lh=(1.0-al)*lh+al*float(ph[0]); la=(1.0-al)*la+al*float(pa[0])
        except Exception: pass
        return float(np.clip(lh,0.15,15.0)),float(np.clip(la,0.15,15.0))

'''
s=s[:start]+replacement+s[end:]
s=s.replace("# BACKTEST_RUNTIME_HARDENING_V3\n", "")
s="# BACKTEST_RUNTIME_HARDENING_V4\n"+s
P.write_text(s,encoding='utf-8')
print('[BACKTEST PATCH] V4 applied: adaptive score prior + chronological blend calibration + overdispersion')
