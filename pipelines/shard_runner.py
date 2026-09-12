import json
from core import shards
LEAGUES=['E0','SP1','D1','I1','F1','N1'];NAMES={'E0':'EPL','SP1':'La_liga','D1':'Bundesliga','I1':'Serie_A','F1':'Ligue_1','N1':'Eredivisie'}
def build_plan(windows=52):
 d=shards.build_units(mlb_seasons=range(2015,2027),statcast_seasons=range(2015,2027),statcast_windows_per_season=windows,understat_targets=[(x,NAMES[x],y) for x in LEAGUES for y in range(2014,2026)],footballdata_targets=[(x,y) for x in LEAGUES for y in range(2010,2026)],npb_seasons=range(2010,2027))
 p=shards.plan(d,shards.default_shard_counts());return p,shards.verify(d,p),shards.estimate(p)
