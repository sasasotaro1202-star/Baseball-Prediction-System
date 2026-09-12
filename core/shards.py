from dataclasses import dataclass, field
UNIT_COST_SECONDS={'mlb_season':3000.0,'statcast_window':45.0,'sofascore_event':1.2,'understat_season':12.0,'footballdata_season':3.0,'npb_season':35.0}
HOST_CONCURRENCY={'mlb':10,'statcast':6,'sofascore':6,'understat':4,'footballdata':4,'npb':4}
@dataclass
class WorkUnit:
    source:str; key:str; params:dict; cost:float=0.0
    def __post_init__(self):
        if not self.cost:self.cost=UNIT_COST_SECONDS.get(f'{self.source}_season',1.0)
@dataclass
class Shard:
    source:str; index:int; total:int; units:list=field(default_factory=list)
    @property
    def cost(self):return sum(u.cost for u in self.units)
    @property
    def estimated_seconds(self):return self.cost/HOST_CONCURRENCY.get(self.source,4)
def build_units(mlb_seasons=(),statcast_seasons=(),statcast_windows_per_season=52,sofascore_event_ids=(),understat_targets=(),footballdata_targets=(),npb_seasons=()):
    d={k:[] for k in HOST_CONCURRENCY}
    for s in mlb_seasons:d['mlb'].append(WorkUnit('mlb',f'mlb_games_{s}',{'season':s},UNIT_COST_SECONDS['mlb_season']))
    for s in statcast_seasons:
      for i in range(statcast_windows_per_season):d['statcast'].append(WorkUnit('statcast',f'statcast_{s}_{i:03d}',{'season':s,'window':i},UNIT_COST_SECONDS['statcast_window']))
    for e in sofascore_event_ids:d['sofascore'].append(WorkUnit('sofascore',f'sofascore_{e}',{'event_id':e},UNIT_COST_SECONDS['sofascore_event']))
    for c,n,s in understat_targets:d['understat'].append(WorkUnit('understat',f'understat_{c}_{s}',{'code':c,'name':n,'season':s},UNIT_COST_SECONDS['understat_season']))
    for c,s in footballdata_targets:d['footballdata'].append(WorkUnit('footballdata',f'footballdata_{c}_{s}',{'code':c,'season':s},UNIT_COST_SECONDS['footballdata_season']))
    for s in npb_seasons:d['npb'].append(WorkUnit('npb',f'npb_{s}',{'season':s},UNIT_COST_SECONDS['npb_season']))
    return d
def split(units,total,source):
    out=[Shard(source,i,max(1,total)) for i in range(max(1,total))]
    for u in sorted(units,key=lambda x:(-x.cost,x.key)):min(out,key=lambda s:(s.cost,s.index)).units.append(u)
    for s in out:s.units.sort(key=lambda u:u.key)
    return out
def plan(d,counts):return {k:split(v,counts.get(k,1),k) for k,v in d.items() if v}
def verify(d,p):
    r={'ok':True,'sources':{}}
    for k,v in d.items():
      if not v:continue
      want={u.key for u in v}; got=[u.key for s in p.get(k,[]) for u in s.units]
      missing=sorted(want-set(got)); dup=[x for x in set(got) if got.count(x)>1]; ok=len(missing)==0 and len(dup)==0 and len(want)==len(got)
      r['sources'][k]={'expected':len(want),'planned':len(got),'missing':missing[:10],'duplicated':dup[:10],'ok':ok};r['ok']=r['ok'] and ok
    return r
def estimate(p,max_parallel_jobs=20):
    times=sorted([s.estimated_seconds for ss in p.values() for s in ss],reverse=True);seq=sum(x*HOST_CONCURRENCY.get(k,4) for k,ss in p.items() for x in [sum(s.estimated_seconds for s in ss)])
    if not times:return {'sequential_hours':0,'wall_clock_hours':0,'shards':0}
    lanes=[0.0]*min(max_parallel_jobs,len(times))
    for t in times:i=lanes.index(min(lanes));lanes[i]+=t
    return {'sequential_hours':round(seq/3600,2),'longest_shard_hours':round(times[0]/3600,2),'wall_clock_hours':round(max(lanes)/3600,2),'shards':len(times),'max_parallel_jobs':len(lanes)}
def default_shard_counts():return {'mlb':4,'statcast':6,'sofascore':12,'understat':2,'footballdata':1,'npb':1}
