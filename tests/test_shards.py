from core import shards
def units(n):return [shards.WorkUnit('sofascore',f'e{i}',{},1.2) for i in range(n)]
def test_complete():
 u={'sofascore':units(137)};p=shards.plan(u,{'sofascore':12});r=shards.verify(u,p);assert r['ok'],r;assert len({x.key for s in p['sofascore'] for x in s.units})==137
def test_deterministic():
 a=[[x.key for x in s.units] for s in shards.split(units(90),7,'sofascore')];b=[[x.key for x in s.units] for s in shards.split(list(reversed(units(90))),7,'sofascore')];assert a==b
def test_missing_and_duplicate_detected():
 u={'sofascore':units(20)};p=shards.plan(u,{'sofascore':4});p['sofascore'][0].units.pop();r=shards.verify(u,p);assert not r['ok'],r['sources']
 u={'sofascore':units(20)};p=shards.plan(u,{'sofascore':4});p['sofascore'][1].units.append(p['sofascore'][0].units[0]);r=shards.verify(u,p);assert not r['ok'],r['sources']
