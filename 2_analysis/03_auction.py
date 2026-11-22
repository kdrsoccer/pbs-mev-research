import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import paths
import pandas as pd, numpy as np
from scipy import stats
A=paths.WORK
SRC=paths.SRC
df=pd.read_parquet(A+'slim.parquet').sort_values('slot').reset_index(drop=True)
raw=pd.read_parquet(SRC,columns=['slot','n_bids','value_eth','bid_margin','n_builders','n_relays'])
print('raw matched (with dup rows):',raw.n_bids.notna().sum(),' dedup matched:',df.n_bids.notna().sum())
rm=raw.dropna(subset=['n_bids','value_eth']); rm['m']=rm.bid_margin*100
print('raw n with finite margin 0<=m<100:',(np.isfinite(rm.m)&(rm.m>=0)&(rm.m<100)).sum())
m=df.dropna(subset=['n_bids']).copy(); m['m']=m.bid_margin*100
print('dedup matched n',len(m))
print('margin NaN (monopoly, single bid)',m.m.isna().sum(), 'frac',m.m.isna().mean(), 'n_bids==1:',(m.n_bids==1).sum())
print('margin==100 (second bid 0):',(m.m>=100).sum(),'margin<0:',(m.m<0).sum(), 'margin>=99',(m.m>=99).sum())
clean=m[np.isfinite(m.m)&(m.m>=0)&(m.m<100)]; print('clean n',len(clean))
q=lambda s:pd.Series({'mean':s.mean(),'median':s.median(),'p10':s.quantile(.1),'p90':s.quantile(.9),'max':s.max(),'min':s.min()})
print('\n=== TABLE 7 (all matched, dedup) ===')
for c in ['n_bids','n_builders','n_relays','m','earliest_bid_ms','latest_bid_ms']:
    print(c, q(m[c]).round(3).to_dict())
print('margin (clean <100):',q(clean.m).round(3).to_dict())
print('latest > 4000ms frac:',(m.latest_bid_ms>4000).mean())
print('earliest > 0 frac:',(m.earliest_bid_ms>0).mean(), 'earliest max',m.earliest_bid_ms.max())
print('top_bid_wei max (ETH):',m.top_bid_wei.max()/1e18, ' n top_bid>1000 ETH:',(m.top_bid_wei>1e21).sum(), 'n top_bid > 100 ETH', (m.top_bid_wei>1e20).sum())
print('top bid vs delivered value: frac top_bid >= value_wei',(m.top_bid_wei>=m.value_wei*0.999).mean(), ' median ratio top/value',(m.top_bid_wei/m.value_wei).median())
print('frac value_wei > top_bid*1.01:',(m.value_wei>m.top_bid_wei*1.01).mean())
# terciles of value (author: qcut on matched incl. NaN-margin slots)
m['ter']=pd.qcut(m.value_eth,3,labels=['Low','Mid','High'])
print('\nmargin median by value tercile (all matched):',m.groupby('ter',observed=True).m.median().round(4).to_dict())
clean['ter']=pd.qcut(clean.value_eth,3,labels=['Low','Mid','High'])
print('margin median by value tercile (clean):',clean.groupby('ter',observed=True).m.median().round(4).to_dict())
print('n_builders median by tercile',m.groupby('ter',observed=True).n_builders.median().to_dict())
print('latest median by tercile',m.groupby('ter',observed=True).latest_bid_ms.median().to_dict())
def sp(x,y,name):
    ok=np.isfinite(x)&np.isfinite(y); r,p=stats.spearmanr(x[ok],y[ok]); print(f'{name}: rho={r:.4f} p={p:.1e} n={ok.sum()}')
sp(m.value_eth,m.m,'value vs margin (all matched, NaN dropped)')
sp(clean.value_eth,clean.m,'value vs margin (clean)')
sp(m.value_eth,m.n_builders,'value vs n_builders')
sp(m.value_eth,m.latest_bid_ms,'value vs latest_bid')
sp(m.value_eth,m.earliest_bid_ms,'value vs earliest_bid')
sp(m.n_builders,m.m,'n_builders vs margin')
sp(m.n_relays,m.m,'n_relays vs margin (all)')
sp(clean.n_relays,clean.m,'n_relays vs margin (clean)')
sub=clean[clean.n_relays.between(2,6)]; sp(sub.n_relays,sub.m,'n_relays vs margin excl 1 and 7+')
sub=clean[clean.n_relays.between(1,7)]; sp(sub.n_relays,sub.m,'n_relays vs margin excl 8')
sp(m.n_relays,m.n_builders,'n_relays vs n_builders')
# bootstrap CIs
rng=np.random.default_rng(1)
def boot(x,y,name,n=8000,B=1000):
    ok=np.isfinite(x)&np.isfinite(y); x=x[ok].values; y=y[ok].values
    rs=[stats.spearmanr(x[i],y[i])[0] for i in (rng.choice(len(x),n,replace=False) for _ in range(B))]
    print(f'boot {name}: mean={np.mean(rs):.3f} CI=[{np.percentile(rs,2.5):.3f},{np.percentile(rs,97.5):.3f}]')
boot(m.value_eth,m.m,'value-margin'); boot(m.value_eth,m.n_builders,'value-builders'); boot(m.value_eth,m.latest_bid_ms,'value-latest'); boot(m.n_relays,m.m,'relays-margin')
print('\n=== RELAY TABLE ===')
print(m.groupby('n_relays').agg(med_margin=('m','median'),n=('m','count'),n_all=('slot','count')).round(3).to_string())
print('\n=== PROPOSER EFFICIENCY ===')
clean['eff']=100-clean.m
print((clean.groupby('ter',observed=True).eff.agg(['mean','median'])).round(2).to_string())
he=clean.groupby('hour_c').m.median(); print('median margin by hour:'); print(he.round(4).to_string())
print('lowest-eff (highest margin) hours',he.nlargest(4).index.tolist(),'highest-eff hours',he.nsmallest(6).index.tolist())
print('mean margin by hour top:',clean.groupby('hour_c').m.mean().nlargest(4).round(3).to_dict())
print('median margin by dow:',clean.groupby('dow_c').m.median().round(4).to_dict())
# monopoly fraction
print('monopoly (n_bids==1) frac',(m.n_bids==1).mean(),'n_builders==1 frac',(m.n_builders==1).mean(), (m.n_builders==1).sum())
m.to_parquet(A+'matched.parquet',index=False)
