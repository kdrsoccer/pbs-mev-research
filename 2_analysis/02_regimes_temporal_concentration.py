import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import paths
import pandas as pd, numpy as np
from scipy import stats
A=paths.WORK
C=paths.REF
df=pd.read_parquet(A+'slim.parquet').sort_values('slot').reset_index(drop=True)
v=df.value_eth
print('=== DESCRIPTIVES ===')
print(f"n={len(df)} mean={v.mean():.4f} median={v.median():.4f} max={v.max():.1f} std={v.std():.3f}")
print('relay min/max date:'); print(df.groupby('relay').t.agg(['min','max','count']))
print('delivered share of all slots:', len(df)/((df.slot.max()-df.slot.min()+1)))

print('\n=== VOL REGIMES ===')
lv=np.log1p(v.values); r=np.diff(lv)
rs=pd.Series(np.r_[np.nan,r])
rv=np.sqrt(300)*rs.rolling(300).std()   # pandas std ddof=1
df['rv']=rv
ok=df.rv.notna()
print('n with rv',ok.sum())
q33,q67=df.rv.quantile([1/3,2/3]); print(f'tertile cuts: {q33:.4f} {q67:.4f}')
p33,p67=np.percentile(df.rv.dropna(),[33,67]); print(f'33/67 pct: {p33:.4f} {p67:.4f}')
for cuts,name in [((0.6554,1.4340),'paper cuts'),((q33,q67),'my tertiles')]:
    reg=pd.cut(df.rv,[-np.inf,cuts[0],cuts[1],np.inf],labels=['Low','Mid','High'])
    g=df.groupby(reg,observed=True).value_eth
    tab=g.agg(mean='mean',median='median',std='std',p25=lambda x:x.quantile(.25),p75=lambda x:x.quantile(.75),p95=lambda x:x.quantile(.95),n='count')
    print(name); print(tab.round(4).to_string())
    print('ratio high/low mean',tab.loc['High','mean']/tab.loc['Low','mean'],'var ratio',(tab.loc['High','std']/tab.loc['Low','std'])**2)
    H,p=stats.kruskal(*[x.values for _,x in g]); print(f'KW H={H:.1f} p={p:.2e}')
    lo=df.value_eth[reg=='Low']; hi=df.value_eth[reg=='High']
    U,pu=stats.mannwhitneyu(lo,hi,alternative='less'); print(f'MWU U={U} p={pu:.2e}')
rho,p=stats.spearmanr(df.rv[ok],df.value_eth[ok]); print(f'Spearman rv vs value rho={rho:.4f} p={p:.2e} n={ok.sum()}')
rng=np.random.default_rng(0); sub=df[ok]
bs=[stats.spearmanr(*sub.sample(8000,random_state=int(rng.integers(1e9)))[['rv','value_eth']].values.T)[0] for _ in range(1000)]
bs=np.array(bs); print(f'bootstrap n=8000 B=1000 mean={bs.mean():.3f} sd={bs.std():.3f} CI=[{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]')
# clock time covered by 300-obs window
dt=(df.t.diff(300).dt.total_seconds()/3600); print('300-obs window hours: median',dt.median(),'mean',dt.mean())

print('\n=== TEMPORAL (correct slot-derived UTC) ===')
hr=df.groupby('hour_c').value_eth.agg(['mean','median','std','count']); print(hr.round(4).to_string())
dw=df.groupby('dow_c').value_eth.agg(['mean','median','std','count']); dw.index=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']; print(dw.round(4).to_string())
H,p=stats.kruskal(*[x.values for _,x in df.groupby('hour_c').value_eth]); print(f'KW hour H={H:.1f}')
H,p=stats.kruskal(*[x.values for _,x in df.groupby('dow_c').value_eth]); print(f'KW dow H={H:.1f}')
print('hour 2 max value',df[df.hour_c==2].value_eth.max(), 'top3 blocks overall:'); print(df.nlargest(3,'value_eth')[['slot','t','value_eth','hour_c','dow_c']])
# compare against author's csv
hp=pd.read_csv(C+'hourly_patterns.csv'); dp=pd.read_csv(C+'daily_patterns.csv')
print('author hourly csv vs mine (max abs diff mean):',np.abs(hp['mean'].values-hr['mean'].values).max(), 'counts diff', np.abs(hp['count'].values-hr['count'].values).max())
print('author daily csv vs mine:',np.abs(dp['mean'].values-dw['mean'].values).max())
# what if stored (wrong) hour used? need raw columns
raw=pd.read_parquet(paths.SRC,columns=['slot','hour','dayofweek','value_eth'])
hw=raw.groupby('hour').value_eth.mean(); print('stored-hour means (raw, with dups) vs author csv maxdiff:',np.abs(hp['mean'].values-hw.values).max())
dwr=raw.groupby('dayofweek').value_eth.mean(); print('stored-dow means vs author csv maxdiff:',np.abs(dp['mean'].values-dwr.values).max())

print('\n=== CONCENTRATION ===')
print('n builders',df.builder_pubkey.nunique())
cnt=df.builder_pubkey.value_counts().sort_values()
def gini(c):
    c=np.sort(np.asarray(c,float)); n=len(c); X=np.arange(0,n+1)/n; Y=np.r_[0,np.cumsum(c)/c.sum()]
    return 1-np.sum((X[1:]-X[:-1])*(Y[1:]+Y[:-1]))
print('pooled gini',gini(cnt.values))
half=cnt.iloc[:len(cnt)//2].sum()/cnt.sum(); print('bottom 50% builders share of blocks',half, 'bottom 50% count',len(cnt)//2)
print('top builder share',cnt.max()/cnt.sum(), 'top5 share',cnt.nlargest(5).sum()/cnt.sum())
print('pooled HHI',10000*((cnt/cnt.sum())**2).sum())
df['week']=df.t.dt.to_period('W').dt.start_time
wk=df.groupby('week')
hhi=wk.builder_pubkey.apply(lambda s:10000*((s.value_counts(normalize=True))**2).sum())
gw=wk.builder_pubkey.apply(lambda s:gini(s.value_counts().values))
w=pd.DataFrame({'hhi':hhi,'gini':gw,'n':wk.size()})
print(w.to_string())
print('weekly gini mean',w.gini.mean(),'first',w.gini.iloc[0],'last',w.gini.iloc[-1],'last-8 mean',w.gini.iloc[-8:].mean())
print('HHI first',w.hhi.iloc[0],'early2025 (Jan-Mar) range',w.loc['2025-01-01':'2025-03-31','hhi'].agg(['min','max']).values,'mid2025 (May-Aug) max',w.loc['2025-05-01':'2025-08-31','hhi'].max(), w.loc['2025-05-01':'2025-08-31','hhi'].idxmax())
print('HHI 2026 range',w.loc['2026-01-01':,'hhi'].agg(['min','max']).values)
print('weeks with HHI>1000 after 2025-01:',w.loc['2025-01-01':][w.loc['2025-01-01':].hhi>1000][['hhi']].to_string())
ah=pd.read_csv(C+'builder_hhi.csv'); print('author hhi csv n weeks',len(ah),'mine',len(w));
m=ah.merge(w.reset_index().assign(week=lambda x:x.week.dt.strftime('%Y-%m-%d')),on='week',how='outer',suffixes=('_a','_m'))
print('max abs hhi diff',(m.hhi_a-m.hhi_m).abs().max(),'max gini diff',(m.gini_a-m.gini_m).abs().max())
print(m[(m.hhi_a-m.hhi_m).abs()>1].to_string())
w.to_csv(A+'weekly_conc.csv')
