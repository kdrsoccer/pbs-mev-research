import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import paths
import pandas as pd, numpy as np
from scipy import stats
import statsmodels.api as sm
A=paths.WORK
C=paths.DATA
m=pd.read_parquet(A+'matched.parquet')
clean=m[np.isfinite(m.m)&(m.m>=0)&(m.m<100)].copy()
print('=== MARGIN REGRESSION (dedup clean n=%d) ==='%len(clean))
X=pd.DataFrame({'log_value':np.log(clean.value_eth),'n_builders':clean.n_builders,'n_relays':clean.n_relays,'hour':clean.hour_c,'dow':clean.dow_c})
res=sm.OLS(clean.m.values,sm.add_constant(X)).fit(cov_type='HC0')
print(res.summary().tables[1]); print('R2',res.rsquared,'n',int(res.nobs))
# with hour/dow fixed effects instead of linear
X2=pd.get_dummies(X,columns=['hour','dow'],drop_first=True,dtype=float)
res2=sm.OLS(clean.m.values,sm.add_constant(X2)).fit()
print('FE version coefs',res2.params[['log_value','n_builders','n_relays']].round(4).to_dict(),'R2',round(res2.rsquared,4))

print('\n=== ETH DECOMPOSITION ===')
eth=pd.read_parquet(C+'eth_price_hourly.parquet'); print('eth rows',len(eth),eth.datetime.min(),eth.datetime.max(), 'tz',eth.datetime.dt.tz)
eth['hour_floor']=eth.datetime.dt.floor('h')
print('eth dup hours',eth.hour_floor.duplicated().sum(),'missing hours',int((eth.hour_floor.max()-eth.hour_floor.min())/pd.Timedelta('1h'))+1-eth.hour_floor.nunique())
full=pd.read_parquet(A+'slim.parquet',columns=['slot','t','value_eth']).sort_values('slot').reset_index(drop=True)
full['hour_floor']=full.t.dt.tz_localize(None).dt.floor('h')
f2=full.merge(eth[['hour_floor','eth_close']],on='hour_floor',how='left')
print('full matched to ETH:',f2.eth_close.notna().sum(),f2.eth_close.notna().mean())
m['hour_floor']=m.t.dt.tz_localize(None).dt.floor('h')
m2=m.merge(eth[['hour_floor','eth_close']],on='hour_floor',how='left')
print('auction-matched slots matched to ETH:',m2.eth_close.notna().sum(),m2.eth_close.notna().mean())
for name,d in [('full delivered',f2),('matched only (author script)',m2)]:
    d=d.dropna(subset=['eth_close']).copy()
    d['log_eth']=np.log(d.eth_close); d['log_val']=np.log(d.value_eth+1e-10)
    sl,ic,r,p,se=stats.linregress(d.log_eth,d.log_val)
    r1=stats.spearmanr(d.eth_close,d.value_eth)[0]; r2=stats.spearmanr(d.eth_close,d.value_eth*d.eth_close)[0]
    print(f'{name}: n={len(d)} intercept={ic:.3f} slope={sl:.4f} R2={r**2:.5f} spearman ETH={r1:.4f} USD={r2:.4f}')
# residual GARCH prep: block-value log return vs ETH log return
d=f2.copy(); d['lv']=np.log1p(d.value_eth); d['r']=d.lv.diff(); d['re']=np.log(d.eth_close).diff()
ok=d.r.notna()&d.re.notna()&np.isfinite(d.re)
print('n for return-on-ETH-return regression:',ok.sum())
sl,ic,r,p,se=stats.linregress(d.re[ok],d.r[ok]); print(f'slope={sl:.4f} R2={r**2:.6f}')
resid=d.r[ok]-(ic+sl*d.re[ok]); np.save(A+'mev_resid.npy',resid.values*100)
# realized-vol decomposition: rolling 300 std of r vs of residual
rv_tot=np.sqrt(300)*d.r.rolling(300).std(); rv_mev=np.sqrt(300)*pd.Series(np.where(ok,d.r-(ic+sl*d.re.fillna(0)),np.nan)).rolling(300).std()
rho=stats.spearmanr(rv_tot,rv_mev,nan_policy='omit')[0]; print('rv_mev vs rv_total spearman',rho)
print('var share: var(resid)/var(r)',np.var(resid)/np.var(d.r[ok]))

print('\n=== HHI vs MARGIN ===')
m['week']=m.t.dt.tz_localize(None).dt.to_period('W').dt.start_time
wm=m.groupby('week').agg(med_margin=('m','median'),med_b=('n_builders','median'),med_bids=('n_bids','median'),n=('slot','count'))
w=pd.read_csv(A+'weekly_conc.csv',index_col=0,parse_dates=True)
j=wm.join(w.drop(columns=['n']),how='inner').sort_index(); j['hhi_lag']=j.hhi.shift(1); jj=j.dropna(subset=['hhi_lag'])
print('weeks',len(j),'with lag',len(jj))
r,p=stats.spearmanr(jj.hhi_lag,jj.med_margin); sl,ic,rr,pp,se=stats.linregress(jj.hhi_lag,jj.med_margin)
print(f'lagged HHI vs margin: spearman rho={r:.3f} p={p:.3f} OLS R2={rr**2:.3f}')
r,p=stats.spearmanr(j.hhi,j.med_margin); print(f'contemporaneous: rho={r:.3f} p={p:.3f}')

print('\n=== COMPETITIVENESS TIME SERIES (weekly medians) ===')
pd.set_option('display.width',200)
print(wm.round(3).to_string())
mo=m.groupby(m.t.dt.tz_localize(None).dt.to_period('M')).agg(med_margin=('m','median'),med_b=('n_builders','median'),med_bids=('n_bids','median'),n=('slot','count'),nb_active=('builder_pubkey','nunique'))
full['month']=full.t.dt.tz_localize(None).dt.to_period('M'); mo['n_delivered']=full.groupby('month').size(); mo['match_rate']=mo.n/mo.n_delivered
print(mo.round(3).to_string())
print('match rate by month min/max',mo.match_rate.min(),mo.match_rate.max())
print('late-2024 (Sep-Dec) weekly med builders:',wm.loc[:'2024-12-31'].med_b.describe()[['min','50%','max']].to_dict())
print('2026 weekly med builders:',wm.loc['2026-01-01':].med_b.describe()[['min','50%','max']].to_dict())
print('weekly med bids range',wm.med_bids.min(),wm.med_bids.max())
print('weekly med margin range',wm.med_margin.min(),wm.med_margin.max(),' 2025 range',wm.loc['2025-01-01':'2025-12-31'].med_margin.agg(['min','max']).to_dict(),' Dec25-Mar26',wm.loc['2025-12-01':'2026-03-31'].med_margin.agg(['min','max']).to_dict())
wm.to_csv(A+'weekly_auction.csv')
