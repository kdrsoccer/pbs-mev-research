"""Level GARCH: every number in the Ljung-Box table and the two GARCH tables of the paper.
Also spot-checks three rows of Table 1 of the theory paper.
Needs: pandas pyarrow numpy scipy statsmodels arch   (pip install them if missing)"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import paths
import os, sys, json, time, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.optimize import minimize
from arch import arch_model
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf
SRC=paths.SRC
t0=time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]",*a,flush=True)

log('loading parquet')
df=pd.read_parquet(SRC,columns=['slot','value_eth'])
df=df.sort_values(['slot','value_eth'],ascending=[True,False]).drop_duplicates('slot').reset_index(drop=True)
v=df.value_eth.values; u=np.log1p(v); r=np.diff(u)*100
log(f'unique slots: {len(df):,}   (paper: 1,271,079)')

def garch_gauss(y,label,q=1,amax=3.0):
    am=arch_model(y,mean='Zero',vol='Garch',p=1,q=q,dist='normal')
    def unpack(th):
        pr=[np.exp(th[0]),amax/(1+np.exp(-th[1]))]
        if q: pr.append(1/(1+np.exp(-th[2])))
        return pr
    def negll(th):
        try: return -am.fix(unpack(th)).loglikelihood
        except Exception: return 1e12
    best=None
    for a0,b0 in [(0.1,0.8),(0.4,0.4),(0.8,0.1)]:
        th0=[np.log(np.var(y)*0.3),np.log((a0/amax)/(1-a0/amax))]+([np.log(b0/(1-b0))] if q else [])
        rr=minimize(negll,th0,method='Nelder-Mead',options={'maxfev':800,'fatol':0.5})
        if best is None or rr.fun<best[0]: best=(rr.fun,unpack(rr.x))
    p=best[1]; a=p[1]; b=p[2] if q else 0.0
    log(f'{label:40s} alpha={a:.3f} beta={b:.3f} alpha+beta={a+b:.3f}')
    return p

# ---------- Table 4 / 5: level GARCH ----------
log('AR(5) filter on level')
m5=AutoReg(u,lags=5,trend='c').fit(); e5=m5.resid*100
log('AR(5) coefs', np.round(np.asarray(m5.params)[1:],3), '  (paper: 0.295 0.124 0.089 0.069 0.072)')
def lbq(x,lags):
    """Ljung-Box Q, same statistic as statsmodels.acorr_ljungbox but with an FFT autocorrelation (seconds instead of ~10 min at n=1.27M)."""
    n=len(x); a=acf(x,nlags=max(lags),fft=True)[1:]
    return [float(n*(n+2)*np.sum(a[:k]**2/(n-np.arange(1,k+1)))) for k in lags]
lb=np.round(lbq(e5**2,[10,20,50]),0).tolist()
log('Ljung-Box level resid^2 Q10/20/50:',lb,'  (paper: 44,779 / 56,366 / 72,464)')
lbd=np.round(lbq(r**2,[10,20,50]),0).tolist()
log('Ljung-Box diff^2 Q10/20/50:',lbd,'  (paper: 257,972 / 268,977 / 280,428)')
log('ACF r lag1:',round(acf(r,nlags=1)[1],3),'(paper -0.425)   ACF r^2 lags1-2:',np.round(acf(r**2,nlags=2)[1:],3),'(paper 0.434, 0.075)')
p=garch_gauss(e5,'Table 4: AR(5) resid GARCH(1,1)   [0.044/0.965]')
# Tighten the optimum (still unconstrained), then Bollerslev-Wooldridge robust SEs evaluated AT that optimum.
# arch's own .fit() cannot be used for this: it imposes alpha+beta<=1 and returns a boundary point with a lower likelihood.
am4=arch_model(e5,mean='Zero',vol='Garch',p=1,q=1,dist='normal'); e2=e5**2; n4=len(e5)
rr=minimize(lambda th:-am4.fix(list(np.exp(th))).loglikelihood,np.log(p),method='Nelder-Mead',options={'xatol':1e-8,'fatol':1e-6,'maxfev':3000})
p4=np.exp(rr.x); ll4=-rr.fun
def lt(th):
    s2=np.asarray(am4.fix(list(th)).conditional_volatility)**2
    return -0.5*(np.log(2*np.pi)+np.log(s2)+e2/s2)
def sandwich(th):
    th=np.asarray(th,float); k=len(th); h=1e-4*np.abs(th); G=np.empty((n4,k))
    for i in range(k):
        d=np.zeros(k); d[i]=h[i]; G[:,i]=(lt(th+d)-lt(th-d))/(2*h[i])          # per-observation scores
    B=G.T@G/n4; fm=lambda t: lt(t).mean(); A=np.empty((k,k))
    for i in range(k):
        for j in range(i,k):
            di=np.zeros(k); dj=np.zeros(k); di[i]=10*h[i]; dj[j]=10*h[j]
            A[i,j]=A[j,i]=(fm(th+di+dj)-fm(th+di-dj)-fm(th-di+dj)+fm(th-di-dj))/(4*di[i]*dj[j])   # Hessian of mean log-likelihood
    Ai=np.linalg.inv(A); return np.sqrt(np.diag(Ai@B@Ai/n4)), G.sum(0)
se4,score=sandwich(p4)
log(f'Table 4: omega={p4[0]:.3f} alpha={p4[1]:.3f} beta={p4[2]:.3f}  log-lik={ll4:,.0f}  n={n4:,}   (paper: 0.648 / 0.044 / 0.965, -4,240,438)')
log(f'   robust SE: omega={se4[0]:.3f} alpha={se4[1]:.3f} beta={se4[2]:.3f}   (paper: .158 / .010 / .007)   score at optimum: {np.round(score,2).tolist()}')
json.dump(dict(table4=dict(omega=float(p4[0]),alpha=float(p4[1]),beta=float(p4[2]),loglik=float(ll4),se=[float(x) for x in se4])),open(paths.WORK+'garch_level.json','w'))
for lag,lab in [(1,'AR(1)'),(20,'AR(20)')]:
    e=AutoReg(u,lags=lag,trend='c').fit().resid*100
    garch_gauss(e,f'Table 5: {lab} resid   [1.007 / 1.010]')
n=len(e5); garch_gauss(e5[:n//2],'Table 5: AR(5) first half   [1.002]'); garch_gauss(e5[n//2:],'Table 5: AR(5) second half  [1.075]')
garch_gauss(r,'Table 5: differenced (diagnostic) [0.539/0.731]')

# ---------- Theory Table 1: pseudo-true alpha* ----------
log('theory table: pseudo-true alpha* (fast)')
rng=np.random.default_rng(7); M=1_000_000
def alpha_star(W,Wp):
    W=W-W.mean(); Wp=Wp-Wp.mean(); s2=W.var(); h=s2+W**2; R=(W-Wp)**2
    def Q(th): s=np.exp(th[0])+np.exp(th[1])*R; return np.mean(np.log(s)+h/s)
    best=min((minimize(Q,[np.log(s2),np.log(a0)],method='Nelder-Mead') for a0 in [0.1,0.5,1.0]),key=lambda x:x.fun)
    return np.exp(best.x[1])
log('  Gaussian  alpha* =',round(alpha_star(rng.normal(size=M),rng.normal(size=M)),3),' (theorem: 0.250)')
log('  Lognormal(1) alpha* =',round(alpha_star(rng.lognormal(0,1,M),rng.lognormal(0,1,M)),3),' (theory paper Table 1: 0.379)')
i=rng.integers(0,len(u),M); j=rng.integers(0,len(u),M)
log('  MEV log1p(v) bootstrap alpha* =',round(alpha_star(u[i],u[j]),3),' (theory paper Table 1: 0.44)')
log('DONE. Tables 4-5 and Ljung-Box should match the bracketed paper values to the printed digits. The three theory rows are Monte Carlo: expect agreement to about +/-0.005.')
