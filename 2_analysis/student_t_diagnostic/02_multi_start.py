"""Search for the actual GARCH(1,1)-t MLE using arch's likelihood via model.fix + Nelder-Mead."""
import pandas as pd, numpy as np, time, warnings, sys
warnings.filterwarnings('ignore')
from arch import arch_model
from scipy.optimize import minimize
A='../../work/'
which=sys.argv[1] if len(sys.argv)>1 else 'value'
if which=='value':
    df=pd.read_parquet(A+'slim.parquet',columns=['slot','value_eth']).sort_values('slot')
    y=np.diff(np.log1p(df.value_eth.values))*100
elif which=='value_w':
    df=pd.read_parquet(A+'slim.parquet',columns=['slot','value_eth']).sort_values('slot')
    y=np.diff(np.log1p(df.value_eth.values))*100; y=np.clip(y,np.percentile(y,1),np.percentile(y,99))
elif which=='resid':
    y=np.load(A+'mev_resid.npy')
else:
    m=pd.read_parquet(A+'matched.parquet',columns=['slot','m']).sort_values('slot')
    y=m.m.values; y=y[np.isfinite(y)&(y>=0)&(y<100)]
am=arch_model(y,mean='Constant',vol='Garch',p=1,q=1,dist='t')
def negll(th):
    mu,lom,la,lb,lnu=th
    a=1/(1+np.exp(-la)); b=(1-a)/(1+np.exp(-lb))   # a in (0,1), b in (0,1-a)
    nu=2.05+np.exp(lnu); om=np.exp(lom)
    try: return -am.fix([mu,om,a,b,nu]).loglikelihood
    except Exception: return 1e12
def unpack(th):
    mu,lom,la,lb,lnu=th; a=1/(1+np.exp(-la)); b=(1-a)/(1+np.exp(-lb)); return dict(mu=mu,omega=np.exp(lom),alpha=a,beta=b,nu=2.05+np.exp(lnu),pers=a+b)
best=None
for a0,b0 in [(0.1,0.88),(0.4,0.59),(0.6,0.39),(0.2,0.79),(0.5,0.49),(0.8,0.19)]:
    th0=[0.0,np.log(np.var(y)*(1-a0-b0)+1e-6),np.log(a0/(1-a0)),np.log((b0/(1-a0))/(1-b0/(1-a0))),np.log(2.0)]
    t=time.time()
    r=minimize(negll,th0,method='Nelder-Mead',options={'maxfev':400,'xatol':1e-4,'fatol':1.0})
    print(which,'start',a0,b0,'->',{k:round(v,5) for k,v in unpack(r.x).items()},'ll',-r.fun,'nfev',r.nfev,f'{time.time()-t:.0f}s',flush=True)
    if best is None or r.fun<best[0]: best=(r.fun,r.x)
print(which,'BEST',{k:round(v,5) for k,v in unpack(best[1]).items()},'ll',-best[0],flush=True)
# polish with arch from best
p=unpack(best[1])
res=am.fit(disp='off',show_warning=False,starting_values=np.array([p['mu'],p['omega'],p['alpha'],p['beta'],p['nu']]),options={'maxiter':2000})
print(which,'arch polish:',res.params.round(5).to_dict(),'ll',res.loglikelihood,'AIC',res.aic,'conv',res.convergence_flag,flush=True)
