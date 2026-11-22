"""Is the 0.100/0.880 estimate a real MLE or just arch's starting-value grid point?"""
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
from arch import arch_model
A='../../work/'
df=pd.read_parquet(A+'slim.parquet',columns=['slot','value_eth']).sort_values('slot')
r=np.diff(np.log1p(df.value_eth.values))*100
am=arch_model(r,mean='Constant',vol='Garch',p=1,q=1,dist='t')
res0=am.fit(disp='off',show_warning=False)
print('default fit: iters',res0.optimization_result.nit if hasattr(res0.optimization_result,'nit') else res0.optimization_result, flush=True)
print(res0.optimization_result)
print('default params',res0.params.round(5).to_dict(),'ll',res0.loglikelihood, flush=True)
print('arch starting values:',am.starting_values(), flush=True)
# evaluate loglik at alternative params via fixed-parameter fit
def ll_at(mu,om,a,b,nu):
    fr=am.fix([mu,om,a,b,nu]); return fr.loglikelihood
print('ll at default-fit params:',ll_at(*res0.params.values))
for a,b in [(0.05,0.93),(0.2,0.78),(0.15,0.84),(0.3,0.69),(0.5,0.49),(0.08,0.91),(0.12,0.86),(0.1,0.85),(0.1,0.89)]:
    print(f'll at alpha={a} beta={b} (omega,nu re-fit by default ones):',ll_at(res0.params['mu'],res0.params['omega'],a,b,res0.params['nu']),flush=True)
# fits from different starting values, several optimizers / more iterations
for sv in [[0,1.0,0.05,0.93,4],[0,5.0,0.3,0.6,4],[0,1.0,0.2,0.78,5],[0,0.5,0.02,0.97,4]]:
    t=time.time()
    res=am.fit(disp='off',show_warning=False,starting_values=np.array(sv),options={'maxiter':2000,'ftol':1e-12})
    print(f'start {sv}: conv={res.convergence_flag} params={res.params.round(5).to_dict()} ll={res.loglikelihood:.1f} nit={getattr(res.optimization_result,"nit",None)} msg={getattr(res.optimization_result,"message",None)} t={time.time()-t:.0f}s',flush=True)
# profile over persistence with omega set from unconditional variance
print('profile: optimise (mu,omega,nu) with alpha,beta fixed via grid search over loglik (coarse)')
best=None
for a in [0.02,0.05,0.1,0.15,0.2,0.3,0.4]:
    for pers in [0.9,0.95,0.98,0.99,0.995,0.999]:
        b=pers-a
        if b<=0: continue
        # crude: omega set so unconditional var matches sample var
        om=np.var(r)*(1-pers)
        ll=ll_at(0.0,om,a,b,4.0)
        if best is None or ll>best[0]: best=(ll,a,b,om)
        print(f'a={a} b={b:.3f} om={om:.3f} ll={ll:.1f}',flush=True)
print('best',best)
