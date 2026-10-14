import sys; sys.path.insert(0,"proto")
from common import *
d=D[D.is_test==0].copy(); d["t"]=d.age-d.age_at_diagnosis
d=d[d.age_at_diagnosis.notna()]
def fit2(gfun):
    res=[]
    for pid,idx in d.groupby("patient_id").indices.items():
        g=gfun(d.t.values[idx]); A=np.c_[np.ones_like(g),g]; v=d.target.values[idx]
        c,*_=np.linalg.lstsq(A,v,rcond=None); res.append(v-A@c)
    return rmse(np.concatenate(res),0)
print("lin",fit2(lambda t:t))
for tau in [3,5,8,12,20]: print("exp",tau,fit2(lambda t:1-np.exp(-np.maximum(t,0)/tau)))
for p in [0.5,0.7]: print("pow",p,fit2(lambda t:np.maximum(t,0)**p))
print("log",fit2(lambda t:np.log1p(np.maximum(t,0))))
# with age instead
def fit2a(gfun):
    res=[]
    for pid,idx in d.groupby("patient_id").indices.items():
        g=gfun(d.age.values[idx]); A=np.c_[np.ones_like(g),g]; v=d.target.values[idx]
        c,*_=np.linalg.lstsq(A,v,rcond=None); res.append(v-A@c)
    return rmse(np.concatenate(res),0)
for tau in [5,10,20]: print("exp-age",tau,fit2a(lambda a:np.exp(a/tau)))
# negative t?
print("t<0 share",(d.t<0).mean(), d.t.min())
