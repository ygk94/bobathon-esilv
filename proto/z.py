import numpy as np, pandas as pd
BINS=[-0.1,.25,.5,.75,1,1.25,1.5,2,2.5,3,4,5,8]
def wpoly(age,v,w,deg,lam):
    # weighted ridge poly fit centred on patient mean age; penalise coefs above intercept
    a=age-age.mean(); A=np.vander(a,deg+1,increasing=True)
    W=np.diag(w); P=np.diag([0]+[lam]*deg)
    c=np.linalg.solve(A.T@W@A+P,A.T@W@v); return A@c, c
def z_features(D, fitmask):
    """fitmask: rows whose target can be used to estimate global factors"""
    F=pd.DataFrame(index=D.index)
    S=D[fitmask & (D.target>1)]
    rf_off=(S.off/S.target).groupby(S.rater_id).median()
    rf_on=(S.on/S.target/1.0).groupby(S.rater_id).median()
    gb=pd.cut(D.time_since_intake_on,BINS).astype(str)
    Sb=gb[S.index]
    # on ratio curve after removing rater on factor
    ron=(S.on/S.target)/S.rater_id.map(rf_on)*(S.on/S.target).median()
    curve=ron.groupby(Sb).median()
    off_scale=D.rater_id.map(rf_off).fillna(rf_off.median())
    on_scale=(D.rater_id.map(rf_on).fillna(rf_on.median())/(S.on/S.target).median())*gb.map(curve).fillna((S.on/S.target).median())
    F["z_off"]=D.off/off_scale; F["z_on"]=D.on/on_scale
    F["off_scale"]=off_scale; F["on_scale"]=on_scale
    # residual variance of z (relative) for weights
    zo=F.z_off[S.index]/S.target; zn=F.z_on[S.index]/S.target
    vo=np.nanvar(zo); vn=np.nanvar(zn)
    for lam in [1.0,20.0]:
        for deg in [1,2]:
            out=np.full(len(D),np.nan); out_on=np.full(len(D),np.nan); out_off=np.full(len(D),np.nan); slope=np.full(len(D),np.nan)
            for pid,idx in D.groupby("patient_id",sort=False).indices.items():
                age=D.age.values[idx]; zof=F.z_off.values[idx]; zon=F.z_on.values[idx]
                vals=np.concatenate([zof,zon]); ages=np.concatenate([age,age]); w=np.concatenate([np.full(len(idx),1/vo),np.full(len(idx),1/vn)])
                m=~np.isnan(vals)
                if m.sum()==0: continue
                am=ages[m]
                a=age-am.mean(); A=np.vander(am-am.mean(),deg+1,increasing=True); W=w[m]
                P=np.diag([1e-6]+[lam*1e-2*W.sum()/len(W)]*deg)
                c=np.linalg.solve(A.T@(A*W[:,None])+P,A.T@(W*vals[m]))
                out[idx]=np.vander(a,deg+1,increasing=True)@c; slope[idx]=c[1]
            F[f"zfit_d{deg}_l{int(lam)}"]=out
            if lam==1.0: F[f"zslope_d{deg}"]=slope
    return F
