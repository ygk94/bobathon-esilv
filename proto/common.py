import pandas as pd, numpy as np
from sklearn.model_selection import GroupKFold
X=pd.read_csv("data/X_train.csv"); y=pd.read_csv("data/y_train.csv"); T=pd.read_csv("data/X_test.csv")
tr=X.merge(y,on="Index"); tr["is_test"]=0; T["is_test"]=1; T["target"]=np.nan
D=pd.concat([tr,T],ignore_index=True).sort_values(["patient_id","age"]).reset_index(drop=True)
TRm=D.is_test.eq(0).values
folds=np.full(len(D),-1)
trD=D[TRm]
for k,(a,b) in enumerate(GroupKFold(5,shuffle=True,random_state=0).split(trD,groups=trD.patient_id)):
    folds[trD.index[b]]=k
D["fold"]=folds
def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))

def polyfeat(g, col, deg, prefix):
    m=g[col].notna().values; a=g.age.values; out=np.full(len(g),np.nan); sl=np.nan
    if m.sum()>deg:
        c=np.polyfit(a[m]-60,g[col].values[m],deg); out=np.polyval(c,a-60); sl=c[-2]
    elif m.sum()>0: out=np.full(len(g),g[col].values[m].mean())
    return out, sl

def add_features(D):
    D=D.copy()
    D["t"]=D.age-D.age_at_diagnosis
    for c in ["cohort","gene","rater_id"]: D[c+"_c"]=D[c].astype("category")
    g=D.groupby("patient_id")
    D["vis_idx"]=g.cumcount(); D["n_vis"]=g.age.transform("size")
    D["age_first"]=g.age.transform("min"); D["age_last"]=g.age.transform("max")
    D["dt_first"]=D.age-D.age_first; D["dt_last"]=D.age_last-D.age
    for c in ["on","off","ledd","time_since_intake_on","time_since_intake_off"]:
        D[f"p_mean_{c}"]=g[c].transform("mean"); D[f"p_cnt_{c}"]=g[c].transform("count")
        D[f"p_max_{c}"]=g[c].transform("max"); D[f"p_min_{c}"]=g[c].transform("min")
        D[f"prev_{c}"]=g[c].transform(lambda s:s.ffill().shift(1)); D[f"next_{c}"]=g[c].transform(lambda s:s.bfill().shift(-1))
    D["off_minus_on"]=D.off-D.on
    for c in ["on","off","ledd"]:
        for deg in [1,2]:
            fits=[];sls=[]
            for pid,gg in D.groupby("patient_id",sort=False):
                o,s=polyfeat(gg,c,deg,c); fits.append(pd.Series(o,index=gg.index)); sls.append(pd.Series(s,index=gg.index))
            D[f"fit{deg}_{c}"]=pd.concat(fits); D[f"slope{deg}_{c}"]=pd.concat(sls)
    return D
