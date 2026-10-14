import sys; sys.path.insert(0,"proto")
from common import *
import lightgbm as lgb, warnings, pickle; warnings.filterwarnings("ignore")
F=pd.read_pickle("proto/F1.pkl"); p1=pd.read_pickle(sys.argv[1] if len(sys.argv)>1 else "proto/p1.pkl")
def smooth_feats(F,p,pre="s"):
    O=pd.DataFrame(index=F.index); O[pre+"p"]=p
    for deg,lam in [(1,0),(2,0.1),(3,0.1),(2,3)]:
        out=np.empty(len(F)); sl=np.empty(len(F))
        for pid,idx in F.groupby("patient_id",sort=False).indices.items():
            a=F.age.values[idx]-F.age.values[idx].mean(); A=np.vander(a,deg+1,increasing=True); v=p.values[idx]
            c=np.linalg.solve(A.T@A+np.diag([0]+[lam]*deg),A.T@v); out[idx]=A@c; sl[idx]=c[1]
        O[f"{pre}fit{deg}_{lam}"]=out; O[f"{pre}sl{deg}_{lam}"]=sl
    g=p.groupby(F.patient_id)
    O[pre+"pmean"]=g.transform("mean"); O[pre+"prev"]=g.shift(1); O[pre+"next"]=g.shift(-1)
    O[pre+"prev2"]=g.shift(2); O[pre+"next2"]=g.shift(-2)
    O[pre+"resid"]=p-O[f"{pre}fit2_0.1"]
    return O
S=smooth_feats(F,p1)
base=[c for c in F.columns if c not in ["Index","patient_id","cohort","gene","rater_id","target","is_test","fold"]]
G=pd.concat([F,S],axis=1); feats=base+list(S.columns)
tr=G.is_test==0; te=G.is_test==1
oof=pd.Series(np.nan,index=G.index); tep=[]
for k in range(5):
    a=G[tr&(G.fold!=k)]; b=G[G.fold==k]
    m=lgb.LGBMRegressor(n_estimators=3000,learning_rate=0.02,num_leaves=31,min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,verbose=-1)
    m.fit(a[feats],a.target,eval_set=[(b[feats],b.target)],callbacks=[lgb.early_stopping(150,verbose=False)])
    oof[b.index]=m.predict(b[feats]); tep.append(m.predict(G.loc[te,feats]))
oof[te]=np.mean(tep,axis=0)
print("stage2 OOF",rmse(oof[tr],G.target[tr]))
out=sys.argv[2] if len(sys.argv)>2 else "proto/p2.pkl"; oof.to_pickle(out)
