import sys; sys.path.insert(0,"proto")
from common import *; from z import z_features
import lightgbm as lgb, warnings, pickle; warnings.filterwarnings("ignore")
F=pd.read_pickle("proto/F1.pkl")
base=[c for c in F.columns if c not in ["Index","patient_id","cohort","gene","rater_id","target","is_test","fold"]]
te=F.is_test==1
pred=pd.Series(np.nan,index=F.index); tep=[]
Zs={}
for k in range(5):
    fm=(F.fold!=k)&(F.is_test==0)
    Z=z_features(F,fm.values); Zs[k]=Z
    G=pd.concat([F,Z],axis=1); feats=base+list(Z.columns)
    a=G[fm]; b=G[F.fold==k]
    m=lgb.LGBMRegressor(n_estimators=3000,learning_rate=0.03,num_leaves=63,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1)
    m.fit(a[feats],a.target,eval_set=[(b[feats],b.target)],callbacks=[lgb.early_stopping(100,verbose=False)])
    pred[b.index]=m.predict(b[feats]); tep.append(m.predict(G.loc[te,feats]))
pred[te]=np.mean(tep,axis=0)
tr=F.is_test==0
print("stage1 OOF",rmse(pred[tr],F.target[tr]))
pickle.dump(Zs,open("proto/Zs.pkl","wb")); pred.to_pickle("proto/p1.pkl")
# smoothing test
for deg in [1,2,3]:
    for lam in [0,0.1,1,10]:
        sm=pred.copy()
        for pid,idx in F.groupby("patient_id",sort=False).indices.items():
            a=F.age.values[idx]-F.age.values[idx].mean(); A=np.vander(a,deg+1,increasing=True); v=pred.values[idx]
            c=np.linalg.solve(A.T@A+np.diag([0]+[lam]*deg),A.T@v); sm.iloc[idx]=A@c
        print("smooth deg",deg,"lam",lam,rmse(sm[tr],F.target[tr]))
