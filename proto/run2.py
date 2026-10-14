import sys; sys.path.insert(0,"proto")
from common import *; from z import z_features
import lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
F=pd.read_pickle("proto/F1.pkl")
base=[c for c in F.columns if c not in ["Index","patient_id","cohort","gene","rater_id","target","is_test","fold"]]
tr_idx=F.index[F.is_test==0]
oof=pd.Series(np.nan,index=tr_idx); oofz={}
for k in range(5):
    fm=(F.fold!=k)&(F.is_test==0)
    Z=z_features(F,fm.values)
    G=pd.concat([F,Z],axis=1); feats=base+list(Z.columns)
    a=G[fm]; b=G[F.fold==k]
    for c in ["zfit_d2_l1","zfit_d1_l1","zfit_d2_l20"]: oofz.setdefault(c,[]).append(rmse(b[c].fillna(37),b.target))
    m=lgb.LGBMRegressor(n_estimators=3000,learning_rate=0.03,num_leaves=63,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1)
    m.fit(a[feats],a.target,eval_set=[(b[feats],b.target)],callbacks=[lgb.early_stopping(100,verbose=False)])
    oof[b.index]=m.predict(b[feats]); print(k,rmse(oof[b.index],b.target),flush=True)
print({k:np.round(v,3) for k,v in oofz.items()})
print("OOF",rmse(oof,F.target[tr_idx]))
