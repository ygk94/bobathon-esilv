import sys; sys.path.insert(0,"proto")
from common import *
import lightgbm as lgb, time
t0=time.time(); F=add_features(D); print("feat",time.time()-t0)
F.to_pickle("proto/F1.pkl")
raw=["sexM","age_at_diagnosis","age","ledd","time_since_intake_on","time_since_intake_off","on","off","cohort_c","gene_c","rater_id_c"]
allf=[c for c in F.columns if c not in ["Index","patient_id","cohort","gene","rater_id","target","is_test","fold"]]
tr=F[F.is_test==0]
for name,feats in [("raw",raw),("all",allf)]:
    oof=np.zeros(len(tr))
    for k in range(5):
        a=tr.fold!=k; b=tr.fold==k
        m=lgb.LGBMRegressor(n_estimators=2000,learning_rate=0.03,num_leaves=63,min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1)
        m.fit(tr.loc[a,feats],tr.loc[a,"target"],eval_set=[(tr.loc[b,feats],tr.loc[b,"target"])],callbacks=[lgb.early_stopping(100,verbose=False)])
        oof[b.values]=m.predict(tr.loc[b,feats])
    print(name,len(feats),"oof rmse",rmse(oof,tr.target), [round(rmse(oof[(tr.fold==k).values],tr.target[tr.fold==k]),3) for k in range(5)])
