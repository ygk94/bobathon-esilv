import pandas as pd, numpy as np
X=pd.read_csv("data/X_train.csv"); y=pd.read_csv("data/y_train.csv")
d=X.merge(y,on="Index")
d["t"]=d.age-d.age_at_diagnosis
# oracle: per-patient poly fit of target vs age
for deg in [1,2,3]:
    res=[]
    for pid,g in d.groupby("patient_id"):
        c=np.polyfit(g.age,g.target,deg); res.append(g.target-np.polyval(c,g.age))
    r=np.concatenate(res); print("deg",deg,"rmse",np.sqrt((r**2).mean()))
# target vs t over population
d["tb"]=pd.cut(d.t,[-1,0,2,4,6,8,10,15,20,40])
print(d.groupby("tb").target.agg(["mean","std","count"]))
# is age sorted within patient?
print("sorted",d.groupby("patient_id").age.apply(lambda s:s.is_monotonic_increasing).mean())
# index ordering test
T=pd.read_csv("data/X_test.csv"); print(T.Index.min(),T.Index.max(), T.head(15).to_string())
# rater bias: off - target by rater
m=d.off.notna(); print((d.off-d.target)[m].groupby(d.rater_id[m]).mean().describe())
print("off-target vs tsio corr", np.corrcoef((d.off-d.target)[d.time_since_intake_off.notna()&m], d.time_since_intake_off[d.time_since_intake_off.notna()&m])[0,1])
mm=m&d.time_since_intake_off.isna(); print("off-target when no intake time: mean",(d.off-d.target)[mm].mean(), "std",(d.off-d.target)[mm].std())
mm=m&d.time_since_intake_off.notna(); print("off-target with intake time: mean",(d.off-d.target)[mm].mean(), "std",(d.off-d.target)[mm].std())
