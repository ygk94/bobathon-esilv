import sys; sys.path.insert(0,"proto")
from common import *
d=D[(D.is_test==0)&(D.target>5)].copy()
r=(d.off/d.target).dropna(); print(np.histogram(r,bins=np.arange(0,2.01,0.1)))
r=(d.on/d.target).dropna(); print(np.histogram(r,bins=np.arange(0,2.01,0.1)))
# rater-corrected
rf=(d.off/d.target).groupby(d.rater_id).median(); rr=(d.off/d.target)/d.rater_id.map(rf)
print("corr r_off std",rr.std(), np.histogram(rr.dropna(),bins=np.arange(0,2.01,0.1))[0])
# does r_off relate to on presence/ledd/tsio_on at same visit?
d["rr"]=rr
print(d.groupby(d.on.notna()).rr.agg(["mean","std","count"]))
print(d.groupby(pd.qcut(d.ledd,5)).rr.agg(["mean","std"]))
print(d.groupby(pd.qcut(d.age,5)).rr.agg(["mean","std"]))
print(d.groupby(pd.qcut(d.target,5)).rr.agg(["mean","std"]))
print(d.groupby("gene",dropna=False).rr.agg(["mean","std"]))
print(d.groupby("cohort").rr.agg(["mean","std"]))
# additive vs multiplicative: std of off-target*rf by target
e=d.off-d.target*d.rater_id.map(rf)
print(e.groupby(pd.qcut(d.target,6)).agg(["mean","std"]))
# serial correlation of rr within patient
d["rr_prev"]=d.groupby("patient_id").rr.shift(1)
print("lag1 corr",d[["rr","rr_prev"]].dropna().corr().iloc[0,1])
