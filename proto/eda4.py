import sys; sys.path.insert(0,"proto")
from common import *
d=D[D.is_test==0].copy(); d=d[d.target>1]
d["t"]=d.age-d.age_at_diagnosis
d["r_on"]=d.on/d.target; d["r_off"]=d.off/d.target
for c in ["r_on","r_off"]:
    s=d[[c,"patient_id","rater_id"]].dropna()
    pm=s.groupby("patient_id")[c].transform("mean"); rm=s.groupby("rater_id")[c].transform("mean")
    print(c,"total var",s[c].var().round(4),"within-patient var",(s[c]-pm).var().round(4),"rater-mean var",rm.var().round(5))
# off additive error decomposition
s=d[["off","target","patient_id","rater_id","time_since_intake_off","ledd"]].dropna(subset=["off"]).copy()
s["e"]=s.off-s.target
print("e total",s.e.var(),"within pat",(s.e-s.groupby("patient_id").e.transform("mean")).var())
print("r_off by ledd-present & tsio-present"); print(d.groupby([d.ledd.isna(),d.time_since_intake_off.isna()]).r_off.agg(["mean","std","count"]))
print("r_off by t"); print(d.groupby(pd.cut(d.t,[-1,1,2,5,10,40]),observed=True).r_off.agg(["mean","std"]))
print("r_on by t"); print(d.groupby(pd.cut(d.t,[-1,1,2,5,10,40]),observed=True).r_on.agg(["mean","std"]))
print("ledd vs t"); print(d.groupby(pd.cut(d.t,[-1,1,2,5,10,40]),observed=True).ledd.agg(["mean","count"]), d.groupby(pd.cut(d.t,[-1,1,2,5,10,40]),observed=True).size())
print("rater r_off means", s.assign(r=s.off/s.target).groupby("rater_id").r.mean().round(3).sort_values().values)
