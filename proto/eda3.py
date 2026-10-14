import sys; sys.path.insert(0,"proto")
from common import *
d=D[D.is_test==0].copy()
d["r_on"]=d.on/d.target; d["r_off"]=d.off/d.target; d["e_off"]=d.off-d.target
print("ON ratio by tsio_on"); print(d.groupby(pd.cut(d.time_since_intake_on,[-.1,.25,.5,.75,1,1.5,2,2.5,3,4,5,7]),observed=True).r_on.agg(["mean","std","count"]).round(3))
print("ON ratio when tsio_on NaN", d.r_on[d.time_since_intake_on.isna()].describe().round(3))
print("OFF err by tsio_off"); print(d.groupby(pd.cut(d.time_since_intake_off,[0,8,10,12,14,16,18,25]),observed=True).r_off.agg(["mean","std","count"]).round(3))
print("OFF ratio when tsio_off NaN", d.r_off[d.time_since_intake_off.isna()].describe().round(3))
print("OFF ratio by ledd missing", d.groupby(d.ledd.isna()).r_off.mean())
print("OFF ratio by tsio NaN & ledd NaN", d.groupby([d.time_since_intake_off.isna(),d.ledd.isna()]).r_off.agg(["mean","std","count"]))
print("ON ratio by ledd bins"); print(d.groupby(pd.cut(d.ledd,[0,300,500,700,900,2000]),observed=True).r_on.agg(["mean","std"]))
# t (disease duration) effect on ratio
print(d.groupby(pd.cut(d.t,[-1,2,5,10,40]),observed=True).r_on.agg(["mean","std"]))
# off noise: additive or multiplicative? std of e_off by target bins
print(d.groupby(pd.cut(d.target,[0,20,30,40,50,60,120]),observed=True).e_off.agg(["mean","std"]))
print("ledd missing & on present", ((d.ledd.isna())&d.on.notna()).mean(), "tsio_on missing & on present", (d.time_since_intake_on.isna()&d.on.notna()).mean())
# integer values?
print("on int", (d.on.dropna()%1==0).mean(), "off int",(d.off.dropna()%1==0).mean())
