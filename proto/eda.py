import pandas as pd, numpy as np
X=pd.read_csv("data/X_train.csv"); y=pd.read_csv("data/y_train.csv"); T=pd.read_csv("data/X_test.csv")
d=X.merge(y,on="Index")
print(d.describe(include="all").T.to_string())
print(d.isna().mean().round(3).to_string())
print("patients train/test",d.patient_id.nunique(),T.patient_id.nunique())
print(d.groupby("patient_id").size().describe())
print(d[["on","off","target","ledd","time_since_intake_on","time_since_intake_off","age","age_at_diagnosis"]].corr().round(2).to_string())
print(d.cohort.value_counts(), d.gene.value_counts(dropna=False), d.rater_id.nunique(), T.rater_id.isin(d.rater_id).mean())
# is target constant-ish within patient? residual of off vs target
m=d.off.notna()
print("rmse target vs off", np.sqrt(((d.target-d.off)[m]**2).mean()))
print("std target",d.target.std())
# sorted visits check
print(d.head(30).to_string())
