import sys; sys.path.insert(0,"proto")
from common import *; from z import z_features
F=pd.read_pickle("proto/F1.pkl")
fm=((F.fold!=0)&(F.is_test==0)).values
Z=z_features(F,fm); b=(F.fold==0).values
for c in Z.columns: 
    m=b&Z[c].notna().values; print(c, round(rmse(Z[c][m],F.target[m]),3), Z[c][m].describe()[["min","max"]].values.round(2))
