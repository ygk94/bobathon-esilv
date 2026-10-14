import sys; sys.path.insert(0,"proto")
from common import *
from scipy.optimize import nnls
files=sys.argv[1:]
P=np.column_stack([pd.read_pickle(f).values for f in files]); tr=(D.is_test==0).values; y=D.target.values
for f,c in zip(files,P.T): print(f"{f:28s} OOF {rmse(c[tr],y[tr]):.4f}")
print("corr\n",np.round(np.corrcoef(P[tr].T),3))
def smooth(p,deg=2,lam=0.1):
    s=p.copy()
    for pid,idx in D.groupby("patient_id",sort=False).indices.items():
        a=D.age.values[idx]-D.age.values[idx].mean(); A=np.vander(a,deg+1,increasing=True)
        c=np.linalg.solve(A.T@A+np.diag([0]+[lam]*deg),A.T@p[idx]); s[idx]=A@c
    return s
# weights: fit on folds != k, evaluate on k (honest)
fo=D.fold.values; ens=np.zeros(len(D))
for k in range(5):
    a=tr&(fo!=k); b=fo==k; w,_=nnls(P[a],y[a]); ens[b]=P[b]@w
w,_=nnls(P[tr],y[tr]); print("weights",np.round(w,3),"sum",w.sum().round(3))
ens[~tr]=P[~tr]@w
print("ENS (cv weights) OOF",round(rmse(ens[tr],y[tr]),4))
for deg,lam in [(2,0.1),(2,1),(3,0.1)]:
    s=smooth(ens,deg,lam); print(f"ENS+smooth d{deg} l{lam}",round(rmse(s[tr],y[tr]),4))
s=smooth(ens,2,0.1); pd.Series(s,index=D.index).to_pickle("proto/ens_final.pkl")
