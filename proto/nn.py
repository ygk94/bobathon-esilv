import sys, time; sys.path.insert(0,"proto")
from common import *
import torch, torch.nn as nn, argparse
ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,default=0); ap.add_argument("--epochs",type=int,default=120)
ap.add_argument("--d",type=int,default=128); ap.add_argument("--layers",type=int,default=4); ap.add_argument("--p1",default="")
ap.add_argument("--out",default="proto/nn0.pkl"); ap.add_argument("--drop",type=float,default=0.15); ap.add_argument("--folds",default="0,1,2,3,4")
args=ap.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)
torch.set_num_threads(8)
F=D.copy(); F["t"]=F.age-F.age_at_diagnosis
num=["age","t","age_at_diagnosis","ledd","time_since_intake_on","time_since_intake_off","on","off","sexM"]
if args.p1:
    p=pd.read_pickle(args.p1); F["p1"]=p.values; num.append("p1")
F["cohortB"]=(F.cohort=="B").astype(float); num.append("cohortB")
stats={c:(np.nanmean(F[c]),np.nanstd(F[c])+1e-6) for c in num}
rater=F.rater_id.astype("category").cat.codes.values; gene=F.gene.astype("category").cat.codes.values+1  # NaN->0
pids=F.patient_id.values; groups=F.groupby("patient_id",sort=False).indices
P=list(groups.keys()); L=12; nf=len(num)
Xn=np.zeros((len(P),L,nf),np.float32); Xm=np.zeros((len(P),L,nf),np.float32); R=np.zeros((len(P),L),np.int64); Gn=np.zeros((len(P),L),np.int64)
Y=np.zeros((len(P),L),np.float32); M=np.zeros((len(P),L),bool); ROW=np.full((len(P),L),-1); pfold=np.zeros(len(P),int); ptest=np.zeros(len(P),bool)
for i,pid in enumerate(P):
    idx=groups[pid]; n=len(idx)
    for j,c in enumerate(num):
        v=F[c].values[idx]; mu,sd=stats[c]; Xm[i,:n,j]=~np.isnan(v); Xn[i,:n,j]=np.nan_to_num((v-mu)/sd)
    # patient-relative age
    R[i,:n]=rater[idx]+1; Gn[i,:n]=gene[idx]; Y[i,:n]=np.nan_to_num(F.target.values[idx]); M[i,:n]=True; ROW[i,:n]=idx
    pfold[i]=F.fold.values[idx[0]]; ptest[i]=F.is_test.values[idx[0]]==1
# relative age feature
rel=np.zeros((len(P),L,1),np.float32)
for i,pid in enumerate(P):
    idx=groups[pid]; a=F.age.values[idx]; rel[i,:len(idx),0]=(a-a.mean())/5
Xn=np.concatenate([Xn,rel],2); Xm=np.concatenate([Xm,np.ones_like(rel)],2)
YS=20.0; YM=37.5
class Net(nn.Module):
    def __init__(s,nf,d,layers):
        super().__init__()
        s.inp=nn.Sequential(nn.Linear(2*nf,d),nn.GELU(),nn.Linear(d,d)); s.re=nn.Embedding(52,d); s.ge=nn.Embedding(6,d)
        el=nn.TransformerEncoderLayer(d,4,4*d,dropout=0.1,batch_first=True,norm_first=True)
        s.enc=nn.TransformerEncoder(el,layers); s.head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.GELU(),nn.Linear(d,1))
    def forward(s,x,m,r,g,mask):
        h=s.inp(torch.cat([x*m,m],-1))+s.re(r)+s.ge(g)
        h=s.enc(h,src_key_padding_mask=~mask); return s.head(h).squeeze(-1)
T=lambda a:torch.from_numpy(a)
Xn_t,Xm_t,R_t,G_t,Y_t,M_t=map(T,[Xn,Xm,R,Gn,Y,M])
pred=np.full(len(F),np.nan); tests=[]
obs_cols=[num.index(c) for c in ["on","off","ledd","time_since_intake_on","time_since_intake_off"]]
def run(model,ids,train=False):
    x=Xn_t[ids]; m=Xm_t[ids].clone()
    if train and args.drop>0:
        dm=(torch.rand(m.shape[0],m.shape[1],len(obs_cols))<args.drop).float()
        m[:,:,obs_cols]=m[:,:,obs_cols]*(1-dm)
    return model(x,m,R_t[ids],G_t[ids],M_t[ids])
te_ids=np.where(ptest)[0]
for k in [int(f) for f in args.folds.split(",")]:
    t0=time.time()
    tr_ids=np.where((~ptest)&(pfold!=k))[0]; va_ids=np.where((~ptest)&(pfold==k))[0]
    model=Net(Xn.shape[2],args.d,args.layers); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-2)
    steps=args.epochs*int(np.ceil(len(tr_ids)/64)); sch=torch.optim.lr_scheduler.OneCycleLR(opt,1e-3,total_steps=steps,pct_start=0.1)
    for ep in range(args.epochs):
        model.train(); perm=np.random.permutation(tr_ids)
        for b in range(0,len(perm),64):
            ids=perm[b:b+64]; out=run(model,ids,True); mm=M_t[ids]
            loss=(((out-(Y_t[ids]-YM)/YS)**2)[mm]).mean(); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step()
        if ep%20==19 or ep==args.epochs-1:
            model.eval()
            with torch.no_grad(): o=run(model,va_ids).numpy()*YS+YM
            mm=M[va_ids]; print(k,ep,round(rmse(o[mm],Y[va_ids][mm]),4),round(time.time()-t0),flush=True)
    model.eval()
    with torch.no_grad():
        o=run(model,va_ids).numpy()*YS+YM; pred[ROW[va_ids][M[va_ids]]]=o[M[va_ids]]
        o=run(model,te_ids).numpy()*YS+YM; tp=np.full(len(F),np.nan); tp[ROW[te_ids][M[te_ids]]]=o[M[te_ids]]; tests.append(tp)
tr=(F.is_test==0).values&~np.isnan(pred)
print("NN OOF",rmse(pred[tr],F.target.values[tr]))
pt=np.nanmean(tests,0); pred[F.is_test.values==1]=pt[F.is_test.values==1]
pd.Series(pred,index=F.index).to_pickle(args.out)
