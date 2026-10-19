import sys, numpy as np
sys.path.insert(0,'/data1/user/lz/SST_Data_Imputation/Data_Imputation')
from scipy import ndimage as ndi
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from datasets.hourly_store import HourlyStore
from preprocessing.fill_compare import VAL_START, VAL_END, NEED_PAST
from preprocessing.fill_temporal import _offsets,_weighted_background,_complete,_boundary_correct
from preprocessing.fill_methods import harmonic_fill
N=int(sys.argv[1]); rng=np.random.default_rng(0); L=48
st=HourlyStore(); oc=st.ocean
cov=st.coverage; pr=st.present
def okwin(s): return s>=L and pr[s-L:s+1].all()
tg=[s for s in st.slot_range(VAL_START,VAL_END) if okwin(s) and cov[s-L:s+1].min()>=0.7]
src=[s for s in st.slot_range(__import__('datetime').datetime(2015,7,1),VAL_START) if okwin(s) and 0.3<=cov[s]<=0.6]
print('targets',len(tg),'sources',len(src),flush=True)
from scipy import sparse
from scipy.sparse.linalg import spsolve
LS=(20,40,60,100)
def screened(B,field,known,ocean,L):
    unk=ocean&~known; n=int(unk.sum()); H,W=field.shape
    idx=-np.ones((H,W),np.int64); idx[unk]=np.arange(n)
    r=np.where(known,field-B,0.0); deg=np.zeros(n); rows=[];cols=[]
    pad=lambda a,v:np.pad(a,1,constant_values=v)
    ocp,unp,idp=pad(ocean,False),pad(unk,False),pad(idx,-1)
    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
        sl=(slice(1+dy,1+dy+H),slice(1+dx,1+dx+W))
        deg+=ocp[sl][unk]; m=unp[sl][unk]; rows.append(idx[unk][m]); cols.append(idp[sl][unk][m])
    rows,cols=np.concatenate(rows),np.concatenate(cols)
    A=sparse.coo_matrix((-np.ones(len(rows)),(rows,cols)),shape=(n,n))+sparse.diags(deg+1.0/L**2)
    # 已知邻居的 delta 作为 Dirichlet 项
    b=np.zeros(n); rp=pad(r,0.0); knp=pad(known,False)
    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
        sl=(slice(1+dy,1+dy+H),slice(1+dx,1+dx+W)); m=knp[sl][unk]; b[m]+=rp[sl][unk][m]
    d=np.zeros((H,W)); d[unk]=spsolve(A.tocsc(),b)
    out=B+d; out[known]=field[known]; return out
R={k:[] for k in ('C','e_tw','e_bc','e_h')}; demos=[]; store=[]
for i in range(N):
    t=int(rng.choice(tg)); s=int(rng.choice(src))
    cl=[(~np.isfinite(st.frame(s-k)))&oc for k in range(0,L+1)]   # cl[k]: 云事件在 lag k
    tr=st.frame(t); fin=np.isfinite(tr)&oc
    hole=cl[0]&fin
    if hole.sum()<2000: continue
    known=fin&~hole; field=np.where(known,tr,np.nan)
    past={}
    for k in range(1,L+1):
        f=st.frame(t-k).copy(); f[cl[k]]=np.nan; past[k]=f
    C=(np.stack([np.isfinite(past[k]) for k in past])&oc).sum(0)
    offs=_offsets(field,known,past)
    Bf=_complete(_weighted_background(past,offs,lambda l:1.0/l),oc)
    Bc=_boundary_correct(Bf,field,known,oc)
    H=np.nan_to_num(harmonic_fill(field,known,oc))
    for k,v in (('C',C),('e_tw',Bf-tr),('e_bc',Bc-tr),('e_h',H-tr)): R[k].append(v[hole])
    store.append((tr,hole,C,Bf,known,field,{L:screened(Bf,field,known,oc,L) for L in LS}))
    z=(C[hole]==0).mean()
    demos.append((z,tr,field,hole,C,Bf,Bc,H))
R={k:np.concatenate(v) for k,v in R.items()}


rm=lambda e:np.sqrt(np.mean(e**2))
from scipy.ndimage import distance_transform_edt as edt, gaussian_filter
rows=[]; E={}; Cs=[];Ds=[]; hp={}
for tr,hole,C,Bf,known,field,SC in store:
    pass
names=['bc(λ=0)']+[f'screened L={L}' for L in LS]
for n in names: E[n]=[]
Bcs=[]
for tr,hole,C,Bf,known,field,SC in store:
    Bc=_boundary_correct(Bf,field,known,oc); tt=np.nan_to_num(tr)
    E['bc(λ=0)'].append((np.nan_to_num(Bc)-tt)[hole])
    for L in LS: E[f'screened L={L}'].append((np.nan_to_num(SC[L])-tt)[hole])
    Cs.append(C[hole]); Ds.append(edt(~known)[hole])
Cc=np.concatenate(Cs); Dc=np.concatenate(Ds)
for n in names: E[n]=np.concatenate(E[n])
out=['%-16s %6s | C=0: 全 | <=20 | 20-50 | >50 px | C>0: 全 | >50 px'%('方法','总')]
for n in names:
    e=E[n]; z=Cc==0; p=Cc>0
    out.append('%-16s %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f'%(n,rm(e),rm(e[z]),rm(e[z&(Dc<=20)]),rm(e[z&(Dc>20)&(Dc<=50)]),rm(e[z&(Dc>50)]),rm(e[p]),rm(e[p&(Dc>50)])))
print('\n'.join(out)); open('screened_results.txt','w').write('\n'.join(out))
