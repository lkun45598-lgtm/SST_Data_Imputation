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
    store.append((tr,hole,C,Bc,H,known))
    z=(C[hole]==0).mean()
    demos.append((z,tr,field,hole,C,Bf,Bc,H))
R={k:np.concatenate(v) for k,v in R.items()}

rm=lambda e:np.sqrt(np.mean(e**2))
def lap(a): return a[1:-1,1:-1]*4-a[:-2,1:-1]-a[2:,1:-1]-a[1:-1,:-2]-a[1:-1,2:]
def rough(fill,tr,hole):
    m=hole[1:-1,1:-1]&hole[:-2,1:-1]&hole[2:,1:-1]&hole[1:-1,:-2]&hole[1:-1,2:]
    return np.sqrt(np.mean(lap(fill)[m]**2))/np.sqrt(np.mean(lap(tr)[m]**2))
cfgs=[('tw+bc',None)]+[(f'blend c={c}',c) for c in (2,5,10,20,40)]+[('harmonic',-1)]
lines=[]
for name,c in cfgs:
    ee=[];C0=[];Cl=[];rg=[]
    for tr,hole,C,Bc,H,known in store:
        if c is None: F=Bc
        elif c==-1: F=H
        else:
            a=C/(C+c); F=a*Bc+(1-a)*H
        e=(F-tr)[hole]; ee.append(e); C0.append(C[hole]); rg.append(rough(np.nan_to_num(F),np.nan_to_num(tr),hole))
    e=np.concatenate(ee); Cc=np.concatenate(C0)
    lines.append('%-14s 总 %.3f | C=0 %.3f | C1-2 %.3f | C3-5 %.3f | C>=6 %.3f | 粗糙度比(1=同真值) %.2f'%(name,rm(e),rm(e[Cc==0]),rm(e[(Cc>=1)&(Cc<3)]),rm(e[(Cc>=3)&(Cc<6)]),rm(e[Cc>=6]),np.mean(rg)))
print('\n'.join(lines)); open('blend_results.txt','w').write('\n'.join(lines))
