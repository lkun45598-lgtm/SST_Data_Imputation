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
    store.append((tr,hole,C,Bc,Bf,known))
    z=(C[hole]==0).mean()
    demos.append((z,tr,field,hole,C,Bf,Bc,H))
R={k:np.concatenate(v) for k,v in R.items()}


from scipy.ndimage import gaussian_filter, distance_transform_edt as edt
import matplotlib.pyplot as plt
def lowfrac(e,hole,sig=8):
    w=hole.astype(float); n=gaussian_filter(w,sig)+1e-9
    lo=gaussian_filter(np.where(hole,e,0),sig)/n
    return (lo[hole]**2).mean()/(e[hole]**2).mean()
def corrlen(e,hole):
    m=hole&np.roll(hole,5,1)&np.roll(hole,10,1)&np.roll(hole,20,1)
    e0=e-e[hole].mean(); out=[]
    for d in (5,10,20): out.append((e0[m]*np.roll(e0,d,1)[m]).mean()/(e0[m]**2).mean())
    return out
rows={'tw':[], 'tw+bc':[]}; imp=[]; dist_e={'tw':[],'tw+bc':[]}; Ds=[]; Cs=[]
for tr,hole,C,Bc,Bf,known in store:
    for n,F in (('tw',Bf),('tw+bc',Bc)):
        e=np.where(hole,np.nan_to_num(F)-np.nan_to_num(tr),0); rows[n].append((lowfrac(e,hole),*corrlen(e,hole)))
        dist_e[n].append(e[hole])
    Ds.append(edt(~known)[hole]); Cs.append(C[hole])
Dc=np.concatenate(Ds); Cc=np.concatenate(Cs); rm=lambda e:np.sqrt(np.mean(e**2))
out=['误差结构 (均值): 低通(σ=8)方差占比 | 空间自相关 lag5/10/20px']
for n in rows: out.append('  %-6s %.2f | %s'%(n,*(np.mean([r[0] for r in rows[n]]),' '.join('%.2f'%np.mean([r[i] for r in rows[n]]) for i in (1,2,3)))))
for n in ('tw','tw+bc'): dist_e[n]=np.concatenate(dist_e[n])
eb=[0,3,10,20,50,1000]; out.append('RMSE 随到最近当前观测的距离: 距离bin | tw(C>0) bc(C>0) | tw(C=0) bc(C=0)')
for a,b in zip(eb[:-1],eb[1:]):
    m=(Dc>a)&(Dc<=b); p=m&(Cc>0); z=m&(Cc==0)
    f=lambda n,k: rm(dist_e[n][k]) if k.sum()>50 else float('nan')
    out.append('  (%d,%d] n0=%d | %.3f %.3f | %.3f %.3f'%(a,b,z.sum(),f('tw',p),f('tw+bc',p),f('tw',z),f('tw+bc',z)))
print('\n'.join(out)); open('bc_diag.txt','w').write('\n'.join(out))
fig,ax=plt.subplots(1,2,figsize=(11,4))
for k,(n,c) in enumerate((('tw','C4'),('tw+bc','C1'))):
    for j,(lab,msk) in enumerate((('C>0',Cc>0),('C=0',Cc==0))):
        mids=[];v=[]
        for a,b in zip(eb[:-1],eb[1:]):
            m=msk&(Dc>a)&(Dc<=b)
            if m.sum()>50: mids.append(b if b<1000 else 80); v.append(rm(dist_e[n][m]))
        ax[j].plot(mids,v,'o-',label=n); ax[j].set_title('RMSE vs dist to obs, '+lab); ax[j].set_xlabel('px'); ax[j].legend()
plt.tight_layout(); plt.savefig('bc_diag.png',dpi=100)
