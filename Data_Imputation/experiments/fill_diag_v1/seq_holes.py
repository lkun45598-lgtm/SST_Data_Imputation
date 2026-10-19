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
R={k:[] for k in ('C','e_tw','e_bc','e_h')}; demos=[]
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
    z=(C[hole]==0).mean()
    demos.append((z,tr,field,hole,C,Bf,Bc,H))
R={k:np.concatenate(v) for k,v in R.items()}
rm=lambda e:np.sqrt(np.mean(e**2)) if len(e) else np.nan
bins=[0,1,3,6,12,24,49]; lab=['0','1-2','3-5','6-11','12-23','24-48']
print('%-6s %6s %7s %7s %7s %8s %8s'%('C','share','tw','tw+bc','harm','bias_tw','bias_bc'))
for a,b,l in zip(bins[:-1],bins[1:],lab):
    m=(R['C']>=a)&(R['C']<b)
    print('%-6s %5.1f%% %7.3f %7.3f %7.3f %8.3f %8.3f'%(l,100*m.mean(),rm(R['e_tw'][m]),rm(R['e_bc'][m]),rm(R['e_h'][m]),R['e_tw'][m].mean(),R['e_bc'][m].mean()))
print('总 RMSE tw %.3f bc %.3f harm %.3f'%(rm(R['e_tw']),rm(R['e_bc']),rm(R['e_h'])))
demos.sort(key=lambda d:-d[0])
for j,d in enumerate(demos[:2]):
    z,tr,field,hole,C,Bf,Bc,H=d
    vmin,vmax=np.nanpercentile(tr[oc&np.isfinite(tr)],[2,98])
    fig,ax=plt.subplots(2,4,figsize=(20,9))
    P=[(tr,'truth','RdYlBu_r',vmin,vmax),(field,'input (hole=blank)','RdYlBu_r',vmin,vmax),(np.where(hole,C,np.nan),'C inside hole','viridis',0,48),(np.where(hole,Bc,tr),'tw+bc fill','RdYlBu_r',vmin,vmax),
       (np.where(hole,H,tr),'harmonic fill','RdYlBu_r',vmin,vmax),(np.where(hole,Bf-tr,np.nan),'tw error','RdBu_r',-2,2),(np.where(hole,Bc-tr,np.nan),'tw+bc error','RdBu_r',-2,2),(np.where(hole,H-tr,np.nan),'harmonic error','RdBu_r',-2,2)]
    for a,(d_,ti,cm,lo,hi) in zip(ax.ravel(),P):
        a.imshow(np.where(oc,d_,np.nan),cmap=cm,vmin=lo,vmax=hi,origin='upper'); a.set_title(ti); a.axis('off')
    fig.suptitle('C=0 share in hole %.0f%%'%(100*z)); plt.tight_layout(); plt.savefig(f'/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/fill_diag_v1/seq_demo{j}.png',dpi=80)
