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
R={k:[] for k in ('C','e_tw','e_bc','e_h')}; demos=[]; store=[]; import clim_fit; cz=np.load('clim_coef.npz'); cidx=cz['idx']; ccoef=cz['coef'].astype(np.float64)
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
    x=clim_fit.basis(st.times[t]); clim=np.full(tr.shape,np.nan); clim.ravel()[cidx]=ccoef@x
    clim=np.where(np.isfinite(clim),clim,np.nanmean(clim))
    # 异常场流程: 过去帧/当前观测各减同时刻气候态, 复用 tw+bc, 再加回气候态
    pa={}
    for k in past:
        xk=clim_fit.basis(st.times[t-k]); ck=np.full(tr.shape,np.nan); ck.ravel()[cidx]=ccoef@xk
        pa[k]=past[k]-np.where(np.isfinite(ck),ck,np.nanmean(ck))
    fa=np.where(known,field-clim,np.nan)
    offa=_offsets(fa,known,pa)
    Ba=_complete(_weighted_background(pa,offa,lambda l:1.0/l),oc)
    Ca=_boundary_correct(Ba,fa,known,oc)+clim
    Ha=np.nan_to_num(harmonic_fill(fa,known,oc))+clim
    store.append((tr,hole,C,Bc,H,known,clim,Ca,Ha))
    z=(C[hole]==0).mean()
    demos.append((z,tr,field,hole,C,Bf,Bc,H))
R={k:np.concatenate(v) for k,v in R.items()}


rm=lambda e:np.sqrt(np.mean(e**2))
names=['tw+bc','clim only','clim+harmonic anom','anom tw+bc','hybrid(C<=2: anom tw+bc)','hybrid(C<=2: clim+harm)','hybrid(C=0: clim+harm)']
res={n:[] for n in names}; Cs=[]; bias={n:[] for n in names}
for tr,hole,C,Bc,H,known,clim,Ca,Ha in store:
    F={'tw+bc':Bc,'clim only':clim,'clim+harmonic anom':Ha,'anom tw+bc':Ca,
       'hybrid(C<=2: anom tw+bc)':np.where(C<=2,Ca,Bc),'hybrid(C<=2: clim+harm)':np.where(C<=2,Ha,Bc),'hybrid(C=0: clim+harm)':np.where(C==0,Ha,Bc)}
    for n in names: res[n].append((F[n]-tr)[hole])
    Cs.append(C[hole])
Cc=np.concatenate(Cs); lines=['%-26s %7s %7s %7s %7s %7s | bias C=0'%('方法','总','C=0','C1-2','C3-5','C>=6')]
for n in names:
    e=np.concatenate(res[n]); lines.append('%-26s %7.3f %7.3f %7.3f %7.3f %7.3f | %+.3f'%(n,rm(e),rm(e[Cc==0]),rm(e[(Cc>=1)&(Cc<3)]),rm(e[(Cc>=3)&(Cc<6)]),rm(e[Cc>=6]),e[Cc==0].mean()))
print('\n'.join(lines)); open('clim_results.txt','w').write('\n'.join(lines))
# 对比图: C=0 占比最高的样本
import matplotlib.pyplot as plt
j=int(np.argmax([ (d[4][d[1]]==0).mean() for d in store]))
tr,hole,C,Bc,H,known,clim,Ca,Ha=store[j]; oc_=oc
lo,hi=np.nanpercentile(tr[oc_&np.isfinite(tr)],[2,98])
fig,ax=plt.subplots(2,4,figsize=(20,9))
P=[(tr,'truth','RdYlBu_r',lo,hi),(np.where(known,tr,np.nan),'input','RdYlBu_r',lo,hi),(np.where(hole,C,np.nan),'C in hole','viridis',0,48),(np.where(hole,Ha,tr),'clim+harmonic anom','RdYlBu_r',lo,hi),
   (np.where(hole,Bc-tr,np.nan),'tw+bc error','RdBu_r',-2,2),(np.where(hole,clim-tr,np.nan),'clim only error','RdBu_r',-2,2),(np.where(hole,Ha-tr,np.nan),'clim+harm anom error','RdBu_r',-2,2),(np.where(hole,Ca-tr,np.nan),'anom tw+bc error','RdBu_r',-2,2)]
for a,(d_,ti,cm,l,h) in zip(ax.ravel(),P):
    a.imshow(np.where(oc_,d_,np.nan),cmap=cm,vmin=l,vmax=h); a.set_title(ti); a.axis('off')
plt.tight_layout(); plt.savefig('clim_demo.png',dpi=80)
