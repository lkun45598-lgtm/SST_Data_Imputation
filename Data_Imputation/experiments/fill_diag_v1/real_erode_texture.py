"""1) 真实帧下一小时参照: 扫 erode 像素数; 2) 观测/填充边界两侧纹理(局部高频标准差)之比。"""
import sys, numpy as np
sys.path.insert(0,'/data1/user/lz/SST_Data_Imputation/Data_Imputation')
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import binary_erosion, label as cclabel, sum as ndsum, gaussian_filter, distance_transform_edt as edt
from datasets.hourly_store import HourlyStore
from preprocessing.fill_compare import VAL_START, VAL_END
from preprocessing.fill_temporal import _offsets,_weighted_background,_complete,_boundary_correct
from preprocessing.fill_methods import _harmonic_solve
a=open('seq_screened.py').read(); exec(a[a.index('def screened'):a.index('R={k:[]')])
b=open('seq_robust.py').read(); exec(b[b.index('def source'):b.index('cfg=')])
N=int(sys.argv[1]); rng=np.random.default_rng(11); L=48
st=HourlyStore(); oc=st.ocean
c=[s for s in st.slot_range(VAL_START,VAL_END) if st.present[s] and 0.12<=st.coverage[s]<=0.4 and s>=L and st.present[s-L:s+2].all()]
mon=np.array([st.times[s].month for s in c]); pick=[]
for m in range(1,13):
    cm=[s for s,mm in zip(c,mon) if mm==m]
    if cm: pick+=list(rng.choice(cm,min(len(cm),max(1,N//12)),replace=False))
ER=[0,1,2,3,5,8]; names=['tw']+[f'bc erode{e}' for e in ER]
E={n:[] for n in names}; Cs=[]; TX=[]
def hp_std(a,valid,sig=2.0):
    num=gaussian_filter(np.where(valid,a,0.),sig); den=gaussian_filter(valid.astype(float),sig)+1e-9
    hp=np.where(valid,a-num/den,0.); v=gaussian_filter(hp**2,sig)/den
    return np.sqrt(np.maximum(v,0))
for t in pick:
    t=int(t); tr=st.frame(t); known=np.isfinite(tr)&oc; field=np.where(known,tr,np.nan)
    ref=st.frame(t+1); m=oc&~known&np.isfinite(ref)
    if m.sum()<2000: continue
    past={l:st.frame(t-l) for l in range(1,L+1)}
    C=(np.stack([np.isfinite(past[l]) for l in past])&oc).sum(0)
    offs=_offsets(field,known,past); Bf=_complete(_weighted_background(past,offs,lambda l:1.0/l),oc)
    F={'tw':Bf}
    for e in ER: F[f'bc erode{e}']=robust_bc(Bf,field,known,oc,e,0)
    for n in names: E[n].append((np.nan_to_num(F[n])-ref)[m])
    Cs.append(C[m])
    # 纹理: 观测侧(距边界1-6px内)与填充侧(同距离)的局部高频标准差，以 erode2 的填充结果为准
    full=np.where(known,np.nan_to_num(tr),np.nan_to_num(F['bc erode2']))
    valid=oc; sd=hp_std(full,valid)
    dk=edt(known); dm=edt(~known)   # dk: 观测像元到缺口距离; dm: 缺口像元到观测距离
    for lo,hi in ((1,4),(4,10),(10,25),(25,60)):
        so=sd[known&(dk>=lo)&(dk<hi)&oc]; sf=sd[(~known)&(dm>=lo)&(dm<hi)&oc]
        if len(so)>500 and len(sf)>500: TX.append((lo,so.mean(),sf.mean()))
Cc=np.concatenate(Cs); rm=lambda e:np.sqrt(np.mean(e**2))
out=['frames %d, ref px %d'%(len(Cs),len(Cc)),'%-12s %6s %7s | C=0 | C1-5 | C>=6'%('方法','RMSE','bias')]
for n in names:
    e=np.concatenate(E[n]); z=Cc==0; mid=(Cc>=1)&(Cc<6); hi=Cc>=6
    out.append('%-12s %.3f %+.3f | %.3f | %.3f | %.3f'%(n,rm(e),e.mean(),rm(e[z]),rm(e[mid]),rm(e[hi])))
out.append('纹理(局部高频标准差 K, erode2 填充结果): 距边界 | 观测侧 | 填充侧 | 比值 填充/观测')
for lo in (1,4,10,25):
    r=[(o,f) for l,o,f in TX if l==lo]
    if r: o=np.mean([x[0] for x in r]); f=np.mean([x[1] for x in r]); out.append('  [%d,..) px | %.3f | %.3f | %.2f  (n=%d 帧)'%(lo,o,f,f/o,len(r)))
print('\n'.join(out)); open('erode_texture_results.txt','w').write('\n'.join(out))
