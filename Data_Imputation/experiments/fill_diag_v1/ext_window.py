import sys, numpy as np
sys.path.insert(0,'/data1/user/lz/SST_Data_Imputation/Data_Imputation')
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import binary_erosion, label as cclabel, sum as ndsum, distance_transform_edt as edt
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from datetime import datetime
from datasets.hourly_store import HourlyStore
from preprocessing.fill_compare import VAL_START, VAL_END
from preprocessing.fill_temporal import _offsets,_weighted_background,_complete
from preprocessing.fill_methods import _harmonic_solve
a=open('seq_screened.py').read(); exec(a[a.index('def screened'):a.index('R={k:[]')])
b=open('seq_robust.py').read(); exec(b[b.index('def source'):b.index('cfg=')])
st=HourlyStore(); oc=st.ocean; nanf=np.full(st.shape,np.nan,np.float32)
fr=lambda s: st.frame(s) if 0<=s<st.T and st.present[s] else nanf
def background(t,field,known,Lmax,step):
    past={l:fr(t-l) for l in range(1,49)}
    B=_weighted_background(past,_offsets(field,known,past),lambda l:1.0/l)
    if Lmax>48:
        ext={l:fr(t-l) for l in range(49,Lmax+1,step)}
        Be=_weighted_background(ext,_offsets(field,known,ext),lambda l:1.0/l)
        m=np.isnan(B)&np.isfinite(Be); B[m]=Be[m]
    return _complete(B,oc), np.isnan(B)
def run(t,Lmax,step,erode=2):
    tr=st.frame(t); known=np.isfinite(tr)&oc; field=np.where(known,tr,np.nan)
    Bf,_=background(t,field,known,Lmax,step); return tr,known,Bf,robust_bc(Bf,field,known,oc,erode,0)
if sys.argv[1]=='eval':
    rng=np.random.default_rng(11); N=int(sys.argv[2]); L=48
    c=[s for s in st.slot_range(VAL_START,VAL_END) if st.present[s] and 0.12<=st.coverage[s]<=0.4 and s>=L and st.present[s-L:s+2].all()]
    mon=np.array([st.times[s].month for s in c]); pick=[]
    for m in range(1,13):
        cm=[s for s,mm in zip(c,mon) if mm==m]
        if cm: pick+=list(rng.choice(cm,min(len(cm),max(1,N//12)),replace=False))
    cfg=[('48h (现状)',48,1),('7天逐6h',168,6),('14天逐6h',336,6),('30天逐12h',720,12)]
    E={n:[] for n,_,_ in cfg}; Cs=[]; fracZ=[]
    for t in pick:
        t=int(t); tr=st.frame(t); known=np.isfinite(tr)&oc; ref=st.frame(t+1); m=oc&~known&np.isfinite(ref)
        if m.sum()<2000: continue
        field=np.where(known,tr,np.nan)
        for n,Lm,sp in cfg:
            Bf,isn=background(t,field,known,Lm,sp); F=robust_bc(Bf,field,known,oc,2,0); E[n].append((np.nan_to_num(F)-ref)[m])
            if n=='48h (现状)': Cs.append(isn[m])
    z=np.concatenate(Cs); rm=lambda e:np.sqrt(np.mean(e**2))
    out=['frames %d, ref px %d, C=0 占 %.1f%%'%(len(Cs),len(z),100*z.mean()),'%-12s %6s %7s | C=0(48h) RMSE bias | 其余'%('背景窗口','RMSE','bias')]
    for n,_,_ in cfg:
        e=np.concatenate(E[n]); out.append('%-12s %.3f %+.3f | %.3f %+.3f | %.3f'%(n,rm(e),e.mean(),rm(e[z]),e[z].mean(),rm(e[~z])))
    print('\n'.join(out)); open('ext_window_results.txt','w').write('\n'.join(out))
else:
    T=[datetime(2024,1,27,17),datetime(2023,7,13,23),datetime(2023,10,7,6)]; tix={t:i for i,t in enumerate(st.times)}
    for q,tm in enumerate(T):
        t=tix[tm]; tr=st.frame(t); known=np.isfinite(tr)&oc; miss=oc&~known
        V={}
        for n,Lm,sp in (('erode2 + 48h',48,1),('erode2 + 7d',168,6),('erode2 + 14d',336,6),('erode2 + 30d',720,12)):
            _,_,_,F=run(t,Lm,sp); V[n]=np.where(known,tr,np.nan_to_num(F))
        allv=np.concatenate([v[miss] for v in V.values()]); lo,hi=np.percentile(allv,[1,99])
        P=[(np.where(known,tr,np.nan),'real input')]+list(V.items())
        fig,ax=plt.subplots(1,5,figsize=(26,6))
        for a_,(ti,d) in zip(ax,[(p[1],p[0]) if i==0 else (p[0],p[1]) for i,p in enumerate(P)]):
            im=a_.imshow(np.where(oc,d,np.nan),cmap='RdYlBu_r',vmin=lo,vmax=hi); a_.set_title(ti); a_.axis('off'); plt.colorbar(im,ax=a_,shrink=.7)
        fig.suptitle(str(tm)); plt.tight_layout(); plt.savefig(f'ext_vis{q}.png',dpi=60); plt.close()
