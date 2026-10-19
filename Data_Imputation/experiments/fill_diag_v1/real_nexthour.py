"""真实帧验证: 参照 = 当前帧缺失、下一小时(或再下一小时)有观测的像元。非因果参照，仅用于评估。"""
import sys, numpy as np
sys.path.insert(0,'/data1/user/lz/SST_Data_Imputation/Data_Imputation')
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import binary_erosion, label as cclabel, sum as ndsum
from datasets.hourly_store import HourlyStore
from preprocessing.fill_compare import VAL_START, VAL_END
from preprocessing.fill_temporal import _offsets,_weighted_background,_complete,_boundary_correct
from preprocessing.fill_methods import _harmonic_solve
a=open('seq_screened.py').read(); exec(a[a.index('def screened'):a.index('R={k:[]')])
b=open('seq_robust.py').read(); exec(b[b.index('def source'):b.index('cfg=')])
N=int(sys.argv[1]); rng=np.random.default_rng(5); L=48
st=HourlyStore(); oc=st.ocean
c=[s for s in st.slot_range(VAL_START,VAL_END) if st.present[s] and 0.12<=st.coverage[s]<=0.4 and s>=L and st.present[s-L:s+2].all()]
mon=np.array([st.times[s].month for s in c]); pick=[]
for m in range(1,13):
    cm=[s for s,mm in zip(c,mon) if mm==m]
    if cm: pick+=list(rng.choice(cm,min(len(cm),max(1,N//12)),replace=False))
cfg=[('tw',None),('tw+bc',(0,0,None)),('L60',(0,0,60)),('erode2',(2,0,None)),('erode2+min500',(2,500,None)),('erode2+min500+L60',(2,500,60))]
E={n:[] for n,_ in cfg}; Cs=[]; fr=[]
for t in pick:
    t=int(t); tr=st.frame(t); known=np.isfinite(tr)&oc; field=np.where(known,tr,np.nan)
    ref=np.concatenate([np.stack([st.frame(t+1)])],0)[0]; m=oc&~known&np.isfinite(ref)
    if m.sum()<2000: continue
    past={l:st.frame(t-l) for l in range(1,L+1)}
    C=(np.stack([np.isfinite(past[l]) for l in past])&oc).sum(0)
    offs=_offsets(field,known,past); Bf=_complete(_weighted_background(past,offs,lambda l:1.0/l),oc)
    for n,p in cfg:
        F=Bf if p is None else robust_bc(Bf,field,known,oc,*p)
        E[n].append((np.nan_to_num(F)-ref)[m])
    Cs.append(C[m]); fr.append(m.sum())
Cc=np.concatenate(Cs); rm=lambda e:np.sqrt(np.mean(e**2))
out=['frames %d, ref px %d'%(len(fr),len(Cc)),'%-18s %6s %7s | C=0: rmse bias | C1-5 | C>=6'%('方法','RMSE','bias')]
for n,_ in cfg:
    e=np.concatenate(E[n]); z=Cc==0; mid=(Cc>=1)&(Cc<6); hi=Cc>=6
    out.append('%-18s %.3f %+.3f | %.3f %+.3f | %.3f | %.3f'%(n,rm(e),e.mean(),rm(e[z]),e[z].mean(),rm(e[mid]),rm(e[hi])))
print('\n'.join(out)); open('real_nexthour_results.txt','w').write('\n'.join(out))
