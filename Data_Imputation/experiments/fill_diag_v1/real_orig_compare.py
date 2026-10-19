"""原始流水线 (时间加权+高斯+3D KNN, hourly_data/hXX series07) vs tw / bc / erode2, 同一批像元、同一个下一小时参照。"""
import sys, os, numpy as np, h5py
sys.path.insert(0,'/data1/user/lz/SST_Data_Imputation/Data_Imputation')
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import binary_erosion, label as cclabel, sum as ndsum
from datetime import datetime, timedelta
from datasets.hourly_store import HourlyStore
from preprocessing.fill_temporal import _offsets,_weighted_background,_complete,_boundary_correct
from preprocessing.fill_methods import _harmonic_solve
a=open('seq_screened.py').read(); exec(a[a.index('def screened'):a.index('R={k:[]')])
b=open('seq_robust.py').read(); exec(b[b.index('def source'):b.index('cfg=')])
st=HourlyStore(); oc=st.ocean; L=48
t2i={t:i for i,t in enumerate(st.times)}
rng=np.random.default_rng(7); per_hour=int(sys.argv[1])
base='/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/hourly_data'
hours=[int(d[1:]) for d in sorted(os.listdir(base)) if d[0]=='h' and d[1:].isdigit() and os.path.exists(f'{base}/{d}/jaxa_knn_filled_07.h5')]
print('hours',hours,flush=True)
names=['orig(时间+高斯+3D KNN)','tw','tw+bc','erode2']; E={n:[] for n in names}; Cs=[]; nfr=0
for h in hours:
    f=h5py.File(f'{base}/h{h:02d}/jaxa_knn_filled_07.h5','r')
    ts=[datetime.strptime(x.decode()[:19].replace('T',' '),'%Y-%m-%d %H:%M:%S') for x in f['timestamps'][:]]
    obs=f['original_obs_mask']; cov=np.array([obs[i][oc].mean() for i in range(len(ts))])
    cand=[i for i in range(len(ts)) if 0.12<=cov[i]<=0.4]
    for i in rng.choice(cand,min(per_hour,len(cand)),replace=False):
        t=t2i.get(ts[i]);
        if t is None or t<L or not st.present[t-L:t+2].all(): continue
        tr=st.frame(t); known=np.isfinite(tr)&oc; ref=st.frame(t+1); m=oc&~known&np.isfinite(ref)
        if m.sum()<2000: continue
        orig=f['sst_data'][int(i)].astype(np.float64)
        field=np.where(known,tr,np.nan); past={l:st.frame(t-l) for l in range(1,L+1)}
        C=(np.stack([np.isfinite(past[l]) for l in past])&oc).sum(0)
        offs=_offsets(field,known,past); Bf=_complete(_weighted_background(past,offs,lambda l:1.0/l),oc)
        F={'orig(时间+高斯+3D KNN)':orig,'tw':Bf,'tw+bc':robust_bc(Bf,field,known,oc,0,0),'erode2':robust_bc(Bf,field,known,oc,2,0)}
        for n in names: E[n].append((np.nan_to_num(F[n])-ref)[m])
        Cs.append(C[m]); nfr+=1
Cc=np.concatenate(Cs); rm=lambda e:np.sqrt(np.mean(e**2))
print('orig 数值范围检查:',float(np.nanmin(orig)),float(np.nanmax(orig)))
out=['frames %d, ref px %d, hours %s'%(nfr,len(Cc),hours),'%-24s %6s %7s | C=0 rmse bias | C1-5 | C>=6'%('方法','RMSE','bias')]
for n in names:
    e=np.concatenate(E[n]); z=Cc==0; mid=(Cc>=1)&(Cc<6); hi=Cc>=6
    out.append('%-24s %.3f %+.3f | %.3f %+.3f | %.3f | %.3f'%(n,rm(e),e.mean(),rm(e[z]),e[z].mean(),rm(e[mid]),rm(e[hi])))
print('\n'.join(out)); open('real_orig_compare_results.txt','w').write('\n'.join(out))
