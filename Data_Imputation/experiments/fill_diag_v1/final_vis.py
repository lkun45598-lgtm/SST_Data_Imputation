import sys, numpy as np, h5py
sys.path.insert(0,'/data1/user/lz/SST_Data_Imputation/Data_Imputation')
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import binary_erosion, label as cclabel, sum as ndsum, distance_transform_edt as edt
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from datetime import datetime
from datasets.hourly_store import HourlyStore
from preprocessing.fill_compare import VAL_START, VAL_END
from preprocessing.fill_temporal import _offsets,_weighted_background,_complete,_boundary_correct
from preprocessing.fill_methods import _harmonic_solve
a=open('seq_screened.py').read(); exec(a[a.index('def screened'):a.index('R={k:[]')])
b=open('seq_robust.py').read(); exec(b[b.index('def source'):b.index('cfg=')])
rng=np.random.default_rng(21); L=48
st=HourlyStore(); oc=st.ocean
c=[s for s in st.slot_range(VAL_START,VAL_END) if st.present[s] and 0.12<=st.coverage[s]<=0.3 and s>=L and st.present[s-L:s+1].all()]
pick=[]
for months in ((1,2,12),(3,4,5),(6,7,8),(9,10,11)):
    cm=[s for s in c if st.times[s].month in months]
    if cm: pick.append(int(rng.choice(cm)))
base='/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/hourly_data'
for q,t in enumerate(pick):
    tm=st.times[t]; tr=st.frame(t); known=np.isfinite(tr)&oc; field=np.where(known,tr,np.nan)
    past={l:st.frame(t-l) for l in range(1,L+1)}
    C=(np.stack([np.isfinite(past[l]) for l in past])&oc).sum(0); miss=oc&~known
    offs=_offsets(field,known,past); Bf=_complete(_weighted_background(past,offs,lambda l:1.0/l),oc)
    F={'tw (no bc)':Bf,'tw+bc (erode0)':robust_bc(Bf,field,known,oc,0,0),'bc erode1':robust_bc(Bf,field,known,oc,1,0),'bc erode2':robust_bc(Bf,field,known,oc,2,0)}
    orig=None
    try:
        f=h5py.File(f'{base}/h{tm.hour:02d}/jaxa_knn_filled_07.h5','r'); ts=[datetime.strptime(x.decode()[:10],'%Y-%m-%d') for x in f['timestamps'][:]]
        i=ts.index(datetime(tm.year,tm.month,tm.day)); orig=f['sst_data'][i].astype(float)
    except Exception as e: print('orig not found',tm,e)
    if orig is not None: F={'orig (time+gauss+3D KNN)':orig,**F}
    fills={k:np.where(known,tr,np.nan_to_num(v)) for k,v in F.items()}
    allv=np.concatenate([v[miss] for v in fills.values()]); lo,hi=np.percentile(allv,[1,99])
    panels=[(field,'real input (blank=cloud)','RdYlBu_r',lo,hi),(np.where(miss,C,np.nan),'C: obs count past 48h','viridis',0,48)]+[(v,k,'RdYlBu_r',lo,hi) for k,v in fills.items()]
    n=len(panels); cols=4; rows=(n+cols-1)//cols
    fig,ax=plt.subplots(rows,cols,figsize=(5.2*cols,5.6*rows)); ax=ax.ravel()
    for a_,(d,ti,cm,l,h) in zip(ax,panels):
        im=a_.imshow(np.where(oc,d,np.nan),cmap=cm,vmin=l,vmax=h); a_.set_title(ti); a_.axis('off'); plt.colorbar(im,ax=a_,shrink=.7)
    for a_ in ax[n:]: a_.axis('off')
    fig.suptitle('%s  coverage %.1f%%  (same colorbar for all fills)'%(tm,100*st.coverage[t])); plt.tight_layout(); plt.savefig(f'final_vis{q}.png',dpi=65); plt.close()
    print(q,tm,round(float(st.coverage[t]),3))
