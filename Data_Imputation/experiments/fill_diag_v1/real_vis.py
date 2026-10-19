import sys, numpy as np
sys.path.insert(0,'/data1/user/lz/SST_Data_Imputation/Data_Imputation')
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from datasets.hourly_store import HourlyStore
from preprocessing.fill_compare import VAL_START, VAL_END, NEED_PAST
from preprocessing.fill_temporal import _offsets,_weighted_background,_complete,_boundary_correct
src=open('seq_screened.py').read(); a=src.index('def screened'); b=src.index('R={k:[]'); exec(src[a:b])
rng=np.random.default_rng(3); L=48
st=HourlyStore(); oc=st.ocean
c=[s for s in st.slot_range(VAL_START,VAL_END) if st.present[s] and 0.10<=st.coverage[s]<=0.25 and s>=L and st.present[s-L:s].all()]
mon=np.array([st.times[s].month for s in c]); pick=[]
for m in (1,4,7,10):
    cm=[s for s,mm in zip(c,mon) if mm==m]
    if cm: pick.append(int(rng.choice(cm)))
print('picked',[(st.times[s].strftime('%Y-%m-%d %H'),round(float(st.coverage[s]),3)) for s in pick])
for q,t in enumerate(pick):
    tr=st.frame(t); known=np.isfinite(tr)&oc; field=np.where(known,tr,np.nan)
    past={l:st.frame(t-l) for l in range(1,L+1)}
    C=(np.stack([np.isfinite(past[l]) for l in past])&oc).sum(0)
    offs=_offsets(field,known,past); Bf=_complete(_weighted_background(past,offs,lambda l:1.0/l),oc)
    Bc=_boundary_correct(Bf,field,known,oc); S=screened(Bf,field,known,oc,60)
    lo,hi=np.nanpercentile(tr[known],[2,98]); miss=oc&~known
    fill=lambda F:np.where(known,tr,np.nan_to_num(F))
    P=[(field,'real input (blank = cloud)','RdYlBu_r',lo,hi),(np.where(miss,C,np.nan),'C in cloud area','viridis',0,48),(fill(Bf),'tw fill','RdYlBu_r',lo,hi),
       (fill(Bc),'tw+bc fill','RdYlBu_r',lo,hi),(fill(S),'screened L=60 fill','RdYlBu_r',lo,hi),(np.where(miss,Bc-S,np.nan),'tw+bc minus screened (K)','RdBu_r',-1,1)]
    fig,ax=plt.subplots(2,3,figsize=(18,12))
    for a_,(d,ti,cm,l,h) in zip(ax.ravel(),P):
        im=a_.imshow(np.where(oc,d,np.nan),cmap=cm,vmin=l,vmax=h); a_.set_title(ti); a_.axis('off'); plt.colorbar(im,ax=a_,shrink=.7)
    fig.suptitle('%s  coverage %.1f%%  C=0 in cloud area %.0f%%'%(st.times[t],100*st.coverage[t],100*(C[miss]==0).mean())); plt.tight_layout(); plt.savefig(f'real_vis{q}.png',dpi=70); plt.close()
