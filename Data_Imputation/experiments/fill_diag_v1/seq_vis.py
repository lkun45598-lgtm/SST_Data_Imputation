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
from scipy import sparse
from scipy.sparse.linalg import spsolve
LS=(20,40,60,100)
def screened(B,field,known,ocean,L):
    unk=ocean&~known; n=int(unk.sum()); H,W=field.shape
    idx=-np.ones((H,W),np.int64); idx[unk]=np.arange(n)
    r=np.where(known,field-B,0.0); deg=np.zeros(n); rows=[];cols=[]
    pad=lambda a,v:np.pad(a,1,constant_values=v)
    ocp,unp,idp=pad(ocean,False),pad(unk,False),pad(idx,-1)
    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
        sl=(slice(1+dy,1+dy+H),slice(1+dx,1+dx+W))
        deg+=ocp[sl][unk]; m=unp[sl][unk]; rows.append(idx[unk][m]); cols.append(idp[sl][unk][m])
    rows,cols=np.concatenate(rows),np.concatenate(cols)
    A=sparse.coo_matrix((-np.ones(len(rows)),(rows,cols)),shape=(n,n))+sparse.diags(deg+1.0/L**2)
    # 已知邻居的 delta 作为 Dirichlet 项
    b=np.zeros(n); rp=pad(r,0.0); knp=pad(known,False)
    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
        sl=(slice(1+dy,1+dy+H),slice(1+dx,1+dx+W)); m=knp[sl][unk]; b[m]+=rp[sl][unk][m]
    d=np.zeros((H,W)); d[unk]=spsolve(A.tocsc(),b)
    out=B+d; out[known]=field[known]; return out
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
    store.append((tr,hole,C,Bf,known,field,{L:screened(Bf,field,known,oc,L) for L in LS}))
    z=(C[hole]==0).mean()
    demos.append((z,tr,field,hole,C,Bf,Bc,H))
R={k:np.concatenate(v) for k,v in R.items()}



import matplotlib.pyplot as plt
z=np.array([(d[2][d[1]]==0).mean() for d in store]); order=np.argsort(-z)
pick=[order[0],order[len(order)//4],order[len(order)//2]]
for q,j in enumerate(pick):
    tr,hole,C,Bf,known,field,SC=store[j]; Bc=_boundary_correct(Bf,field,known,oc); S=SC[60]
    lo,hi=np.nanpercentile(tr[oc&np.isfinite(tr)],[2,98]); fill=lambda F:np.where(hole,F,tr)
    P=[(tr,'truth','RdYlBu_r',lo,hi),(field,'input (hole blank)','RdYlBu_r',lo,hi),(np.where(hole,C,np.nan),'C in hole','viridis',0,48),(fill(np.nan_to_num(Bf)),'tw fill','RdYlBu_r',lo,hi),
       (fill(np.nan_to_num(Bc)),'tw+bc fill','RdYlBu_r',lo,hi),(fill(np.nan_to_num(S)),'screened L=60 fill','RdYlBu_r',lo,hi),
       (np.where(hole,Bf-tr,np.nan),'tw error','RdBu_r',-2,2),(np.where(hole,Bc-tr,np.nan),'tw+bc error','RdBu_r',-2,2),(np.where(hole,S-tr,np.nan),'screened error','RdBu_r',-2,2)]
    fig,ax=plt.subplots(3,3,figsize=(15,15))
    for a,(d_,ti,cm,l,h) in zip(ax.ravel(),P):
        im=a.imshow(np.where(oc,d_,np.nan),cmap=cm,vmin=l,vmax=h); a.set_title(ti); a.axis('off'); plt.colorbar(im,ax=a,shrink=.7)
    fig.suptitle('sample %d: C=0 share in hole %.0f%%, hole %d px'%(q,100*z[j],hole.sum())); plt.tight_layout(); plt.savefig(f'fill_vis{q}.png',dpi=70); plt.close()
