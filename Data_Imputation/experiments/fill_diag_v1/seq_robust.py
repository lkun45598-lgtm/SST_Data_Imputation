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
    store.append((tr,hole,C,Bf,known,field,None))
    z=(C[hole]==0).mean()
    demos.append((z,tr,field,hole,C,Bf,Bc,H))
R={k:np.concatenate(v) for k,v in R.items()}



from scipy.ndimage import binary_erosion, label as cclabel, sum as ndsum, distance_transform_edt as edt
from preprocessing.fill_methods import _harmonic_solve
def source(known,erode,minsize):
    src=known.copy()
    if erode>0: src=binary_erosion(src,iterations=erode)&known
    if minsize>0:
        lab,n=cclabel(src)
        if n: src=np.isin(lab,1+np.flatnonzero(ndsum(src,lab,range(1,n+1))>=minsize))
    return src
def robust_bc(Bf,field,known,oc,erode,minsize,L=None):
    src=source(known,erode,minsize)
    if src.sum()<100: src=known
    if L is None:
        r=np.where(src,field-Bf,0.0); out=Bf+_harmonic_solve(r,src,oc)*(~src)
    else:
        out=screened(Bf,field,src,oc,L)
    return np.where(known,field,out)
cfg=[('bc',0,0,None),('erode1',1,0,None),('erode2',2,0,None),('erode3',3,0,None),('min100',0,100,None),('min500',0,500,None),('erode2+min100',2,100,None),('erode2+min500',2,500,None),('erode2+min100+L60',2,100,60),('L60 only',0,0,60)]
rm=lambda e:np.sqrt(np.mean(e**2)); res={c[0]:[] for c in cfg}; Cs=[]; Ds=[]
for tr,hole,C,Bf,known,field,_ in store:
    tt=np.nan_to_num(tr)
    for n,er,ms,L in cfg: res[n].append((np.nan_to_num(robust_bc(Bf,field,known,oc,er,ms,L))-tt)[hole])
    Cs.append(C[hole]); Ds.append(edt(~known)[hole])
Cc=np.concatenate(Cs); Dc=np.concatenate(Ds)
out=['%-18s %6s | C=0 | C1-2 | C3-5 | C>=6 | C=0&>50px | C>0&>50px'%('方法','总')]
for n,*_ in cfg:
    e=np.concatenate(res[n]); z=Cc==0
    out.append('%-18s %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f'%(n,rm(e),rm(e[z]),rm(e[(Cc>=1)&(Cc<3)]),rm(e[(Cc>=3)&(Cc<6)]),rm(e[Cc>=6]),rm(e[z&(Dc>50)]),rm(e[(~z)&(Dc>50)])))
print('\n'.join(out)); open('robust_results.txt','w').write('\n'.join(out))

# 真实帧重画: 宽色标
import matplotlib.pyplot as plt
from datasets.hourly_store import HourlyStore
from preprocessing.fill_compare import NEED_PAST
t=int(np.flatnonzero([str(x).startswith('2023-10-07 06') for x in st.times])[0])
tr=st.frame(t); known=np.isfinite(tr)&oc; field=np.where(known,tr,np.nan)
past={l:st.frame(t-l) for l in range(1,49)}
offs=_offsets(field,known,past); Bf=_complete(_weighted_background(past,offs,lambda l:1.0/l),oc)
V={'tw':Bf,'tw+bc':_boundary_correct(Bf,field,known,oc),'bc erode2+min500':robust_bc(Bf,field,known,oc,2,500),'bc erode2+min500+L60':robust_bc(Bf,field,known,oc,2,500,60)}
fills={k:np.where(known,tr,np.nan_to_num(v)) for k,v in V.items()}
allv=np.concatenate([f[oc&~known] for f in fills.values()]); lo,hi=np.percentile(allv,[0.5,99.5])
print('real frame fill value range 0.5-99.5 pct: %.2f %.2f ; input obs 2-98: %s'%(lo,hi,np.nanpercentile(tr[known],[2,98])))
fig,ax=plt.subplots(2,3,figsize=(18,12)); miss=oc&~known
P=[(fills['tw'],'tw','RdYlBu_r',lo,hi),(fills['tw+bc'],'tw+bc','RdYlBu_r',lo,hi),(fills['bc erode2+min500'],'bc, erode2+min500','RdYlBu_r',lo,hi),
   (fills['bc erode2+min500+L60'],'bc, erode2+min500+L60','RdYlBu_r',lo,hi),
   (np.where(miss,fills['tw+bc']-fills['tw'],np.nan),'tw+bc - tw (K, ±3)','RdBu_r',-3,3),(np.where(miss,fills['bc erode2+min500']-fills['tw'],np.nan),'bc erode2+min500 - tw (K, ±3)','RdBu_r',-3,3)]
for a,(d,ti,cm,l,h) in zip(ax.ravel(),P):
    im=a.imshow(np.where(oc,d,np.nan),cmap=cm,vmin=l,vmax=h); a.set_title(ti); a.axis('off'); plt.colorbar(im,ax=a,shrink=.7)
fig.suptitle('2023-10-07 06:00 real frame, wide colorbar'); plt.tight_layout(); plt.savefig('real_vis2_robust.png',dpi=70)
