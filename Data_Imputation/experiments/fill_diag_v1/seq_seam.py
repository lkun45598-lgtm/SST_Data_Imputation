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

from preprocessing.seam_metrics import seam_stats, seam_reference, gradient_ratio, highpass_stats
meths={'truth':None,'tw':lambda Bf,fi,kn:Bf,'tw+bc':lambda Bf,fi,kn:robust_bc(Bf,fi,kn,oc,0,0),'erode2':lambda Bf,fi,kn:robust_bc(Bf,fi,kn,oc,2,0),
       'L60':lambda Bf,fi,kn:robust_bc(Bf,fi,kn,oc,0,0,60),'harmonic':lambda Bf,fi,kn:np.nan_to_num(harmonic_fill(fi,kn,oc))}
acc={n:{'sj':[],'vj':[],'gr':[],'hc':[],'hr':[]} for n in meths}
for tr,hole,C,Bf,known,field,_ in store:
    tt=np.nan_to_num(tr)
    for n,fn in meths.items():
        F=tt if fn is None else np.where(known,tt,np.nan_to_num(fn(Bf,field,known)))
        st_=seam_stats(F,known,hole); acc[n]['sj'].append(st_['slope_jump']); acc[n]['vj'].append(st_['value_jump'])
        acc[n]['gr'].append(gradient_ratio(F,tt,hole)); h=highpass_stats(F,tr,hole,oc)
        acc[n]['hc'].append(h['hp_corr']); acc[n]['hr'].append(h['hp_ratio'])
out=['%-9s %10s %10s %9s %8s %8s'%('方法','slope_jump','value_jump','grad比','hp相关','hp幅度比')]
for n in meths:
    m=lambda k:np.nanmean(acc[n][k])
    out.append('%-9s %10.4f %10.4f %9.2f %8.2f %8.2f'%(n,m('sj'),m('vj'),m('gr'),m('hc'),m('hr')))
print('\n'.join(out)); open('seam_results.txt','w').write('\n'.join(out)+'\n(单位 K/px; truth 行是真值自身的同一指标，作为参考)')
