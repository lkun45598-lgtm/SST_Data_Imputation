import sys, numpy as np
sys.path.insert(0,'/data1/user/lz/SST_Data_Imputation/Data_Imputation')
exec(open('feather.py').read().split("\nif sys.argv[1]=='eval':")[0])
def bc_clip(Bf,field,known,erode,clip,mode):
    src=source(known,erode,0)
    if src.sum()<100: src=known
    r=np.where(src,field-Bf,0.0)
    if clip is not None: r=np.clip(r,-clip,clip) if mode=='hard' else clip*np.tanh(r/clip)
    out=Bf+_harmonic_solve(r,src,oc)*(~src); return np.where(known,field,out)
rng=np.random.default_rng(11); N=int(sys.argv[1]); L=48
c=[s for s in st.slot_range(VAL_START,VAL_END) if st.present[s] and 0.12<=st.coverage[s]<=0.4 and s>=L and st.present[s-L:s+2].all()]
mon=np.array([st.times[s].month for s in c]); pick=[]
for m in range(1,13):
    cm=[s for s,mm in zip(c,mon) if mm==m]
    if cm: pick+=list(rng.choice(cm,min(len(cm),max(1,N//12)),replace=False))
cfg=[('tw (无bc)',None,None),('bc erode2',None,None)]+[(f'硬限幅 ±{k}K',k,'hard') for k in (0.5,1.0,2.0)]+[(f'tanh软限幅 {k}K',k,'tanh') for k in (0.5,1.0,2.0)]
E={n:[] for n,_,_ in cfg}; BUMP={n:[] for n,_,_ in cfg}; FR={n:[] for n,_,_ in cfg}
for t in pick:
    t=int(t); tr=st.frame(t); known=np.isfinite(tr)&oc; ref=st.frame(t+1); mm=oc&~known&np.isfinite(ref); field=np.where(known,tr,np.nan)
    if mm.sum()<2000: continue
    Bf,z=fill_feather(t,field,known,None); miss=oc&~known
    for n,k,mode in cfg:
        F=Bf if n.startswith('tw') else bc_clip(Bf,field,known,2,k,mode)
        E[n].append((np.nan_to_num(F)-ref)[mm]); d=np.abs(np.nan_to_num(F)-Bf)[miss]; BUMP[n].append(np.percentile(d,99)); FR[n].append((d>1.0).mean())
rm=lambda e:np.sqrt(np.mean(e**2)); out=['frames %d'%len(E['tw (无bc)']),'%-16s %6s %7s | bc改动 |F-tw| P99(K) | 改动>1K的云区像元占比'%('方法','RMSE','bias')]
for n,_,_ in cfg:
    e=np.concatenate(E[n]); out.append('%-16s %.3f %+.3f | %.2f | %.1f%%'%(n,rm(e),e.mean(),np.mean(BUMP[n]),100*np.mean(FR[n])))
print('\n'.join(out)); open('clip_results.txt','w').write('\n'.join(out))
