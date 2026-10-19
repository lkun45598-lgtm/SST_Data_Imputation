import sys, numpy as np, h5py
P='/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/hourly_store_v1/sst_hourly.h5'
N=int(sys.argv[1]); L=48
rng=np.random.default_rng(1)
f=h5py.File(P,'r'); pres=f['present'][:]; ocean=f['land_mask'][:]==0
cand=np.array([t for t in range(L,len(pres)) if pres[t-L:t+1].all()])
ts=np.sort(rng.choice(cand,N,replace=False))
bins=[0,1,3,6,12,24,49]; hist=np.zeros(len(bins)-1); gstat=[]; jump=[]
for t in ts:
    v=(f['sst'][t-L:t]!=-32768)&ocean
    C=v.sum(0).astype(float)
    cur=(f['sst'][t]!=-32768)&ocean
    miss=ocean&~cur
    hist+=np.histogram(C[miss],bins=bins)[0]
    gy,gx=np.gradient(C); g=np.hypot(gx,gy)
    ok=ocean&np.roll(ocean,1,0)&np.roll(ocean,-1,0)&np.roll(ocean,1,1)&np.roll(ocean,-1,1)
    gstat.append(np.percentile(g[ok&miss],[50,90,99]))
    jump.append((g[ok&miss]>8).mean())
hist/=hist.sum()
print('缺失像元的 C 分布 (过去48h观测次数):')
for a,b,h in zip(bins[:-1],bins[1:],hist): print('  C in [%d,%d): %.1f%%'%(a,b,100*h))
gs=np.mean(gstat,0); print('缺失区 |∇C| (次/px) 中位/P90/P99: %.2f %.2f %.2f'%tuple(gs))
print('|∇C|>8 次/px 的缺失像元占比: %.1f%%'%(100*np.mean(jump)))
