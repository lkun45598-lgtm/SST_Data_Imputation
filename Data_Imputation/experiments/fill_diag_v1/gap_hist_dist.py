import sys, numpy as np, h5py
from scipy import ndimage as ndi
P='/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/hourly_store_v1/sst_hourly.h5'
N=int(sys.argv[1]); L=48; MIN=100
rng=np.random.default_rng(0)
f=h5py.File(P,'r'); pres=f['present'][:]; land=f['land_mask'][:]>0; ocean=~land
cand=[t for t in range(L,len(pres)) if pres[t-L:t+1].all()]
ts=rng.choice(cand,N,replace=False); ts.sort()
rows=[]  # (area, d_center_to_obs_now, d_center_to_hist, center_has_hist)
for t in ts:
    blk=f['sst'][t-L:t+1]; v=(blk!=-32768)&ocean
    cur=v[-1]; hist=v[:-1].any(0)
    miss=ocean&~cur
    if cur[ocean].mean()<0.02: continue
    lab,n=ndi.label(miss)
    dnow=ndi.distance_transform_edt(~cur)          # 到当前帧最近观测
    dhist=ndi.distance_transform_edt(~hist)        # 到最近"过去48h有观测"的像素
    areas=ndi.sum(miss,lab,range(1,n+1))
    for i in range(1,n+1):
        if areas[i-1]<MIN: continue
        m=lab==i
        c=np.unravel_index(np.argmax(np.where(m,dnow,-1)),m.shape)
        rows.append((areas[i-1],dnow[c],dhist[c],hist[c]))
r=np.array(rows,float); A=r[:,0]
def q(x,w=None): return np.percentile(x,[50,90,99,100]).round(1)
print('frames',len(ts),'blobs',len(r))
print('块中心到当前观测 px  [中位,P90,P99,max]',q(r[:,1]))
print('中心到最近历史观测 px[中位,P90,P99,max]',q(r[:,2]))
print('中心自身过去48h无观测的块占比 %.1f%%  (面积加权 %.1f%%)'%(100*(1-r[:,3]).mean(),100*((1-r[:,3])*A).sum()/A.sum()))
for th in (5,10,20,50): print('中心距历史观测 >%d px 的块占比 %.1f%%'%(th,100*(r[:,2]>th).mean()))
big=A>=2000
print('大块(>=2000px) n=%d 中心无历史占比 %.1f%% 中位距离 %.1f'%(big.sum(),100*(1-r[big,3]).mean(),np.median(r[big,2])))
