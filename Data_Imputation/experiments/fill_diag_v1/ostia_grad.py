import sys, numpy as np, h5py
sys.path.insert(0,'/data1/user/lz/SST_Data_Imputation/Data_Imputation')
from datasets.hourly_store import HourlyStore
def stats(a, ok):
    # ok: 像元自身及4邻域都有效
    gy,gx=np.gradient(a); g=np.hypot(gx,gy)
    lap=4*a[1:-1,1:-1]-a[:-2,1:-1]-a[2:,1:-1]-a[1:-1,:-2]-a[1:-1,2:]
    o=ok[1:-1,1:-1]; gg=g[1:-1,1:-1]
    # 梯度的变化率: |∇|∇T||
    g2y,g2x=np.gradient(g); gj=np.hypot(g2x,g2y)[1:-1,1:-1]
    return gg[o].mean(), np.sqrt((lap[o]**2).mean()), gj[o].mean(), (np.abs(lap[o])>0.3).mean()
f=h5py.File('/data/sst_data/sst_missing_value_imputation/processed_data/processed_sst_valid.h5','r')
land=f['land_mask'][:]>0; idx=np.random.default_rng(0).choice(f['ground_truth_sst'].shape[0],60,replace=False); idx.sort()
def inner(m): 
    o=m.copy(); o[1:-1,1:-1]=m[1:-1,1:-1]&m[:-2,1:-1]&m[2:,1:-1]&m[1:-1,:-2]&m[1:-1,2:]; return o
R=[]
for i in idx:
    a=f['ground_truth_sst'][i].astype(float); ok=inner(~land&np.isfinite(a)); R.append(stats(np.nan_to_num(a),ok))
ost=np.mean(R,0)
st=HourlyStore(); sl=[s for s in range(0,st.T,37) if st.present[s] and st.coverage[s]>=0.85][:60]
R=[]
for s in sl:
    a=st.frame(s); ok=inner(np.isfinite(a)&st.ocean); R.append(stats(np.nan_to_num(a),ok))
jx=np.mean(R,0)
print('%-22s %10s %12s %14s %12s'%('', '|∇T| K/px','Laplacian RMS','|∇|∇T|| K/px','|lap|>0.3 占比'))
print('%-22s %10.4f %12.4f %14.4f %12.3f'%(('OSTIA ground truth',*ost)))
print('%-22s %10.4f %12.4f %14.4f %12.3f'%(('JAXA 晴空帧(覆盖>=0.85)',*jx)))
