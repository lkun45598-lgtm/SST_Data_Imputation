"""训练期(2015-07-01 ~ 2023-07-05)逐像元谐波气候态: 常数 + 年周期2阶 + 日周期2阶 + 年1阶x日1阶交叉。
每 STRIDE 个小时取一帧。只用训练期，验证/测试无泄漏。输出 clim_coef.npz (P,Hocean) """
import sys, time, numpy as np
from datetime import datetime
sys.path.insert(0,'/data1/user/lz/SST_Data_Imputation/Data_Imputation')
from datasets.hourly_store import HourlyStore
STRIDE=4
def basis(dt):
    doy=dt.timetuple().tm_yday+dt.hour/24; a=2*np.pi*doy/365.25; d=2*np.pi*dt.hour/24
    ca,sa,ca2,sa2,cd,sd,cd2,sd2=np.cos(a),np.sin(a),np.cos(2*a),np.sin(2*a),np.cos(d),np.sin(d),np.cos(2*d),np.sin(2*d)
    return np.array([1,ca,sa,ca2,sa2,cd,sd,cd2,sd2,ca*cd,ca*sd,sa*cd,sa*sd],np.float64)
if __name__=='__main__':
    st=HourlyStore(); oc=st.ocean; idx=np.flatnonzero(oc.ravel()); n=len(idx)
    slots=[s for s in st.slot_range(datetime(2015,7,1),datetime(2023,7,5)) if st.present[s]][::STRIDE]
    p=13; pairs=[(i,j) for i in range(p) for j in range(i,p)]
    A=np.zeros((len(pairs),n),np.float32); b=np.zeros((p,n),np.float32); cnt=np.zeros(n,np.float32)
    t0=time.time()
    for q,s in enumerate(slots):
        x=basis(st.times[s]); f=st.frame(s).ravel()[idx]; m=np.isfinite(f)
        if m.sum()==0: continue
        mf=m.astype(np.float32); y=np.where(m,f,0).astype(np.float32); cnt+=mf
        for k,(i,j) in enumerate(pairs): A[k]+=np.float32(x[i]*x[j])*mf
        for i in range(p): b[i]+=np.float32(x[i])*y
        if q%500==0: print(q,len(slots),'%.0fs'%(time.time()-t0),flush=True)
    M=np.zeros((n,p,p),np.float64)
    for k,(i,j) in enumerate(pairs): M[:,i,j]=M[:,j,i]=A[k]
    M+=1e-3*np.eye(p)[None]*np.maximum(cnt,1)[:,None,None]
    coef=np.linalg.solve(M,b.T.astype(np.float64)[:,:,None])[:,:,0]
    coef[cnt<500]=np.nan
    np.savez('clim_coef.npz',coef=coef.astype(np.float32),idx=idx,cnt=cnt); print('done',time.time()-t0)
