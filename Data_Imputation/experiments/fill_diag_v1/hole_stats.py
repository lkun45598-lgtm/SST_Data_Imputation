import sys,numpy as np; sys.path.insert(0,'/data1/user/lz/SST_Data_Imputation/Data_Imputation')
from datetime import datetime
from datasets.artificial_holes import *
from scipy.ndimage import distance_transform_edt as edt, uniform_filter
st=HourlyStore(); tr_slots=st.slot_range(datetime(2015,7,7),datetime(2023,7,5)); bank=build_bank_from_store(st,tr_slots,n=300)
cand=[s for s in tr_slots if st.present[s] and st.coverage[s]>=0.7 and s>=100 and st.present[s-80:s+1].all()]
src=[s for s in tr_slots if s>=100 and st.present[s-80:s+1].all() and 0.2<=st.coverage[s]<=0.6]
rng=np.random.default_rng(int(sys.argv[2])); P=int(sys.argv[3]); rows=[]
for ratio in (0.25,0.5,0.6):
    for _ in range(int(sys.argv[1])):
        s=make_sample(st,int(rng.choice(cand)),bank,rng,ratio,src,full_window=False)
        if s is None: continue
        h,C,tr=s['hole'],s['C'],s['truth']
        real_cloud=st.ocean&~np.isfinite(tr)                        # 真实云/无真值
        far=edt(~real_cloud)>=3                                      # 离真实云边>=3px 的像元
        good=h&far&np.isfinite(tr)                                   # 有真值且离云边的洞内像元
        full=uniform_filter(good.astype(float),size=P,mode='constant')>=0.999   # 以该点为中心的 PxP 块全部是 good
        rows.append((ratio,h.sum()/st.ocean.sum(),(C[h]==0).mean(),(C[h]<=2).mean(),full.sum(),good.sum()/max(h.sum(),1)))
r=np.array(rows)
print('块大小 %dpx'%P,'| 样本',len(r))
for ratio in (0.25,0.5,0.6):
    m=r[:,0]==ratio
    if m.any(): print('  有效块数>=100的样本占比 %.0f%% | >=500 %.0f%% | 为0 %.0f%%'%(100*(r[m,4]>=100).mean(),100*(r[m,4]>=500).mean(),100*(r[m,4]==0).mean()))
    if m.any(): print('目标挖洞比例 %.2f: 洞占海洋 %.2f | C=0占洞 %.1f%% C<=2占洞 %.1f%% | 完整42px有效块中心数 中位 %d (P10 %d, 最小 %d) | 洞内"有真值且离云边>=3px"占比 %.2f'%(ratio,r[m,1].mean(),100*r[m,2].mean(),100*r[m,3].mean(),np.median(r[m,4]),np.percentile(r[m,4],10),r[m,4].min(),r[m,5].mean()))
