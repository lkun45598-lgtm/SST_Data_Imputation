import sys, numpy as np
sys.path.insert(0,'/data1/user/lz/SST_Data_Imputation/Data_Imputation')
exec(open('feather.py').read().split("\nif sys.argv[1]=='eval':")[0])
from preprocessing.fill_temporal import _smooth
from preprocessing.seam_metrics import seam_stats
rng=np.random.default_rng(11); N=int(sys.argv[1]); L=48
c=[s for s in st.slot_range(VAL_START,VAL_END) if st.present[s] and 0.12<=st.coverage[s]<=0.4 and s>=L and st.present[s-L:s+2].all()]
mon=np.array([st.times[s].month for s in c]); pick=[]
for m in range(1,13):
    cm=[s for s,mm in zip(c,mon) if mm==m]
    if cm: pick+=list(rng.choice(cm,min(len(cm),max(1,N//12)),replace=False))
Ss=[0,2,4,8,16]; E={s:[] for s in Ss}; GR={s:[] for s in Ss}; OJ={s:[] for s in Ss}; TX={s:[] for s in Ss}; Zs=[]
def hp_std(a,sig=2.0):
    num=gaussian_filter(np.where(oc,a,0.),sig); den=gaussian_filter(oc.astype(float),sig)+1e-9
    hp=np.where(oc,a-num/den,0.); return np.sqrt(np.maximum(gaussian_filter(hp**2,sig)/den,0))
for t in pick:
    t=int(t); tr=st.frame(t); known=np.isfinite(tr)&oc; ref=st.frame(t+1); mm=oc&~known&np.isfinite(ref); field=np.where(known,tr,np.nan)
    if mm.sum()<2000: continue
    Bh,z=fill_feather(t,field,known,None); Zs.append(z[mm]); res={}
    for s in Ss:
        Bs=Bh if s==0 else _smooth(Bh,oc,s)
        res[s]=np.where(known,np.nan_to_num(tr),np.nan_to_num(robust_bc(Bs,field,known,oc,2,0))); E[s].append((res[s]-ref)[mm])
    cl=oc&~known&~z
    for s in Ss:
        F=res[s]; o=seam_stats(F,known,oc&~known); OJ[s].append(o['value_jump'])
        sd=hp_std(F); dk=edt(known); dm=edt(~known); so=sd[known&(dk>=4)&(dk<10)]; sf=sd[(~known)&oc&(dm>=4)&(dm<10)]
        if len(so)>300 and len(sf)>300: TX[s].append(sf.mean()/so.mean())
    if z.sum()>3000 and cl.sum()>3000:
        for s in Ss:
            F=res[s]; S=gaussian_filter(np.where(oc,F,0),2)/(gaussian_filter(oc.astype(float),2)+1e-9); gy,gx=np.gradient(S); g=np.hypot(gx,gy)
            rest=oc&~known&~binary_dilation(z,iterations=3); GR[s].append(g[band(z)].mean()/g[rest].mean())
z=np.concatenate(Zs); rm=lambda e:np.sqrt(np.mean(e**2))
out=['frames %d; 梯度比帧数 %d'%(len(Zs),len(GR[0])),'%-9s %6s | C=0区 RMSE | C=0边界带梯度比 | 观测→云边界值台阶 | 填充侧/观测侧纹理(4-10px)'%('平滑σ(px)','RMSE')]
for s in Ss:
    e=np.concatenate(E[s]); out.append('%-9s %.3f | %.3f | %.2f | %.3f | %.2f'%(s,rm(e),rm(e[z]),np.mean(GR[s]),np.nanmean(OJ[s]),np.mean(TX[s])))
print('\n'.join(out)); open('smooth_results.txt','w').write('\n'.join(out))
