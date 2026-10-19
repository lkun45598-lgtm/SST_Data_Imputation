import sys, numpy as np
sys.path.insert(0,'/data1/user/lz/SST_Data_Imputation/Data_Imputation')
exec(open('ext_window.py').read().split("if sys.argv[1]=='eval'")[0])
from scipy.ndimage import binary_dilation, gaussian_filter
def fill_feather(t,field,known,W):
    past={l:fr(t-l) for l in range(1,49)}
    B=_weighted_background(past,_offsets(field,known,past),lambda l:1.0/l); z=np.isnan(B)&oc&~known
    Bh=_complete(B,oc)
    if W is None or z.sum()<50: return Bh,z
    ext={l:fr(t-l) for l in range(49,337,6)}
    Be=_weighted_background(ext,_offsets(field,known,ext),lambda l:1.0/l); have=np.isfinite(Be)&z
    Bef=_complete(np.where(np.isfinite(Be),Be,np.nan),oc)
    K=np.isfinite(B)&oc&np.isfinite(Be)       # 两个背景都有值的像元: 用来对齐
    if K.sum()<500: return Bh,z
    Bal=screened(Bef,np.where(K,B,np.nan),K,oc,W*2)   # 对齐后的扩展背景
    d=edt(z); w=np.clip(1-d/W,0,1)*have            # 边界处权重1，向内线性降到0
    return np.where(z,w*Bal+(1-w)*Bh,Bh),z
def band(m,w=3): return (binary_dilation(m,iterations=w)&~binary_erosion(m,iterations=w))&oc
if sys.argv[1]=='eval':
    rng=np.random.default_rng(11); N=int(sys.argv[2]); L=48
    c=[s for s in st.slot_range(VAL_START,VAL_END) if st.present[s] and 0.12<=st.coverage[s]<=0.4 and s>=L and st.present[s-L:s+2].all()]
    mon=np.array([st.times[s].month for s in c]); pick=[]
    for m in range(1,13):
        cm=[s for s,mm in zip(c,mon) if mm==m]
        if cm: pick+=list(rng.choice(cm,min(len(cm),max(1,N//12)),replace=False))
    Ws=[None,10,20,40]; E={w:[] for w in Ws}; Zs=[]; GR={w:[] for w in Ws}; VJ={w:[] for w in Ws}
    from preprocessing.seam_metrics import seam_stats
    for t in pick:
        t=int(t); tr=st.frame(t); known=np.isfinite(tr)&oc; ref=st.frame(t+1); mm=oc&~known&np.isfinite(ref); field=np.where(known,tr,np.nan)
        if mm.sum()<2000: continue
        res={}
        for W in Ws:
            Bf,z=fill_feather(t,field,known,W); F=np.where(known,np.nan_to_num(tr),np.nan_to_num(robust_bc(Bf,field,known,oc,2,0))); res[W]=(F,z)
        z=res[None][1]; Zs.append(z[mm]); 
        for W in Ws: E[W].append((res[W][0]-ref)[mm])
        if z.sum()>3000 and (oc&~known&~z).sum()>3000:
            cl=oc&~known&~z
            for W in Ws:
                F=res[W][0]; S=gaussian_filter(np.where(oc,F,0),2)/(gaussian_filter(oc.astype(float),2)+1e-9); gy,gx=np.gradient(S); g=np.hypot(gx,gy)
                b=band(z); rest=oc&~known&~binary_dilation(z,iterations=3)
                GR[W].append(g[b].mean()/g[rest].mean()); VJ[W].append(seam_stats(F,cl,z)['value_jump'])
    z=np.concatenate(Zs); rm=lambda e:np.sqrt(np.mean(e**2))
    out=['frames %d, C=0 占评估像元 %.1f%%; 梯度比统计帧数 %d'%(len(Zs),100*z.mean(),len(GR[None])),'%-8s %6s | C=0区 RMSE bias | 其余 RMSE | C=0边界带梯度/其余云区梯度 | C=0边界值台阶'%('羽化宽度','RMSE')]
    for W in Ws:
        e=np.concatenate(E[W]); out.append('%-8s %.3f | %.3f %+.3f | %.3f | %.2f | %.3f'%('无' if W is None else f'{W}px',rm(e),rm(e[z]),e[z].mean(),rm(e[~z]),np.mean(GR[W]),np.nanmean(VJ[W])))
    print('\n'.join(out)); open('feather_results.txt','w').write('\n'.join(out))
else:
    T=[datetime(2024,1,27,17),datetime(2023,7,13,23),datetime(2023,10,7,6)]; tix={t:i for i,t in enumerate(st.times)}
    for q,tm in enumerate(T):
        t=tix[tm]; tr=st.frame(t); known=np.isfinite(tr)&oc; field=np.where(known,tr,np.nan); miss=oc&~known; V={}
        for W in (None,20,40):
            Bf,z=fill_feather(t,field,known,W); V['baseline (erode2)' if W is None else f'feather {W}px']=np.where(known,tr,np.nan_to_num(robust_bc(Bf,field,known,oc,2,0)))
        lo,hi=np.percentile(np.concatenate([v[miss] for v in V.values()]),[1,99]); fig,ax=plt.subplots(2,3,figsize=(21,13))
        for j,(n,F) in enumerate(V.items()):
            im=ax[0,j].imshow(np.where(oc,F,np.nan),cmap='RdYlBu_r',vmin=lo,vmax=hi); ax[0,j].set_title(n); plt.colorbar(im,ax=ax[0,j],shrink=.7)
            S=gaussian_filter(np.where(oc,F,0),2)/(gaussian_filter(oc.astype(float),2)+1e-9); gy,gx=np.gradient(S); g=np.where(oc,np.hypot(gx,gy),np.nan)
            im=ax[1,j].imshow(g,cmap='magma',vmin=0,vmax=0.27); ax[1,j].contour(z.astype(float),[0.5],colors='lime',linewidths=0.8); ax[1,j].set_title('|∇T| (σ=2) + C=0 boundary'); plt.colorbar(im,ax=ax[1,j],shrink=.7)
        for a_ in ax.ravel(): a_.axis('off')
        fig.suptitle(str(tm)); plt.tight_layout(); plt.savefig(f'feather_vis{q}.png',dpi=55); plt.close()
