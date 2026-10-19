import sys, numpy as np
sys.path.insert(0,'/data1/user/lz/SST_Data_Imputation/Data_Imputation')
exec(open('ext_window.py').read().split("if sys.argv[1]=='eval'")[0])
from scipy.ndimage import gaussian_filter
T=[datetime(2024,1,27,17),datetime(2023,7,13,23),datetime(2023,10,7,6)]; tix={t:i for i,t in enumerate(st.times)}
def smooth(a,s=2.0):
    v=oc.astype(float); n=gaussian_filter(v,s)+1e-9; return np.where(oc,gaussian_filter(np.where(oc,a,0),s)/n,np.nan)
for q,tm in enumerate(T):
    t=tix[tm]; tr=st.frame(t); known=np.isfinite(tr)&oc; field=np.where(known,tr,np.nan)
    past={l:fr(t-l) for l in range(1,49)}; B=_weighted_background(past,_offsets(field,known,past),lambda l:1.0/l); z=np.isnan(B)&oc&~known
    Bf=_complete(B,oc); F=np.where(known,np.nan_to_num(tr),np.nan_to_num(robust_bc(Bf,field,known,oc,2,0)))
    S=smooth(F); gy,gx=np.gradient(S); g=np.hypot(gx,gy)
    lo,hi=np.nanpercentile(F[oc],[1,99]); vmax=np.nanpercentile(g,98)
    fig,ax=plt.subplots(1,3,figsize=(21,7))
    im=ax[0].imshow(np.where(oc,F,np.nan),cmap='RdYlBu_r',vmin=lo,vmax=hi); ax[0].set_title('filled (erode2, 48h)'); plt.colorbar(im,ax=ax[0],shrink=.7)
    im=ax[1].imshow(g,cmap='magma',vmin=0,vmax=vmax); ax[1].set_title('|∇T| of Gaussian-smoothed (σ=2) field, K/px'); plt.colorbar(im,ax=ax[1],shrink=.7)
    im=ax[2].imshow(g,cmap='magma',vmin=0,vmax=vmax)
    ax[2].contour(known.astype(float),[0.5],colors='cyan',linewidths=0.6); ax[2].contour(z.astype(float),[0.5],colors='lime',linewidths=0.9)
    ax[2].set_title('same + cyan: observed/cloud boundary, green: C=0 region boundary'); plt.colorbar(im,ax=ax[2],shrink=.7)
    for a_ in ax: a_.axis('off')
    inb=lambda m,w:(binary_erosion(m,iterations=w)^binary_erosion(m,iterations=0))
    from scipy.ndimage import binary_dilation
    def band(m,w=3): return (binary_dilation(m,iterations=w)&~binary_erosion(m,iterations=w))&oc
    gv=g[np.isfinite(g)]
    print(tm,'median|∇|=%.4f | 观测/云边界带 %.4f | C=0边界带 %.4f | 其余云区 %.4f'%(np.median(gv),np.nanmean(g[band(known)&~known|band(known)]),np.nanmean(g[band(z)]) if z.any() else np.nan,np.nanmean(g[(oc&~known&~band(z)&~band(known))])))
    fig.suptitle(str(tm)); plt.tight_layout(); plt.savefig(f'grad_map{q}.png',dpi=60); plt.close()
