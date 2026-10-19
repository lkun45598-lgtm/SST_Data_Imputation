import sys, numpy as np, h5py
sys.path.insert(0,'/data1/user/lz/SST_Data_Imputation/Data_Imputation')
exec(open('feather.py').read().split("\nif sys.argv[1]=='eval':")[0])
from preprocessing.fill_temporal import _smooth
from matplotlib.backends.backend_pdf import PdfPages
rng=np.random.default_rng(2026); L=48
c=[s for s in st.slot_range(VAL_START,VAL_END) if st.present[s] and 0.12<=st.coverage[s]<=0.30 and s>=L and st.present[s-L:s+1].all()]
pick=[]
for months in ((12,1,2),(3,4,5),(6,7,8),(9,10,11)):
    cm=[s for s in c if st.times[s].month in months]
    pick+=list(rng.choice(cm,min(3,len(cm)),replace=False))
base='/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/hourly_data'
def bc_clip(Bf,field,known,erode=2,clip=1.0):
    src=source(known,erode,0)
    if src.sum()<100: src=known
    r=np.where(src,field-Bf,0.0); r=clip*np.tanh(r/clip)
    out=Bf+_harmonic_solve(r,src,oc)*(~src); return np.where(known,field,out)
pdf=PdfPages('gallery_clip/all_frames.pdf'); idx=[]
for q,t in enumerate(pick):
    t=int(t); tm=st.times[t]; tr=st.frame(t); known=np.isfinite(tr)&oc; field=np.where(known,tr,np.nan); miss=oc&~known
    past={l:fr(t-l) for l in range(1,49)}; B=_weighted_background(past,_offsets(field,known,past),lambda l:1.0/l); z=np.isnan(B)&oc&~known
    Bh=_complete(B,oc); C=(np.stack([np.isfinite(past[l]) for l in past])&oc).sum(0)
    mk=lambda Bx: np.where(known,tr,np.nan_to_num(robust_bc(Bx,field,known,oc,2,0)))
    V={'tw only (no bc)':np.where(known,tr,np.nan_to_num(Bh)),'bc erode2 (before clip)':mk(Bh),'bc erode2 + tanh clip 1K (new)':np.where(known,tr,np.nan_to_num(bc_clip(Bh,field,known)))}
    try:
        f=h5py.File(f'{base}/h{tm.hour:02d}/jaxa_knn_filled_07.h5','r'); ts=[datetime.strptime(x.decode()[:10],'%Y-%m-%d') for x in f['timestamps'][:]]
        orig=f['sst_data'][ts.index(datetime(tm.year,tm.month,tm.day))].astype(float); V={'original pipeline (time+gauss+3D KNN)':np.where(known,tr,orig),**V}
    except Exception: pass
    allv=np.concatenate([v[miss] for v in V.values()]); bad=int((~np.isfinite(allv)).sum())
    if bad: print('  非有限值',bad,[k for k,v in V.items() if (~np.isfinite(v[miss])).any()],flush=True)
    lo,hi=np.nanpercentile(allv[np.isfinite(allv)],[1,99])
    P=[('real input (blank=cloud)',np.where(known,tr,np.nan),'RdYlBu_r',lo,hi),('C: obs count in past 48h',np.where(miss,C,np.nan),'viridis',0,48)]+[(k,v,'RdYlBu_r',lo,hi) for k,v in V.items()]
    F=V['bc erode2 + tanh clip 1K (new)']; S=gaussian_filter(np.where(oc,F,0),2)/(gaussian_filter(oc.astype(float),2)+1e-9); gy,gx=np.gradient(S); P.append(('new minus tw only (K)',np.where(miss,V['bc erode2 + tanh clip 1K (new)']-V['tw only (no bc)'],np.nan),'RdBu_r',-1.5,1.5)); P.append(('|∇T| of new result, green=C=0 boundary',np.where(oc,np.hypot(gx,gy),np.nan),'magma',0,0.27))
    fig,ax=plt.subplots(2,4,figsize=(24,13)); ax=ax.ravel()
    for a_,(ti,d,cm,l,h) in zip(ax,P):
        im=a_.imshow(np.where(oc,d,np.nan),cmap=cm,vmin=l,vmax=h); a_.set_title(ti,fontsize=13); a_.axis('off'); plt.colorbar(im,ax=a_,shrink=.7)
        if ti.startswith('|∇T|') and z.any(): a_.contour(z.astype(float),[0.5],colors='lime',linewidths=0.8)
    for a_ in ax[len(P):]: a_.axis('off')
    fig.suptitle('%s   coverage %.1f%%   C=0 share of cloud area %.0f%%   (same colorbar on all fills)'%(tm,100*st.coverage[t],100*z.sum()/max(miss.sum(),1)),fontsize=15)
    plt.tight_layout(); name='gallery_clip/%02d_%s.png'%(q,tm.strftime('%Y%m%d_%H')); plt.savefig(name,dpi=60); pdf.savefig(fig,dpi=60); plt.close(); idx.append((name,tm,float(st.coverage[t])))
    print(name,flush=True)
pdf.close()
open('gallery_clip/README.txt','w').write('每个 PNG: 真实输入 / C / 原流水线 / tw / 现状(tw+bc erode2) / 现状-tw 差 / 梯度图。填充面板同色标。\n'+'\n'.join('%s  %s  覆盖率 %.1f%%'%(n,t,100*cv) for n,t,cv in idx))
