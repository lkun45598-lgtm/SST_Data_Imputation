"""三个真实低覆盖时刻的对比图: 经纬度坐标、色标单位、陆地/云遮挡分色、完整标题。"""
import sys, numpy as np, h5py
sys.argv=['x','vis']
exec(open('ext_window.py').read().split("if sys.argv[1]=='eval'")[0])
import matplotlib.colors as mcolors
with h5py.File(st.path,'r') as f: lat=f['lat'][:]; lon=f['lon'][:]
origin='upper' if lat[0]>lat[-1] else 'lower'
ext=[lon.min(),lon.max(),lat.min(),lat.max()]
print('lat',lat[0],lat[-1],'lon',lon[0],lon[-1],'origin',origin)
LAND,GAP='#dccba5','#b8b8b8'
def show(ax,arr,cmap,vmin,vmax,label,title,gap=None):
    base=np.where(oc,0.0,np.nan)
    ax.imshow(np.where(oc,0,np.nan),cmap=mcolors.ListedColormap([GAP]),extent=ext,origin=origin,aspect='equal')   # 先铺底色(云遮挡)
    ax.imshow(np.where(~oc,0,np.nan),cmap=mcolors.ListedColormap([LAND]),extent=ext,origin=origin,aspect='equal')   # 陆地
    im=ax.imshow(np.where(oc,arr,np.nan),cmap=cmap,vmin=vmin,vmax=vmax,extent=ext,origin=origin,aspect='equal')
    ax.set_title(title,fontsize=11); ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
    ax.tick_params(labelsize=8); cb=plt.colorbar(im,ax=ax,shrink=.78,pad=.02); cb.set_label(label,fontsize=9)
base='/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/hourly_data'
T=[datetime(2024,1,27,17),datetime(2023,7,13,23),datetime(2023,10,7,6)]; tix={t:i for i,t in enumerate(st.times)}
for q,tm in enumerate(T):
    t=tix[tm]; tr=st.frame(t); known=np.isfinite(tr)&oc; miss=oc&~known; field=np.where(known,tr,np.nan)
    past={l:fr(t-l) for l in range(1,49)}; C=(np.stack([np.isfinite(past[l]) for l in past])&oc).sum(0)
    Bf,_=background(t,field,known,48,1); Bl,_=background(t,field,known,336,6)
    V={'tw (time-weighted background only)':Bf,'tw+bc (bias correction, no edge erosion)':robust_bc(Bf,field,known,oc,0,0),
       'erode2 bc (current best)':robust_bc(Bf,field,known,oc,2,0),'erode2 bc + 14-day window for C=0':robust_bc(Bl,field,known,oc,2,0)}
    orig=None
    try:
        f=h5py.File(f'{base}/h{tm.hour:02d}/jaxa_knn_filled_07.h5','r'); ts=[datetime.strptime(x.decode()[:10],'%Y-%m-%d') for x in f['timestamps'][:]]
        orig=f['sst_data'][ts.index(datetime(tm.year,tm.month,tm.day))].astype(float)
    except Exception as e: print('orig missing',e)
    fills={k:np.where(known,tr,np.nan_to_num(v)) for k,v in V.items()}
    if orig is not None: fills={'original pipeline (time+Gaussian+3D KNN)':orig,**fills}
    allv=np.concatenate([v[miss] for v in fills.values()]); lo,hi=np.nanpercentile(allv,[1,99])
    fig,ax=plt.subplots(2,4,figsize=(24,12.5)); ax=ax.ravel()
    show(ax[0],field,'RdYlBu_r',lo,hi,'SST (K)','(a) Real observation (gray = cloud, beige = land)')
    show(ax[1],np.where(miss,C,np.nan),'viridis',0,48,'No. of valid obs in past 48 h','(b) History coverage C inside cloud area')
    for k,(a_,(n,v)) in enumerate(zip(ax[2:],fills.items())): show(a_,v,'RdYlBu_r',lo,hi,'SST (K)','(%s) %s'%('cdefgh'[k],n))
    for a_ in ax[2+len(fills):]: a_.axis('off')
    fig.suptitle('JAXA hourly SST gap filling, %s UTC, ocean coverage %.1f%% (all SST panels share one colorbar; no ground truth)'%(tm.strftime('%Y-%m-%d %H:00'),100*st.coverage[t]),fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(f'vis_all{q}.png',dpi=70); plt.close(); print(q,tm)
