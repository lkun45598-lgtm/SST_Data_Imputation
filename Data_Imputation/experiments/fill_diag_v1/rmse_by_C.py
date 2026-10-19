import sys, numpy as np
sys.path.insert(0,'/data1/user/lz/SST_Data_Imputation/Data_Imputation')
from scipy import ndimage as ndi
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from datasets.hourly_store import HourlyStore
from preprocessing.fill_compare import build_cloud_bank, VAL_START, VAL_END, NEED_PAST
from preprocessing.fill_temporal import _offsets,_weighted_background,_complete,_boundary_correct
from preprocessing.fill_methods import harmonic_fill
N=int(sys.argv[1]); rng=np.random.default_rng(0)
st=HourlyStore(); oc=st.ocean
bank=build_cloud_bank(st,n=200)
val=[s for s in st.slot_range(VAL_START,VAL_END) if st.present[s] and 0.3<=st.coverage[s]<=0.6 and s>=NEED_PAST and st.present[s-NEED_PAST:s].all()]
val=np.array(val); mon=np.array([st.times[s].month for s in val])
ts=[]
for m in range(1,13):
    c=val[mon==m]
    if len(c): ts+=list(rng.choice(c,min(len(c),max(1,N//12)),replace=False))
print('frames',len(ts),flush=True)
R={k:[] for k in ('C','D','rad','e_tw','e_bc','e_h','mon')}; demo=None
for t in ts:
    t=int(t); tr=st.frame(t); fin=np.isfinite(tr)&oc
    hole=bank[rng.integers(len(bank))]&fin
    if hole.sum()<500: continue
    known=fin&~hole; field=np.where(known,tr,np.nan)
    past={l:st.frame(t-l) for l in range(1,49)}
    valid=np.stack([np.isfinite(past[l]) for l in past])&oc
    C=valid.sum(0); D=ndi.distance_transform_edt(C==0)
    rad=ndi.distance_transform_edt(~known)
    offs=_offsets(field,known,past)
    Bf=_complete(_weighted_background(past,offs,lambda l:1.0/l),oc)
    Bc=_boundary_correct(Bf,field,known,oc)
    H=np.nan_to_num(harmonic_fill(field,known,oc))
    for k,v in (('C',C),('D',D),('rad',rad),('e_tw',Bf-tr),('e_bc',Bc-tr),('e_h',H-tr)): R[k].append(v[hole])
    R['mon'].append(np.full(hole.sum(),st.times[t].month))
    if demo is None and False or demo is None and (C[hole]==0).mean()>0.05: demo=(tr,hole,C,Bf-tr,Bc-tr)
demo=demo or (tr,hole,C,Bf-tr,Bc-tr)
R={k:np.concatenate(v) for k,v in R.items()}; np.savez('/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/fill_diag_v1/rmse_by_C.npz',**R)
rm=lambda e:np.sqrt(np.mean(e**2)) if len(e) else np.nan
bins=[0,1,3,6,12,24,49]; lab=['0','1-2','3-5','6-11','12-23','24-48']
print('%-6s %6s %8s %8s %8s %8s %8s'%('C','share','RMSE_tw','RMSE_bc','RMSE_h','bias_tw','bias_bc'))
rows=[]
for a,b,l in zip(bins[:-1],bins[1:],lab):
    m=(R['C']>=a)&(R['C']<b); rows.append((l,m.mean(),rm(R['e_tw'][m]),rm(R['e_bc'][m]),rm(R['e_h'][m]),R['e_tw'][m].mean(),R['e_bc'][m].mean()))
    print('%-6s %5.1f%% %8.3f %8.3f %8.3f %8.3f %8.3f'%((l,100*m.mean())+rows[-1][2:]))
z=R['C']==0; print('C=0 内按 D48 (到最近历史观测 px):')
dz=[(0,5),(5,10),(10,20),(20,500)]; zr=[]
for a,b in dz:
    m=z&(R['D']>a)&(R['D']<=b); zr.append((f'{a}-{b}',m.sum()/z.sum(),rm(R['e_tw'][m]),rm(R['e_h'][m])))
    print('  D48 (%d,%d] 占C=0的%.1f%%  RMSE_tw %.3f  RMSE_h %.3f'%((a,b,100*zr[-1][1],zr[-1][2],zr[-1][3])))
print('总 RMSE tw %.3f bc %.3f  纯空间harmonic %.3f'%(rm(R['e_tw']),rm(R['e_bc']),rm(R['e_h'])))
fig,ax=plt.subplots(1,3,figsize=(16,4.2))
x=np.arange(len(rows)); w=.27
ax[0].bar(x-w,[r[2] for r in rows],w,label='temporal (tw)'); ax[0].bar(x,[r[3] for r in rows],w,label='temporal+bc'); ax[0].bar(x+w,[r[4] for r in rows],w,label='spatial harmonic')
ax[0].set_xticks(x); ax[0].set_xticklabels([f'{r[0]}\n({100*r[1]:.0f}%)' for r in rows]); ax[0].set_xlabel('C: obs count in past 48h (share of hole px)'); ax[0].set_ylabel('RMSE (K)'); ax[0].legend(); ax[0].set_title('RMSE vs coverage C')
x2=np.arange(len(zr)); ax[1].bar(x2-.2,[r[2] for r in zr],.4,label='temporal (tw)'); ax[1].bar(x2+.2,[r[3] for r in zr],.4,label='spatial harmonic')
ax[1].set_xticks(x2); ax[1].set_xticklabels([f'{r[0]}\n({100*r[1]:.0f}%)' for r in zr]); ax[1].set_xlabel('C=0: dist to nearest 48h obs (px)'); ax[1].legend(); ax[1].set_title('C=0 subset by D48')
tr,hole,C,e1,e2=demo; im=ax[2].imshow(np.where(oc,np.where(hole,C,np.nan),np.nan),cmap='viridis'); plt.colorbar(im,ax=ax[2],label='C in hole'); ax[2].set_title('demo frame: C inside hole'); ax[2].axis('off')
plt.tight_layout(); plt.savefig('/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/fill_diag_v1/rmse_by_C.png',dpi=130)
fig,ax=plt.subplots(1,3,figsize=(16,4.5))
for a,(d,ti,v) in zip(ax,((np.where(hole,C,np.nan),'C in hole','viridis'),(np.where(hole,e1,np.nan),'error tw (K)','RdBu_r'),(np.where(hole,e2,np.nan),'error tw+bc (K)','RdBu_r'))[0:3] and [(np.where(hole,C,np.nan),'C in hole','viridis'),(np.where(hole,e1,np.nan),'tw error (K)','RdBu_r'),(np.where(hole,e2,np.nan),'tw+bc error (K)','RdBu_r')]):
    im=a.imshow(d,cmap=v,**({'vmin':-2,'vmax':2} if 'error' in ti else {})); a.set_title(ti); a.axis('off'); plt.colorbar(im,ax=a,shrink=.8)
plt.tight_layout(); plt.savefig('/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/fill_diag_v1/demo_maps.png',dpi=130)
