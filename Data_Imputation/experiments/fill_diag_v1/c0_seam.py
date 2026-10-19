import sys, numpy as np
sys.path.insert(0,'/data1/user/lz/SST_Data_Imputation/Data_Imputation')
exec(open('ext_window.py').read().split("if sys.argv[1]=='eval'")[0])
from preprocessing.seam_metrics import seam_stats, seam_reference
from skimage.restoration import inpaint_biharmonic
def complete_bih(B):
    unk=np.isnan(B)&oc; img=np.where(np.isfinite(B),B,0.0)
    _,(iy,ix)=edt(~(np.isfinite(B)&oc),return_indices=True); img[~oc]=img[iy[~oc],ix[~oc]]; img[unk]=0
    out=inpaint_biharmonic(img,unk) if unk.any() else img
    return np.where(np.isfinite(B),B,out)
rng=np.random.default_rng(3); L=48
c=[s for s in st.slot_range(VAL_START,VAL_END) if st.present[s] and 0.12<=st.coverage[s]<=0.4 and s>=L and st.present[s-L:s+2].all()]
pick=rng.choice(c,int(sys.argv[1]),replace=False)
M=['harm 48h','biharm 48h','harm 14d']; R={m:{'c0':[],'ob':[],'ref':[],'e':[]} for m in M}; ng=0
for t in pick:
    t=int(t); tr=st.frame(t); known=np.isfinite(tr)&oc; field=np.where(known,tr,np.nan); ref=st.frame(t+1); mm=oc&~known&np.isfinite(ref)
    past={l:fr(t-l) for l in range(1,49)}
    B=_weighted_background(past,_offsets(field,known,past),lambda l:1.0/l); z=np.isnan(B)&oc&~known
    cl=oc&~known&~z
    if z.sum()<3000 or cl.sum()<3000: continue
    ext={l:fr(t-l) for l in range(49,337,6)}; Be=_weighted_background(ext,_offsets(field,known,ext),lambda l:1.0/l); B14=B.copy(); q=np.isnan(B14)&np.isfinite(Be); B14[q]=Be[q]
    V={'harm 48h':_complete(B,oc),'biharm 48h':complete_bih(B),'harm 14d':_complete(B14,oc)}
    for n,Bf in V.items():
        F=np.where(known,np.nan_to_num(tr),np.nan_to_num(robust_bc(Bf,field,known,oc,2,0)))
        a=seam_stats(F,cl,z); R[n]['c0'].append((a['slope_jump'],a['value_jump']))
        o=seam_stats(F,known,oc&~known); R[n]['ob'].append((o['slope_jump'],o['value_jump']))
        r=seam_reference(F,known,z); R[n]['ref'].append((r['slope_jump'],r['value_jump']))
        R[n]['e'].append((F-ref)[mm])
    ng+=1
print('frames',ng); out=['边界两侧斜率跳变 slope_jump / 值台阶 value_jump (K/px)','%-11s | C>0云区->C=0区边界 | 观测->云区边界 | 观测区内参照 | RMSE'%'']
for n in M:
    f=lambda k:np.nanmean(np.array(R[n][k]),0); c0,ob,rf=f('c0'),f('ob'),f('ref'); e=np.concatenate(R[n]['e'])
    out.append('%-11s | %.3f / %.3f | %.3f / %.3f | %.3f / %.3f | %.3f'%(n,*c0,*ob,*rf,np.sqrt((e**2).mean())))
print('\n'.join(out)); open('c0_seam_results.txt','w').write('\n'.join(out))
