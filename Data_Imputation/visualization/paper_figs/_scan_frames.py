#!/usr/bin/env python3
"""Quick contact sheet: FNO-filled SST + gradient for candidate frames.
Loads FNO once, renders small previews so we can pick the best scene."""
import sys, argparse
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

DI = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DI)); sys.path.insert(0, str(DI/"baselines"))
sys.path.insert(0, str(DI/"visualization/paper_figs"))
from baselines.eval_all_methods import load_fno, run_torch_model, FNO_MEAN, FNO_STD, W
from style import apply_paper_style, CMAPS, LAND_COLOR

H12 = DI/"experiments/hourly_data/h12"
OUT = DI/"visualization/paper_figs/output"
GRAD = mpl.colormaps["magma"]; SST = CMAPS["sst"]

def gradmag(f, ocean):
    ff = np.where(ocean, f, np.nan); gy, gx = np.gradient(ff)
    return np.where(ocean, np.sqrt(gy**2+gx**2), np.nan)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--frames", type=str, required=True)
    args = ap.parse_args()
    import torch
    dev = torch.device(f"cuda:{args.gpu}")
    apply_paper_style()
    frames = [int(x) for x in args.frames.split(",")]
    with h5py.File(H12/"jaxa_knn_filled_08.h5","r") as f:
        sst=f["sst_data"][:].astype(np.float32)
        obs=f["original_obs_mask"][:].astype(np.uint8)
        miss=f["original_missing_mask"][:].astype(np.uint8)
        land=f["land_mask"][:].astype(np.uint8)
        ts=[t.decode() if isinstance(t,bytes) else t for t in f["timestamps"][:]]
    ocean=(land==0); model=load_fno(dev)
    n=len(frames)
    fig,axes=plt.subplots(2,n,figsize=(3.0*n,6.2),layout="constrained")
    if n==1: axes=axes.reshape(2,1)
    for j,idx in enumerate(frames):
        cloud=((miss[idx]==1)&ocean).astype(np.float32)
        ss=np.stack([sst[max(0,idx-(W-1)+t)] for t in range(W)])
        mm=np.stack([miss[max(0,idx-(W-1)+t)] for t in range(W)])
        out=run_torch_model(model,ss,mm,cloud,FNO_MEAN,FNO_STD,dev)
        full=np.where(cloud>0,out,sst[idx])
        fc=np.where(ocean,full-273.15,np.nan)
        obsf=np.where((obs[idx]==1)&ocean,sst[idx]-273.15,np.nan)
        vmin,vmax=np.nanpercentile(obsf,[2,98])
        g=gradmag(fc,ocean); gmax=np.nanpercentile(g,98)
        for r,(img,cm,vv) in enumerate([(fc,SST,(vmin,vmax)),(g,GRAD,(0,gmax))]):
            ax=axes[r,j]
            ax.imshow(np.where(~ocean,1.0,np.nan),cmap=mpl.colors.ListedColormap([LAND_COLOR]),origin="lower",aspect="equal")
            ax.imshow(img,cmap=cm,vmin=vv[0],vmax=vv[1],origin="lower",aspect="equal")
            ax.contour(cloud,levels=[0.5],colors="w",linewidths=0.4,origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
        o=(obs[idx]*ocean).sum()/ocean.sum()
        axes[0,j].set_title(f"#{idx} {ts[idx][:10]}\nobs {o*100:.0f}%",fontsize=9)
    axes[0,0].set_ylabel("FNO filled SST",fontsize=10)
    axes[1,0].set_ylabel("|grad SST|",fontsize=10)
    p=OUT/"_scan_frames.png"; fig.savefig(p,dpi=140); plt.close(fig)
    print("saved",p)

if __name__=="__main__":
    main()
