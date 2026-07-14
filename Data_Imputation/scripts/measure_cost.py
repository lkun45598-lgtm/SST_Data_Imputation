"""P1-8: model size + compute cost (params / size / peak GPU mem / inference time / FLOPs)."""
import sys, time; from pathlib import Path
import numpy as np, torch
DI=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(DI))
from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal
dev=torch.device("cuda:0")
m=FNO_CBAM_SST_Temporal(out_size=(451,351),modes1=80,modes2=64,width=64,depth=6,cbam_reduction_ratio=16).to(dev).eval()

# --- params ---
tot=sum(p.numel() for p in m.parameters())
train=sum(p.numel() for p in m.parameters() if p.requires_grad)
# breakdown by top-level submodule
bd={}
for n,mod in m.named_children():
    bd[n]=sum(p.numel() for p in mod.parameters())
print(f"total params      : {tot:,}  ({tot/1e6:.1f} M)")
print(f"trainable         : {train:,}")
print(f"model size (fp32) : {tot*4/1024**2:.0f} MB  ({tot*4/1024**3:.2f} GB)")
print("top submodule param breakdown:")
for n,v in sorted(bd.items(),key=lambda x:-x[1]):
    print(f"  {n:<22}{v:,>0}  ({v/1e6:.1f} M, {100*v/tot:.0f}%)")

# --- inference input (batch=1, 30 frames) ---
st=torch.randn(1,30,451,351,device=dev); mt=torch.zeros(1,30,451,351,device=dev)

# --- peak memory (inference, bs=1) ---
torch.cuda.reset_peak_memory_stats(dev); torch.cuda.empty_cache()
with torch.no_grad(): _=m(st,mt)
torch.cuda.synchronize()
peak=torch.cuda.max_memory_allocated(dev)/1024**3
print(f"\npeak GPU mem (inference, bs=1) : {peak:.2f} GB")

# --- inference time (bs=1) ---
with torch.no_grad():
    for _ in range(3): _=m(st,mt)         # warmup
    torch.cuda.synchronize(); t0=time.time(); N=20
    for _ in range(N): _=m(st,mt)
    torch.cuda.synchronize(); dt=(time.time()-t0)/N
print(f"inference time (bs=1)         : {dt*1000:.1f} ms/frame  ({1/dt:.1f} frames/s)")
print(f"  => full product 73,004 frames single-GPU ~ {73004*dt/3600:.1f} h")

# --- FLOPs via thop ---
try:
    from thop import profile
    macs,_=profile(m,inputs=(st,mt),verbose=False)
    print(f"FLOPs (bs=1)                  : {2*macs/1e9:.1f} GFLOPs  (MACs {macs/1e9:.1f} G)")
except Exception as e:
    print(f"FLOPs: thop failed ({e})")
