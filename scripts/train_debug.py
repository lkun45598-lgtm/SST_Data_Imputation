#!/usr/bin/env python3
"""Debug DDP startup"""
import os
import sys
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, 'Data_Imputation')

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29510'
    
    print(f"[Rank {rank}] init_process_group...", flush=True)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=30))
    torch.cuda.set_device(rank)
    print(f"[Rank {rank}] DDP OK", flush=True)

def cleanup():
    dist.destroy_process_group()

def train_worker(rank, world_size):
    print(f"[Rank {rank}] started", flush=True)
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    from inference.jaxa_inference_dataset import JAXAFinetuneDataset
    print(f"[Rank {rank}] loading dataset...", flush=True)
    dataset = JAXAFinetuneDataset(
        data_dir='/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/hourly_data/h01',
        series_ids=[0], window_size=30, mask_ratio=0.2, min_mask_size=10, max_mask_size=50,
        normalize=True, mean=299.9221, std=2.6919, cache_size=100, seed=42
    )
    print(f"[Rank {rank}] dataset OK: {len(dataset)}", flush=True)
    
    from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal
    print(f"[Rank {rank}] creating model...", flush=True)
    model = FNO_CBAM_SST_Temporal(out_size=(451, 351), modes1=80, modes2=64, width=64, depth=6).to(device)
    print(f"[Rank {rank}] model OK", flush=True)
    
    cleanup()

def main():
    mp.spawn(train_worker, args=(4,), nprocs=4, join=True)

if __name__ == '__main__':
    main()
