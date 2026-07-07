import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model = nn.Sequential(nn.Linear(1024, 4096), nn.ReLU(), nn.Linear(4096, 10)).to(device)
    model = DDP(model, device_ids=[local_rank])

    x = torch.randn(50000, 1024)
    y = torch.randint(0, 10, (50000,))
    ds = TensorDataset(x, y)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=global_rank, shuffle=True)
    loader = DataLoader(ds, batch_size=64, sampler=sampler, num_workers=0, pin_memory=False)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        sampler.set_epoch(epoch)
        model.train()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        if global_rank == 0:
            print(f"epoch {epoch} loss {loss.item():.4f}", flush=True)

    if global_rank == 0:
        torch.save(model.module.state_dict(), os.path.expanduser("~/ckpt.pt"))
        print("saved checkpoint to ~/ckpt.pt", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
