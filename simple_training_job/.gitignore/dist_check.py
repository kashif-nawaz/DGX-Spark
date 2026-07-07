# This tiny program verifies that PyTorch Distributed (NCCL) is working.
#
# All four nodes join one communication group. Each node contributes its
# rank number (0, 1, 2, 3), and NCCL performs an all-reduce SUM operation.
#
# Expected result:
#   0 + 1 + 2 + 3 = 6
#
# If everything is working correctly, every node should print:
#   rank X/4 all_reduce result = 6.0

import os
import torch
import torch.distributed as dist

# Initialize the distributed process group using the NCCL backend.
# This establishes communication between all participating nodes.
dist.init_process_group(backend="nccl")

# Get this process's global rank (0..3) and the total number of processes.
rank = dist.get_rank()
world = dist.get_world_size()

# Each DGX Spark has one GPU, so LOCAL_RANK is always 0.
# Tell PyTorch which GPU this process should use.
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# Create a GPU tensor containing this node's rank value.
# Rank 0 -> tensor([0])
# Rank 1 -> tensor([1])
# Rank 2 -> tensor([2])
# Rank 3 -> tensor([3])
t = torch.ones(1, device="cuda") * rank

# Perform an all-reduce SUM across all nodes.
# After this call, every node receives the same result.
#
# Before:
#   Rank 0 : 0
#   Rank 1 : 1
#   Rank 2 : 2
#   Rank 3 : 3
#
# After:
#   Rank 0 : 6
#   Rank 1 : 6
#   Rank 2 : 6
#   Rank 3 : 6
dist.all_reduce(t)

# Display the result on every node.
print(f"rank {rank}/{world} all_reduce result = {t.item()}", flush=True)

# Leave the distributed communication group cleanly.
dist.destroy_process_group()
