import os
from contextlib import contextmanager
from typing import Any

import torch


def is_dist() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def init_distributed(backend: str = "nccl") -> None:
    if is_dist():
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend=backend)


def get_rank() -> int:
    if not is_dist():
        return 0
    return torch.distributed.get_rank()


def get_world_size() -> int:
    if not is_dist():
        return 1
    return torch.distributed.get_world_size()


def barrier() -> None:
    if is_dist():
        torch.distributed.barrier()


def all_reduce_tensor(t: torch.Tensor, op: str = "sum") -> torch.Tensor:
    if not is_dist():
        return t
    op_map = {
        "sum": torch.distributed.ReduceOp.SUM,
        "avg": torch.distributed.ReduceOp.AVG if hasattr(torch.distributed.ReduceOp, "AVG") else torch.distributed.ReduceOp.SUM,
        "max": torch.distributed.ReduceOp.MAX,
    }
    reduce_op = op_map.get(op, torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(t, op=reduce_op)
    if op == "avg" and get_world_size() > 1 and reduce_op == torch.distributed.ReduceOp.SUM:
        t = t / get_world_size()
    return t


@contextmanager
def on_main_process():
    if get_rank() == 0:
        yield True
    else:
        yield False


def gather_all_tensors(t: torch.Tensor) -> torch.Tensor:
    if not is_dist():
        return t
    tensors = [torch.zeros_like(t) for _ in range(get_world_size())]
    torch.distributed.all_gather(tensors, t)
    return torch.cat(tensors, dim=0)


