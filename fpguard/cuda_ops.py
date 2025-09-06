import os
from typing import Tuple

import torch
from torch.utils.cpp_extension import load as load_ext


_ext = None


def _load_extension() -> None:
    global _ext
    if _ext is not None:
        return
    this_dir = os.path.dirname(os.path.abspath(__file__))
    src_cpp = os.path.join(this_dir, "cuda", "ops.cpp")
    src_cu = os.path.join(this_dir, "cuda", "sim_topk.cu")
    if not (os.path.exists(src_cpp) and os.path.exists(src_cu)):
        _ext = None
        return
    _ext = load_ext(
        name="fpguard_cuda_ops",
        sources=[src_cpp, src_cu],
        verbose=False,
        extra_cuda_cflags=["-O3"],
    )


def available() -> bool:
    if not torch.cuda.is_available():
        return False
    _load_extension()
    return _ext is not None


@torch.no_grad()
def cosine_sim_topk(a: torch.Tensor, b: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # a: [Q, D], b: [K, D]
    assert a.is_cuda and b.is_cuda, "Inputs must be CUDA tensors"
    a_norm = torch.nn.functional.normalize(a, dim=-1)
    b_norm = torch.nn.functional.normalize(b, dim=-1)
    scores = a_norm @ b_norm.t()  # [Q, K]

    if available():
        vals, idx = _ext.rowwise_topk(scores, int(k))
        return vals, idx
    # Fallback
    vals, idx = torch.topk(scores, k=min(k, scores.shape[-1]), dim=-1)
    return vals, idx


