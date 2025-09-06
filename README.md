# FPGuard

Distributed FPGuard: Parallel Fingerprint Attribution for scalable contamination detection in distributed generative models.

- Token-level multi-layer fingerprint extraction
- Contrastive projection heads for discriminative fingerprints
- FAISS-based scalable fingerprint bank (IVF/HNSW, PQ compression)
- Inference-time contamination scoring, top-k retrieval, and token-wise attribution
- CUDA kernel for fast row-wise Top-K with auto-fallback
- Post-hoc detoxification via attribution aggregation
- CLI tools for training, bank building, inference, and detox

## Requirements

- Python 3.9+
- PyTorch 2.2+ (CUDA optional but recommended)
- Build toolchain for CUDA extension (Linux recommended; macOS uses CPU fallback)

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Notes:
- For GPU FAISS, use a CUDA build (e.g., `faiss-gpu`) matching your CUDA version.
- On macOS, CUDA is unavailable; the CUDA extension will auto-disable.

## CUDA Extension (Optional)

JIT-compiled CUDA extension for row-wise Top-K in similarity search.

Files:
- `fpguard/cuda/ops.cpp`
- `fpguard/cuda/sim_topk.cu`
- Loader: `fpguard/cuda_ops.py`

Runtime behavior:
- If `torch.cuda.is_available()` and NVCC are present, the extension compiles on first use.
- Otherwise, fallback to `torch.topk` is used.

Manual check:
```python
from fpguard.cuda_ops import available
print("CUDA ops available:", available())
```

Troubleshooting:
- Ensure `nvcc --version` matches `torch.version.cuda`.
- Clear cache: remove `~/.cache/torch_extensions/*`.

## Quickstart

### Inference + attribution
```python
import torch, numpy as np
from fpguard.inference import contamination_scores

Q, D, K, top_k = 64, 512, 10000, 32
queries = torch.randn(Q, D, device="cuda" if torch.cuda.is_available() else "cpu")
bank = np.random.randn(K, D).astype("float32")

gamma, top_idx = contamination_scores(queries, bank, alpha=0.7, top_k=top_k)
print(gamma.shape, top_idx.shape)  # [Q], [Q, top_k]
```

### Post-hoc detoxification
```python
import numpy as np
from fpguard.detox import aggregate_attribution

T, M = 128, 10000
attrib = np.abs(np.random.randn(T, M)).astype("float32")
seq_ids = [f"seq_{j}" for j in range(M)]
removed_ids = aggregate_attribution(attrib, seq_ids, threshold=50.0)
print("Remove:", removed_ids[:5])
```

## Python APIs

Hidden states → fingerprints:
```python
from fpguard.models import HiddenStateExtractor
from fpguard.fingerprints import ProjectionHeads, concat_fingerprints

extractor = HiddenStateExtractor("meta-llama/Llama-2-7b-hf")
_, hidden = extractor.encode_layers("Hello FPGuard", [4, 8, 12, 16])
layer_dims = {lid: hidden[lid].shape[-1] for lid in hidden}
proj = ProjectionHeads(layer_dims, projection_dim=128)
fp_by_layer = proj(hidden)  # dict[layer_id] -> [T, 128]
fp = concat_fingerprints(fp_by_layer, layer_order=[4, 8, 12, 16])  # [T, 512]
```

FAISS bank:
```python
import numpy as np
from fpguard.bank import FingerprintBank, FingerprintMeta

bank = FingerprintBank(dim=512, pq_m=16, use_hnsw=True)
vecs = np.random.randn(50000, 512).astype("float32")
metas = [FingerprintMeta(seq_id=str(i), token_idx=0, layer_id=0, node_id=0) for i in range(len(vecs))]
bank.train(vecs)
bank.add(vecs, metas)
dists, idx, metas_ref = bank.search(vecs[:10], top_k=32)
```

## Distributed Notes

- `fpguard/distributed.py` provides minimal helpers for DDP/NCCL usage.
- Shard dataset/bank building across nodes; merge indices or search per-shard and fuse.

## Troubleshooting

- Missing torch/transformers: `pip install -r requirements.txt`.
- CUDA build fails: check `nvcc`, CUDA toolkit, and drivers.
- FAISS GPU: install `faiss-gpu` matching your CUDA version.
- JIT cache issues: delete `~/.cache/torch_extensions/*`.

## License

TBD.
