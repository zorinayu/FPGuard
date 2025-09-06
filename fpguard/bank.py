from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np


@dataclass
class FingerprintMeta:
    seq_id: str
    token_idx: int
    layer_id: int
    node_id: int


class FingerprintBank:
    def __init__(self, dim: int, pq_m: int = 16, use_hnsw: bool = True):
        self.dim = dim
        self.pq_m = pq_m
        self.use_hnsw = use_hnsw

        self.coarse = faiss.IndexFlatL2(dim)
        self.pq = faiss.ProductQuantizer(dim, pq_m, 8)

        self.index = faiss.IndexIVFPQ(self.coarse, dim, 4096, pq_m, 8)
        if use_hnsw:
            self.index = faiss.IndexHNSWFlat(dim, 32)

        self.metas: List[FingerprintMeta] = []

    def train(self, vectors: np.ndarray) -> None:
        if isinstance(self.index, faiss.IndexIVFPQ):
            if not self.index.is_trained:
                self.index.train(vectors)

    def add(self, vectors: np.ndarray, metas: List[FingerprintMeta]) -> None:
        assert vectors.shape[0] == len(metas)
        if isinstance(self.index, faiss.IndexIVFPQ) and not self.index.is_trained:
            self.train(vectors)
        self.index.add(vectors)
        self.metas.extend(metas)

    def search(self, queries: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray, List[FingerprintMeta]]:
        distances, indices = self.index.search(queries, top_k)
        return distances, indices, self.metas


