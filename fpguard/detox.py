import numpy as np
from typing import Dict, List, Tuple


def aggregate_attribution(attrib_matrix: np.ndarray, seq_ids: List[str], threshold: float) -> List[str]:
    # attrib_matrix: [T, M] attribution weights; seq_ids map columns to training sequences
    per_seq: Dict[str, float] = {}
    for j, sid in enumerate(seq_ids):
        per_seq[sid] = per_seq.get(sid, 0.0) + float(attrib_matrix[:, j].sum())
    removed = [sid for sid, score in per_seq.items() if score > threshold]
    return removed


