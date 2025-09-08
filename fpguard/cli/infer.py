import json
import os
from typing import Optional

import numpy as np  # type: ignore
import typer  # type: ignore
import faiss  # type: ignore
import torch  # type: ignore

from fpguard.inference import contamination_scores


app = typer.Typer(add_completion=False)


@app.command()
def main(
    queries_path: str = typer.Argument(..., help="Path to .npy queries [Q, D]"),
    bank_dir: str = typer.Argument(..., help="Directory with bank.faiss and bank.json"),
    top_k: int = typer.Option(32, help="Top-K neighbors"),
    alpha: float = typer.Option(0.7, help="Contamination mixing factor"),
    output_path: Optional[str] = typer.Option(None, help="Optional output .npz to save scores and indices"),
):
    """Client inference: compute contamination scores against a saved bank."""
    queries = np.load(queries_path)
    assert queries.ndim == 2 and queries.dtype == np.float32, "queries must be float32 [Q, D]"

    cfg_path = os.path.join(bank_dir, "bank.json")
    index_path = os.path.join(bank_dir, "bank.faiss")
    assert os.path.exists(cfg_path) and os.path.exists(index_path), "Missing bank artifacts"

    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    d = int(cfg["dim"])  # noqa: F841

    index = faiss.read_index(index_path)

    # Extract raw bank vectors when possible; if not, fall back by probing indices via add/reconstruct
    # For simplicity, we store vectors separately: here we reconstruct if available
    if hasattr(index, "reconstruct_n"):
        K = int(index.ntotal)
        bank_vecs = np.zeros((K, queries.shape[1]), dtype=np.float32)
        for i in range(0, K, 100000):
            n_chunk = min(100000, K - i)
            bank_vecs[i : i + n_chunk] = index.reconstruct_n(i, n_chunk)
    else:
        raise RuntimeError("Index type does not support reconstruct_n; please provide bank vectors.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.from_numpy(queries).to(device)
    gamma, top_idx = contamination_scores(q, bank_vecs, alpha=alpha, top_k=top_k)

    if output_path:
        np.savez_compressed(output_path, scores=gamma.cpu().numpy(), indices=top_idx.cpu().numpy())
        typer.echo(f"Saved outputs to {output_path}")
    else:
        typer.echo(f"Computed scores for {len(gamma)} queries. Top-1 mean: {float(gamma.mean()):.4f}")


if __name__ == "__main__":
    app()


