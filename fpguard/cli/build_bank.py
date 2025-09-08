import json
import os
from typing import Optional

import numpy as np  # type: ignore
import typer  # type: ignore

from fpguard.bank import FingerprintBank, FingerprintMeta


app = typer.Typer(add_completion=False)


@app.command()
def main(
    vectors_path: str = typer.Argument(..., help="Path to .npy array of shape [N, D]"),
    output_dir: str = typer.Argument(..., help="Directory to save FAISS index and metadata"),
    pq_m: int = typer.Option(16, help="PQ M (subquantizers) for IVFPQ"),
    use_hnsw: bool = typer.Option(True, help="Use HNSWFlat instead of IVFPQ"),
    batch_size: int = typer.Option(100000, help="Add vectors in batches to reduce memory spikes"),
    meta_path: Optional[str] = typer.Option(None, help="Optional JSONL metas per vector"),
):
    """Build a fingerprint FAISS bank from precomputed vectors (.npy) and save artifacts."""
    os.makedirs(output_dir, exist_ok=True)

    typer.echo(f"Loading vectors from {vectors_path} ...")
    vecs = np.load(vectors_path)
    assert vecs.ndim == 2 and vecs.dtype == np.float32, "vectors must be float32 [N, D]"
    n, d = vecs.shape

    bank = FingerprintBank(dim=int(d), pq_m=int(pq_m), use_hnsw=bool(use_hnsw))

    metas = []
    if meta_path is not None and os.path.exists(meta_path):
        typer.echo(f"Loading metas from {meta_path} ...")
        with open(meta_path, "r") as f:
            for line_idx, line in enumerate(f):
                record = json.loads(line)
                metas.append(
                    FingerprintMeta(
                        seq_id=str(record.get("seq_id", str(line_idx))),
                        token_idx=int(record.get("token_idx", 0)),
                        layer_id=int(record.get("layer_id", 0)),
                        node_id=int(record.get("node_id", line_idx)),
                    )
                )
        if len(metas) != n:
            typer.echo("Meta count does not match vectors; regenerating default metas")
            metas = []

    if not metas:
        metas = [FingerprintMeta(seq_id=str(i), token_idx=0, layer_id=0, node_id=i) for i in range(n)]

    typer.echo("Training (if needed)...")
    if not use_hnsw:
        # For IVFPQ, train on all or subset
        train_sample = vecs if n <= 500000 else vecs[np.random.choice(n, 500000, replace=False)]
        bank.train(train_sample)

    typer.echo("Adding vectors to index...")
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bank.add(vecs[start:end], metas[start:end])
        typer.echo(f"Added {end} / {n}")

    # Save FAISS index and metas
    index_path = os.path.join(output_dir, "bank.faiss")
    meta_out = os.path.join(output_dir, "metas.jsonl")
    typer.echo(f"Saving index to {index_path}")
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError("faiss is required to save the index") from e

    faiss.write_index(bank.index, index_path)
    with open(meta_out, "w") as f:
        for m in bank.metas:
            f.write(json.dumps({"seq_id": m.seq_id, "token_idx": m.token_idx, "layer_id": m.layer_id, "node_id": m.node_id}) + "\n")

    # Save config
    cfg = {"dim": int(d), "pq_m": int(pq_m), "use_hnsw": bool(use_hnsw), "num_vectors": int(n)}
    with open(os.path.join(output_dir, "bank.json"), "w") as f:
        json.dump(cfg, f)

    typer.echo("Done.")


if __name__ == "__main__":
    app()


