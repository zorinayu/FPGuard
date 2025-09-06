from pydantic import BaseModel, Field
from typing import List


class FPGuardConfig(BaseModel):
    model_name: str = Field(default="meta-llama/Llama-2-7b-hf")
    layers: List[int] = Field(default_factory=lambda: [4, 8, 12, 16])
    projection_dim: int = 128
    temperature: float = 0.07
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    dropout: float = 0.1
    mask_ratio: float = 0.15
    synonym_prob: float = 0.1
    top_k: int = 32
    cosine_alpha: float = 0.7
    base_threshold: float = 0.75
    dynamic_beta: float = 0.2
    pq_m: int = 16
    pq_bits: int = 8


