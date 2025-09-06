from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HiddenStateExtractor:
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_layers(self, text: str, layer_ids: List[int]) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        hidden_states: Tuple[torch.Tensor, ...] = outputs.hidden_states  # [layer, b, t, d]
        bsz, seq_len, _ = hidden_states[-1].shape[0], hidden_states[-1].shape[1], hidden_states[-1].shape[2]
        assert bsz == 1
        by_layer: Dict[int, torch.Tensor] = {}
        for lid in layer_ids:
            by_layer[lid] = hidden_states[lid][0]  # [t, d]
        return inputs["input_ids"][0], by_layer


