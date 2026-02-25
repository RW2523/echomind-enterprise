"""
Qwen3 0.6B Embeddings: HF + safetensors, GPU, load once at startup.
Uses last_token_pool + L2 normalize for retrieval.
"""
from __future__ import annotations
import logging
from typing import List, Union

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

# Default retrieval instruction (query-side)
DEFAULT_QUERY_INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"


def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


def _query_instruction(task: str, query: str) -> str:
    return f"Instruct: {task}\nQuery: {query}"


class HFEmbedder:
    """Load once at startup; use GPU; inference_mode + optional bf16."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        use_bf16: bool = True,
        low_cpu_mem_usage: bool = True,
        use_safetensors: bool = True,
        max_length: int = 8192,
    ):
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        self._query_instruction = DEFAULT_QUERY_INSTRUCT

        kwargs = {"low_cpu_mem_usage": low_cpu_mem_usage, "trust_remote_code": True}
        if use_safetensors:
            kwargs["use_safetensors"] = True
        dtype = torch.bfloat16 if use_bf16 and device == "cuda" else torch.float32
        kwargs["torch_dtype"] = dtype

        logger.info("Loading embedder tokenizer: %s", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        logger.info("Loading embedder model: %s on %s", model_id, device)
        self.model = AutoModel.from_pretrained(model_id, **kwargs)
        self.model.to(device)
        self.model.eval()
        self._embedding_dim = self.model.config.hidden_size

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def encode(
        self,
        texts: List[str],
        is_query: bool = False,
        batch_size: int = 32,
    ) -> List[List[float]]:
        """Encode texts; if is_query, wrap with instruction. Returns list of vectors (L2-normalized)."""
        if not texts:
            return []
        if is_query:
            texts = [_query_instruction(self._query_instruction, t) for t in texts]
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_dict = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}
            with torch.inference_mode():
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=(self.device == "cuda")):
                    outputs = self.model(**batch_dict)
            emb = _last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.cpu().float().tolist())
        return [vec for batch in all_embeddings for vec in batch]

    def encode_single(self, text: str, is_query: bool = False) -> List[float]:
        out = self.encode([text], is_query=is_query, batch_size=1)
        return out[0] if out else []


# Global singleton (set in lifespan)
embedder: HFEmbedder | None = None


def get_embedder() -> HFEmbedder:
    if embedder is None:
        raise RuntimeError("Embedder not initialized; start app with lifespan first.")
    return embedder
