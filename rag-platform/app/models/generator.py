"""
Qwen-14B (Qwen2.5-14B-Instruct) generator: HF + safetensors, GPU, load once at startup.
"""
from __future__ import annotations
import logging
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class HFGenerator:
    """Load once at startup; use GPU; inference_mode + optional bf16."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        use_bf16: bool = True,
        low_cpu_mem_usage: bool = True,
        use_safetensors: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ):
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        kwargs = {"low_cpu_mem_usage": low_cpu_mem_usage, "trust_remote_code": True}
        if use_safetensors:
            kwargs["use_safetensors"] = True
        dtype = torch.bfloat16 if use_bf16 and device == "cuda" else torch.float32
        kwargs["torch_dtype"] = dtype

        logger.info("Loading generator tokenizer: %s", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        logger.info("Loading generator model: %s on %s", model_id, device)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.to(device)
        self.model.eval()

    def generate(
        self,
        messages: List[dict],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Chat-style generation from messages [{"role":"user"/"assistant"/"system","content":"..."}]."""
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature if temperature is not None else self.temperature
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=(self.device == "cuda")):
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        response = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response.strip()


# Global singleton (set in lifespan)
generator: HFGenerator | None = None


def get_generator() -> HFGenerator:
    if generator is None:
        raise RuntimeError("Generator not initialized; start app with lifespan first.")
    return generator
