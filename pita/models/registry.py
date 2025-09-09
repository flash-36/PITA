from __future__ import annotations

from typing import Dict

# Map friendly aliases to HF model ids; fall back if unknown
_ALIAS_TO_ID: Dict[str, str] = {
    "llama-1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama-7b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "gpt-oss-60b": "Qwen/Qwen2-7B-Instruct",  # lol TODO: change this to the actual model id
}


def resolve_model_id(name_or_id: str) -> str:
    return _ALIAS_TO_ID.get(name_or_id, name_or_id)
