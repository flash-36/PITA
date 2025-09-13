from __future__ import annotations

from typing import Dict, Tuple

# Map friendly aliases to HF model ids; fall back if unknown
_ALIAS_TO_ID: Dict[str, str] = {
    "llama-1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama-7b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "gemma-2b": "google/gemma-2-2b-it",
    "gemma-9b": "google/gemma-2-9b-it",
}

# Families map to (ref, cls) concrete aliases
_FAMILY_TO_PAIR: Dict[str, Tuple[str, str]] = {
    "llama": ("llama-7b", "llama-1b"),
    "gemma": ("gemma-9b", "gemma-2b"),
}


def resolve_model_id(name_or_id: str) -> str:
    return _ALIAS_TO_ID.get(name_or_id, name_or_id)


def resolve_family_pair(family: str) -> Tuple[str, str]:
    pair = _FAMILY_TO_PAIR.get(family)
    if pair is None:
        raise KeyError(f"Unknown model family: {family}")
    return pair
