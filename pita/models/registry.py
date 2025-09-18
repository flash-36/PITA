from __future__ import annotations

from typing import Dict, Tuple

# Map friendly aliases to HF model ids; fall back if unknown
_ALIAS_TO_ID: Dict[str, str] = {
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma-1b": "google/gemma-3-1b-it",
    "gemma-4b": "google/gemma-3-4b-it",
    "gpt2-medium": "openai-community/gpt2-medium",
    "gpt2": "openai-community/gpt2",
}

# Families map to (ref, cls) concrete aliases
_FAMILY_TO_PAIR: Dict[str, Tuple[str, str]] = {
    "llama": ("llama-3b", "llama-1b"),
    "gemma": ("gemma-4b", "gemma-1b"),
    "gpt": ("gpt2-medium", "gpt2"),
}


def resolve_model_id(name_or_id: str) -> str:
    return _ALIAS_TO_ID.get(name_or_id, name_or_id)


def resolve_family_pair(family: str) -> Tuple[str, str]:
    pair = _FAMILY_TO_PAIR.get(family)
    if pair is None:
        raise KeyError(f"Unknown model family: {family}")
    return pair
