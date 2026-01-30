from __future__ import annotations

from typing import Dict, Tuple


_ALIAS_TO_ID: Dict[str, str] = {
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-8b-v3.1": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma-1b": "google/gemma-3-1b-it",
    "gemma-4b": "google/gemma-3-4b-it",
    "gpt2-medium": "openai-community/gpt2-medium",
    "gpt2": "openai-community/gpt2",
    # Phi aliases
    "phi-mini": "microsoft/Phi-4-mini-reasoning",
    "phi-plus": "microsoft/Phi-4-reasoning-plus",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-math-7b": "Qwen/Qwen2.5-Math-7B-Instruct",
    "qwen-math-1.5b": "Qwen/Qwen2.5-Math-1.5B-Instruct",
}


_FAMILY_TO_PAIR: Dict[str, Tuple[str, str]] = {
    "llama": ("llama-3b", "llama-1b"),
    "llama-old": ("llama-8b-v3.1", "llama-1b"),
    "gemma": ("gemma-4b", "gemma-1b"),
    "gpt": ("gpt2-medium", "gpt2"),
    # Phi family: (ref_model_alias, value_classifier_alias)
    "phi": ("phi-plus", "phi-mini"),
    "qwen": ("qwen-7b", "qwen-1.5b"),
    "qwen-math": ("qwen-math-7b", "qwen-math-1.5b"),
}


def resolve_model_id(name_or_id: str) -> str:
    return _ALIAS_TO_ID.get(name_or_id, name_or_id)


def resolve_family_pair(family: str) -> Tuple[str, str]:
    pair = _FAMILY_TO_PAIR.get(family)
    if pair is None:
        raise KeyError(f"Unknown model family: {family}")
    return pair
