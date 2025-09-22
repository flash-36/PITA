from __future__ import annotations

from typing import List

from transformers import PreTrainedTokenizerBase


def build_instruction_prompt(
    instruction: str,
    tokenizer: PreTrainedTokenizerBase,
    use_chat_template: bool,
) -> str:
    if (
        use_chat_template
        and hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None)
    ):
        messages = [{"role": "user", "content": instruction}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return instruction


def build_reward_model_prompt(
    question: str,
    y_a: str,
    y_b: str,
    tokenizer: PreTrainedTokenizerBase,
) -> List[str]:
    msgs_a = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": y_a},
    ]
    msgs_b = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": y_b},
    ]
    if hasattr(tokenizer, "apply_chat_template") and getattr(
        tokenizer, "chat_template", None
    ):
        return [
            tokenizer.apply_chat_template(
                msgs_a, tokenize=False, add_generation_prompt=False
            ),
            tokenizer.apply_chat_template(
                msgs_b, tokenize=False, add_generation_prompt=False
            ),
        ]
    return [
        f"User: {question}\nAssistant: {y_a}",
        f"User: {question}\nAssistant: {y_b}",
    ]
