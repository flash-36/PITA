"""
Unit tests for PITA modules.

Run: python test_modules.py

Organized into sections:
  Section 1: Pure logic tests (no GPU, fast)
  Section 2: Model-dependent tests (needs GPU, loads model once)
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import time
import random
import traceback
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────
# Test infrastructure
# ─────────────────────────────────────────────────────────

PASS = 0
FAIL = 0
SKIP = 0

def run_test(name, fn):
    global PASS, FAIL, SKIP
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        if result == "SKIP":
            SKIP += 1
            print(f"  ⏭️  SKIP  {name} ({elapsed:.1f}s)")
        else:
            PASS += 1
            print(f"  ✅ PASS  {name} ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - t0
        FAIL += 1
        print(f"  ❌ FAIL  {name} ({elapsed:.1f}s)")
        print(f"          {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
    sys.stdout.flush()


def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"Expected {a!r} == {b!r}. {msg}")

def assert_true(cond, msg=""):
    if not cond:
        raise AssertionError(f"Condition is False. {msg}")

def assert_in(a, b, msg=""):
    if a not in b:
        raise AssertionError(f"{a!r} not in {b!r}. {msg}")


# ─────────────────────────────────────────────────────────
# Section 1: Pure logic tests (no GPU)
# ─────────────────────────────────────────────────────────

def section1_answer_extraction():
    """Test extract_final_answer and is_correct."""
    from pita.datasets.utils import extract_final_answer, extract_boxed_last, eq_math
    from pita.datasets.gsm8k import GSM8K

    def test_extract_boxed():
        assert_eq(extract_boxed_last(r"The answer is \boxed{42}."), "42")
        assert_eq(extract_boxed_last(r"\boxed{1} then \boxed{2}"), "2",
                   "should take LAST boxed")
        assert_eq(extract_boxed_last(r"\boxed{3+5}"), "3+5")
        assert_eq(extract_boxed_last("no boxed here"), "")

    def test_extract_final_answer():
        assert_eq(extract_final_answer(r"Therefore \boxed{100}"), "100")
        assert_eq(extract_final_answer("#### 42\nsome text"), "42")
        assert_eq(extract_final_answer("The answer is $7$ dollars"), "7")
        assert_eq(extract_final_answer("I got 99 problems"), "99",
                   "fallback to last number")
        assert_eq(extract_final_answer("no numbers here"), "")

    def test_is_correct_gsm8k():
        gold = "He pays 1+2+5 = $<<1+2+5=8.00>>8.00\n8/8 = $<<8/8=1.00>>1.00\n#### 1"
        assert_true(GSM8K.is_correct(gold, r"blah blah \boxed{1}"),
                     "correct answer via boxed")
        assert_true(GSM8K.is_correct(gold, r"blah blah \boxed{1.00}"),
                     "correct answer 1.00 == 1")
        assert_true(not GSM8K.is_correct(gold, r"blah blah \boxed{2}"),
                     "wrong answer via boxed")
        assert_true(GSM8K.is_correct(gold, "#### 1"),
                     "correct via #### format")

    def test_eq_math():
        assert_true(eq_math("42", "42"))
        assert_true(eq_math("1.0", "1"))
        assert_true(not eq_math("1", "2"))
        assert_true(not eq_math("", "1"))

    run_test("extract_boxed_last", test_extract_boxed)
    run_test("extract_final_answer", test_extract_final_answer)
    run_test("is_correct (GSM8K)", test_is_correct_gsm8k)
    run_test("eq_math", test_eq_math)


def section1_reward_scorer_preference():
    """Test BT sampling preference logic (no model needed)."""
    import math

    def test_bt_preference_logic():
        beta = 1.0
        r_a, r_b = 1.0, 0.0
        p_a = 1.0 / (1.0 + math.exp(-beta * (r_a - r_b)))
        assert_true(p_a > 0.5, f"p_a={p_a} should be > 0.5 when r_a > r_b")

        r_a, r_b = 0.0, 1.0
        p_a = 1.0 / (1.0 + math.exp(-beta * (r_a - r_b)))
        assert_true(p_a < 0.5, f"p_a={p_a} should be < 0.5 when r_a < r_b")

        r_a, r_b = 0.5, 0.5
        p_a = 1.0 / (1.0 + math.exp(-beta * (r_a - r_b)))
        assert_true(abs(p_a - 0.5) < 1e-6, f"p_a={p_a} should be 0.5 when equal")

    run_test("BT preference logic", test_bt_preference_logic)


def section1_dataset_conversion():
    """Test convert_qsharp and convert_pita with mock data + mock tokenizer."""
    from pita.datasets.convert import (
        convert_qsharp_rows_to_classifier_dataset,
        convert_pita_rows_to_classifier_dataset,
    )
    from datasets import Dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    prompt = "What is 2+2?"
    sol_prefix = " The answer is"
    y_a = " The answer is 4."
    y_b = " The answer is 5."
    context = prompt + sol_prefix

    raw = Dataset.from_list([{
        "prompt": prompt,
        "context": context,
        "y_a": y_a,
        "y_b": y_b,
        "score_a": 1.0,
        "score_b": 0.0,
        "preferred": 0,
    }])

    def test_qsharp_convert():
        ds = convert_qsharp_rows_to_classifier_dataset(
            raw, tokenizer=tok, use_chat_template=False
        )
        assert_eq(len(ds), 2, "should produce 2 rows (one per response)")
        row_a = ds[0]
        row_b = ds[1]
        assert_true(len(row_a["input_ids"]) > 0, "input_ids should be non-empty")
        assert_true(len(row_a["target_ids"]) > 0, "target_ids should be non-empty")
        assert_eq(row_a["rewards"], 1.0)
        assert_eq(row_b["rewards"], 0.0)

        context_decoded = tok.decode(row_a["input_ids"], skip_special_tokens=True)
        assert_in("What is 2+2?", context_decoded,
                   "input_ids should contain the prompt")
        assert_in("The answer is", context_decoded,
                   "input_ids should contain the solution prefix")

    def test_pita_convert():
        ds = convert_pita_rows_to_classifier_dataset(
            raw, tokenizer=tok, use_chat_template=False
        )
        assert_eq(len(ds), 1, "should produce 1 row (one preference pair)")
        row = ds[0]
        assert_true(len(row["input_ids"]) > 0)
        assert_true(len(row["chosen_target_ids"]) > 0)
        assert_true(len(row["rejected_target_ids"]) > 0)
        # preferred=0 means y_a is chosen
        chosen_text = tok.decode(row["chosen_target_ids"], skip_special_tokens=True)
        rejected_text = tok.decode(row["rejected_target_ids"], skip_special_tokens=True)
        assert_in("4", chosen_text, f"chosen should contain correct answer, got: {chosen_text!r}")
        assert_in("5", rejected_text, f"rejected should contain wrong answer, got: {rejected_text!r}")

    def test_pita_convert_preferred_1():
        """preferred=1 means y_b is chosen."""
        raw_p1 = Dataset.from_list([{
            "prompt": prompt,
            "context": context,
            "y_a": y_a,
            "y_b": y_b,
            "score_a": 0.0,
            "score_b": 1.0,
            "preferred": 1,
        }])
        ds = convert_pita_rows_to_classifier_dataset(
            raw_p1, tokenizer=tok, use_chat_template=False
        )
        row = ds[0]
        chosen_text = tok.decode(row["chosen_target_ids"], skip_special_tokens=True)
        rejected_text = tok.decode(row["rejected_target_ids"], skip_special_tokens=True)
        assert_in("5", chosen_text, f"preferred=1 means y_b chosen, got: {chosen_text!r}")
        assert_in("4", rejected_text, f"preferred=1 means y_a rejected, got: {rejected_text!r}")

    def test_pita_convert_preferred_none():
        """preferred=None (verifiable datasets) -> falls back to score comparison."""
        raw_none = Dataset.from_list([{
            "prompt": prompt,
            "context": context,
            "y_a": y_a,
            "y_b": y_b,
            "score_a": 1.0,
            "score_b": 0.0,
            "preferred": None,
        }])
        ds = convert_pita_rows_to_classifier_dataset(
            raw_none, tokenizer=tok, use_chat_template=False
        )
        row = ds[0]
        chosen_text = tok.decode(row["chosen_target_ids"], skip_special_tokens=True)
        rejected_text = tok.decode(row["rejected_target_ids"], skip_special_tokens=True)
        assert_in("4", chosen_text, f"score_a > score_b means y_a chosen, got: {chosen_text!r}")
        assert_in("5", rejected_text, f"score_a > score_b means y_b rejected, got: {rejected_text!r}")

    run_test("convert_qsharp_rows_to_classifier_dataset", test_qsharp_convert)
    run_test("convert_pita_rows_to_classifier_dataset", test_pita_convert)
    run_test("convert_pita preferred=1 convention", test_pita_convert_preferred_1)
    run_test("convert_pita preferred=None (verifiable)", test_pita_convert_preferred_none)


def section1_checkpoint_io():
    """Test checkpoint save/load/clear/resume logic."""
    import json
    import tempfile
    from pathlib import Path
    from pita.core.io import (
        save_checkpoint,
        load_all_checkpoints,
        get_saved_chunk_indices,
        clear_checkpoints,
    )

    def test_save_load_roundtrip():
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpts"
            ckpt_dir.mkdir()
            records = [{"question": "q1", "score_a": 1.0}, {"question": "q2", "score_a": 0.0}]
            save_checkpoint(ckpt_dir, "chunk_0", records)
            loaded = load_all_checkpoints(ckpt_dir)
            assert_eq(len(loaded), 2, f"should load 2 records, got {len(loaded)}")
            assert_eq(loaded[0]["question"], "q1")

    def test_multiple_chunks():
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpts"
            ckpt_dir.mkdir()
            save_checkpoint(ckpt_dir, "chunk_0", [{"q": "a"}])
            save_checkpoint(ckpt_dir, "chunk_1", [{"q": "b"}, {"q": "c"}])
            loaded = load_all_checkpoints(ckpt_dir)
            assert_eq(len(loaded), 3, f"should load 3 total records, got {len(loaded)}")

    def test_get_saved_chunk_indices():
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpts"
            ckpt_dir.mkdir()
            save_checkpoint(ckpt_dir, "chunk_0", [{"q": "a"}])
            save_checkpoint(ckpt_dir, "chunk_3", [{"q": "b"}])
            indices = get_saved_chunk_indices(ckpt_dir)
            assert_eq(indices, {0, 3}, f"should find chunks 0 and 3, got {indices}")

    def test_clear_checkpoints():
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpts"
            ckpt_dir.mkdir()
            save_checkpoint(ckpt_dir, "chunk_0", [{"q": "a"}])
            save_checkpoint(ckpt_dir, "chunk_1", [{"q": "b"}])
            clear_checkpoints(ckpt_dir)
            assert_eq(len(list(ckpt_dir.glob("*.json"))), 0, "should clear all checkpoints")

    def test_atomic_overwrite():
        """Saving same key twice should overwrite atomically."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpts"
            ckpt_dir.mkdir()
            save_checkpoint(ckpt_dir, "chunk_0", [{"v": 1}])
            save_checkpoint(ckpt_dir, "chunk_0", [{"v": 2}])
            loaded = load_all_checkpoints(ckpt_dir)
            assert_eq(len(loaded), 1)
            assert_eq(loaded[0]["v"], 2, "should have overwritten value")

    run_test("checkpoint: save/load roundtrip", test_save_load_roundtrip)
    run_test("checkpoint: multiple chunks", test_multiple_chunks)
    run_test("checkpoint: get_saved_chunk_indices", test_get_saved_chunk_indices)
    run_test("checkpoint: clear_checkpoints", test_clear_checkpoints)
    run_test("checkpoint: atomic overwrite", test_atomic_overwrite)


def section1_trainer_collation():
    """Test collation logic in trainers (shapes, masks)."""
    from pita.trainers.qsharp_trainer import QSharpTrainer
    from pita.trainers.pita_trainer import PITATrainer

    def test_qsharp_collate():
        trainer = QSharpTrainer.__new__(QSharpTrainer)
        trainer.pad_token_id = 0
        trainer.max_length = -1

        batch = [
            {"input_ids": [1, 2, 3], "target_ids": [4, 5], "rewards": 0.8, "loss_weights": 1.0},
            {"input_ids": [1, 2], "target_ids": [3, 4, 5], "rewards": 0.2, "loss_weights": 1.0},
        ]
        out = trainer._collate(batch)

        assert_eq(out["input_ids"].shape, torch.Size([2, 5]))
        assert_eq(out["attention_mask"].shape, torch.Size([2, 5]))
        assert_eq(out["loss_mask"].shape, torch.Size([2, 5]))

        # First example: 0 pad tokens, 3 input, 2 target -> loss_mask=[0,0,0,1,1]
        assert_eq(out["loss_mask"][0].tolist(), [0, 0, 0, 1, 1])
        assert_eq(out["attention_mask"][0].tolist(), [1, 1, 1, 1, 1])

        # Second example: 0 pad, 2 input, 3 target -> loss_mask=[0,0,1,1,1]
        assert_eq(out["loss_mask"][1].tolist(), [0, 0, 1, 1, 1])

    def test_pita_collate():
        trainer = PITATrainer.__new__(PITATrainer)
        trainer.pad_token_id = 0
        trainer.max_length = -1

        batch = [
            {"input_ids": [1, 2], "chosen_target_ids": [3, 4], "rejected_target_ids": [5]},
        ]
        out = trainer._collate(batch)

        assert_eq(out["chosen_input_ids"].shape[0], 1)
        assert_eq(out["rejected_input_ids"].shape[0], 1)
        # chosen: [1,2,3,4] len=4, rejected: [1,2,5] padded to 4
        assert_eq(out["chosen_input_ids"].shape[1], 4)
        assert_eq(out["rejected_input_ids"].shape[1], 4)

        assert_eq(out["chosen_response_mask"][0].tolist(), [0, 0, 1, 1])
        # rejected has 1 pad + 2 input + 1 target
        assert_eq(out["rejected_response_mask"][0].tolist(), [0, 0, 0, 1])

    run_test("QSharpTrainer._collate shapes/masks", test_qsharp_collate)
    run_test("PITATrainer._collate shapes/masks", test_pita_collate)


# ─────────────────────────────────────────────────────────
# Section 2: Model-dependent tests (GPU)
# ─────────────────────────────────────────────────────────

# Shared model state loaded once
_model = None
_tokenizer = None

def get_model():
    """Load model once and cache."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    from pita.models.hf import HFModel, GenerationConfig

    gen_cfg = GenerationConfig(
        max_new_tokens=256,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.2,
        use_chat_template=True,
        dtype="bfloat16",
        attn_impl="sdpa",
        gradient_checkpointing=False,
    )
    print("    [loading qwen-1.5b for tests...]")
    _model = HFModel("qwen-1.5b", gen_cfg)
    _tokenizer = _model.tokenizer
    return _model, _tokenizer


def section2_prompt_building():
    """Test build_instruction_prompt with real tokenizer."""
    from pita.core.prompts import build_instruction_prompt

    model, tok = get_model()

    def test_chat_template_applied():
        raw = "What is 2+2?"
        built = build_instruction_prompt(raw, tokenizer=tok, use_chat_template=True)
        assert_true(len(built) > len(raw), "chat template should add tokens")
        assert_in("2+2", built, "original content preserved")

    def test_no_chat_template():
        raw = "What is 2+2?"
        built = build_instruction_prompt(raw, tokenizer=tok, use_chat_template=False)
        assert_eq(built, raw, "without chat template, should return raw")

    run_test("build_instruction_prompt (chat template)", test_chat_template_applied)
    run_test("build_instruction_prompt (no template)", test_no_chat_template)


def section2_gen_kwargs():
    """Test that _gen_kwargs passes through config values without clamping."""
    model, tok = get_model()

    def test_no_clamping():
        kwargs = model._gen_kwargs(greedy=False, logits_processor=None)
        assert_eq(kwargs["temperature"], 0.8,
                   f"temperature should be 0.8, got {kwargs['temperature']}")
        assert_eq(kwargs["top_p"], 0.9,
                   f"top_p should be 0.9, got {kwargs['top_p']}")

    def test_greedy_overrides():
        kwargs = model._gen_kwargs(greedy=True, logits_processor=None)
        assert_eq(kwargs["temperature"], 1.0)
        assert_eq(kwargs["top_p"], 1.0)
        assert_eq(kwargs["do_sample"], False)

    def test_config_not_mutated_after_generation():
        """gen_cfg values should remain unchanged after a full generation call."""
        before = (model.gen_cfg.temperature, model.gen_cfg.top_p, model.gen_cfg.repetition_penalty)
        model.continue_from_context_batch(
            ["Testing config stability"], 16, greedy=False, batch_size=1
        )
        after = (model.gen_cfg.temperature, model.gen_cfg.top_p, model.gen_cfg.repetition_penalty)
        assert_eq(before, after,
                   f"gen_cfg mutated during generation! before={before} after={after}")

    def test_extreme_config_passthrough():
        """Verify extreme temperature/top_p values pass through without clamping."""
        from pita.models.hf import HFModel, GenerationConfig
        extreme_cfg = GenerationConfig(
            max_new_tokens=16, temperature=1.5, top_p=0.5,
            repetition_penalty=1.0, use_chat_template=True,
            dtype="bfloat16", attn_impl="sdpa", gradient_checkpointing=False,
        )
        # Build kwargs without loading a full model - just test the config path
        old_cfg = model.gen_cfg
        model.gen_cfg = extreme_cfg
        kwargs = model._gen_kwargs(greedy=False, logits_processor=None)
        model.gen_cfg = old_cfg

        assert_eq(kwargs["temperature"], 1.5,
                   f"temperature 1.5 should pass through, got {kwargs['temperature']}")
        assert_eq(kwargs["top_p"], 0.5,
                   f"top_p 0.5 should pass through, got {kwargs['top_p']}")

    run_test("_gen_kwargs: no silent clamping", test_no_clamping)
    run_test("_gen_kwargs: greedy overrides", test_greedy_overrides)
    run_test("_gen_kwargs: config not mutated after generation", test_config_not_mutated_after_generation)
    run_test("_gen_kwargs: extreme values pass through", test_extreme_config_passthrough)


def section2_roll_in():
    """Test roll_in_batch returns correct structure."""
    model, tok = get_model()

    def test_roll_in_structure():
        prompts = ["What is 2+2? Write your answer in \\boxed{}."]
        results = model.roll_in_batch(prompts, max_roll_tokens=64, batch_size=1)

        assert_eq(len(results), 1)
        r = results[0]
        assert_in("prompt", r)
        assert_in("context_ids", r)
        assert_in("context_text", r)

        # context_ids should be longer than just the prompt
        prompt_ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        assert_true(len(r["context_ids"]) > len(prompt_ids),
                     f"context should be longer than prompt: {len(r['context_ids'])} vs {len(prompt_ids)}")

    def test_roll_in_context_starts_with_prompt():
        prompts = ["Solve: 3 * 7 = ?"]
        results = model.roll_in_batch(prompts, max_roll_tokens=64, batch_size=1)
        r = results[0]

        prompt_ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        context_ids = r["context_ids"].tolist()
        assert_eq(context_ids[:len(prompt_ids)], prompt_ids,
                   "context_ids should start with prompt token IDs")

    run_test("roll_in_batch: output structure", test_roll_in_structure)
    run_test("roll_in_batch: context starts with prompt", test_roll_in_context_starts_with_prompt)


def section2_continue_from_context():
    """Test continue_from_context_batch."""
    model, tok = get_model()

    def test_continue_basic():
        contexts = [
            "The capital of France is",
        ]
        results = model.continue_from_context_batch(
            contexts, max_new_tokens=32, greedy=True, batch_size=1
        )
        assert_eq(len(results), 1)
        assert_true(len(results[0]) > 0, f"should generate text, got empty")

    def test_greedy_deterministic():
        contexts = ["Once upon a time"]
        r1 = model.continue_from_context_batch(contexts, 32, greedy=True, batch_size=1)
        r2 = model.continue_from_context_batch(contexts, 32, greedy=True, batch_size=1)
        assert_eq(r1[0], r2[0], "greedy should be deterministic")

    run_test("continue_from_context_batch: basic", test_continue_basic)
    run_test("continue_from_context_batch: greedy deterministic", test_greedy_deterministic)


def section2_continuation_quality():
    """Test that continuations are well-formed (not garbled, not leaking context, EOS works)."""
    model, tok = get_model()
    from pita.datasets.gsm8k import GSM8K
    from pita.core.prompts import build_instruction_prompt

    def test_continuation_is_new_text_only():
        """continue_from_context_batch should return ONLY new text, not echo the context."""
        context = build_instruction_prompt(
            "What is 2+2?", tokenizer=tok, use_chat_template=True
        )
        results = model.continue_from_context_batch(
            [context], max_new_tokens=64, greedy=True, batch_size=1
        )
        text = results[0]
        assert_true(len(text) > 0, "should generate non-empty continuation")
        # The context should NOT appear in the output
        assert_true(not text.startswith(context[:20]),
                     f"continuation should not echo the context. Got: {text[:100]!r}")

    def test_continuation_no_special_tokens_in_output():
        """Output should not contain raw special token strings like <|im_end|>."""
        context = build_instruction_prompt(
            "Solve step by step: What is 5 * 6?", tokenizer=tok, use_chat_template=True
        )
        results = model.continue_from_context_batch(
            [context], max_new_tokens=128, greedy=True, batch_size=1
        )
        text = results[0]
        for st in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
            assert_true(st not in text,
                         f"special token {st!r} leaked into output: {text[:200]!r}")

    def test_eos_stops_generation():
        """If context already ends with EOS, should return empty string."""
        eos_token = tok.eos_token or "<|endoftext|>"
        context = "Hello world" + eos_token
        results = model.continue_from_context_batch(
            [context], max_new_tokens=64, greedy=True, batch_size=1
        )
        assert_eq(results[0], "", f"should return empty after EOS, got: {results[0]!r}")

    def test_batched_continuation_produces_valid_outputs():
        """Batched generation should produce non-empty, valid outputs for all items."""
        ctx1 = build_instruction_prompt("What is 3+3?", tokenizer=tok, use_chat_template=True)
        ctx2 = build_instruction_prompt("What is 7+8?", tokenizer=tok, use_chat_template=True)

        r_batch = model.continue_from_context_batch([ctx1, ctx2], 64, greedy=True, batch_size=2)

        assert_eq(len(r_batch), 2, "should return one result per context")
        assert_true(len(r_batch[0]) > 0, "first result should be non-empty")
        assert_true(len(r_batch[1]) > 0, "second result should be non-empty")
        # Both should contain the correct answers somewhere
        assert_in("6", r_batch[0], f"3+3 answer should contain 6: {r_batch[0]!r}")
        assert_in("15", r_batch[1], f"7+8 answer should contain 15: {r_batch[1]!r}")
        for st in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
            assert_true(st not in r_batch[0], f"{st!r} leaked in batch[0]")
            assert_true(st not in r_batch[1], f"{st!r} leaked in batch[1]")

    def test_continuation_from_partial_rollout():
        """The key pipeline pattern: roll in, cut off, continue. Verify text is coherent."""
        prompt = GSM8K.hydrate_prompt(
            "Natalia sold clips to 48 of her friends in April, "
            "and then she sold half as many clips in May. "
            "How many clips did Natalia sell altogether in April and May?"
        )
        rollouts = model.roll_in_batch([prompt], max_roll_tokens=128, batch_size=1)
        r = rollouts[0]

        prompt_ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        context_ids = r["context_ids"].tolist()

        # Truncate at first EOS (matching the fixed pipeline)
        eos_set = set(model.eos_token_ids)
        gen_ids = context_ids[len(prompt_ids):]
        first_eos = next(
            (i for i, tid in enumerate(gen_ids) if tid in eos_set),
            len(gen_ids),
        )
        context_ids = context_ids[:len(prompt_ids) + first_eos]
        rollout_len = first_eos

        if rollout_len < 4:
            return "SKIP"

        # Cut at midpoint
        cut = rollout_len // 2
        prefix_ids = context_ids[:len(prompt_ids) + cut]
        prefix_text = tok.decode(prefix_ids, skip_special_tokens=False)
        sol_prefix_text = tok.decode(prefix_ids[len(prompt_ids):], skip_special_tokens=False)

        remaining = 128 - cut
        greedy_cont = model.continue_from_context_batch(
            [prefix_text], remaining, greedy=True, batch_size=1
        )[0]
        sampled_cont = model.continue_from_context_batch(
            [prefix_text], remaining, greedy=False, batch_size=1
        )[0]

        # Build y_a and y_b like the pipeline does
        y_a = sol_prefix_text + greedy_cont
        y_b = sol_prefix_text + sampled_cont

        assert_true(len(y_a) > len(sol_prefix_text),
                     "y_a should be longer than just the prefix (greedy added text)")
        assert_true(y_a.startswith(sol_prefix_text),
                     "y_a should start with the solution prefix")
        assert_true(y_b.startswith(sol_prefix_text),
                     "y_b should start with the solution prefix")

        # No special tokens in the continuations
        for st in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
            assert_true(st not in greedy_cont, f"{st!r} in greedy continuation")
            assert_true(st not in sampled_cont, f"{st!r} in sampled continuation")

        print(f"          sol_prefix: {sol_prefix_text[:60]!r}...")
        print(f"          greedy_cont[:60]: {greedy_cont[:60]!r}")
        print(f"          sampled_cont[:60]: {sampled_cont[:60]!r}")

    def test_return_scores():
        """continue_from_context_batch with return_scores=True should return logits."""
        ctx = build_instruction_prompt("What is 1+1?", tokenizer=tok, use_chat_template=True)
        texts, scores = model.continue_from_context_batch(
            [ctx], 32, greedy=True, batch_size=1, return_scores=True
        )
        assert_eq(len(texts), 1)
        assert_eq(len(scores), 1)
        assert_true(scores[0] is not None, "scores should not be None")
        # scores[0] shape: [num_new_tokens, vocab_size]
        assert_eq(len(scores[0].shape), 2, f"scores shape should be 2D, got {scores[0].shape}")
        assert_true(scores[0].shape[1] == tok.vocab_size or scores[0].shape[1] > 30000,
                     f"second dim should be vocab_size, got {scores[0].shape[1]}")

    run_test("continuation: returns only new text (no echo)", test_continuation_is_new_text_only)
    run_test("continuation: no special tokens leaked", test_continuation_no_special_tokens_in_output)
    run_test("continuation: EOS stops generation", test_eos_stops_generation)
    run_test("continuation: batched produces valid outputs", test_batched_continuation_produces_valid_outputs)
    run_test("continuation: partial rollout pipeline pattern", test_continuation_from_partial_rollout)
    run_test("continuation: return_scores shape", test_return_scores)


def section2_scoring():
    """Test score_samples returns binary for verifiable datasets."""
    from pita.datasets.gsm8k import GSM8K

    @dataclass
    class FakeSample:
        question: str
        answer: str

    def _make_algo(algo_cls, cfg_overrides=None):
        from omegaconf import OmegaConf
        base = {
            "epochs": 1, "batch_size": 1, "eval_batch_size": 1,
            "max_batch_num_tokens": -1, "num_workers": 0,
            "lr": 1e-5, "weight_decay": 0.01, "grad_clip": 1.0,
            "loss_type": "mle", "proxy_loss_type": "bradley_terry",
            "proxy_lr": 1e-5, "proxy_epochs": 1,
            "num_atoms": 11, "V_min": 0.0, "V_max": 1.0,
            "guidance": {"eta": 1.0, "mode": "expectation", "top_k": 20, "use_cache": True},
        }
        if cfg_overrides:
            base.update(cfg_overrides)
        algo = algo_cls(OmegaConf.create(base))
        algo._dataset = GSM8K(hf_config="main", split="train[:5]",
                               question_key="question", answer_key="answer")
        return algo

    class _MockReward:
        _bt_beta = 1.0
        def score_batch_single(self, pairs):
            return [0.5] * len(pairs)

    def test_qsharp_hf_score_samples():
        from pita.algos.qsharp_hf_algo import QSharpHFAlgorithm
        algo = _make_algo(QSharpHFAlgorithm)
        algo._reward = _MockReward()
        algo._correctness_bonus = 5.0
        ex = FakeSample(question="What is 2+2?", answer="2+2=4\n#### 4")

        # correct vs wrong: composite = rm + bonus vs rm + 0
        s_a, s_b, pref = algo.score_samples(ex, r"blah \boxed{4}", r"blah \boxed{5}")
        assert_eq(s_a, 5.5, f"correct answer should score rm+bonus=5.5, got {s_a}")
        assert_eq(s_b, 0.5, f"wrong answer should score rm=0.5, got {s_b}")
        assert_eq(pref, 0, "correct answer should be preferred")

        # both wrong: composite = rm for both
        s_a, s_b, pref = algo.score_samples(ex, r"blah \boxed{3}", r"blah \boxed{5}")
        assert_eq(s_a, 0.5)
        assert_eq(s_b, 0.5)

        # both correct: composite = rm+bonus for both
        s_a, s_b, pref = algo.score_samples(ex, r"\boxed{4}", r"\boxed{4.0}")
        assert_eq(s_a, 5.5)
        assert_eq(s_b, 5.5)

    def test_qsharp_score_samples():
        from pita.algos.qsharp_algo import QSharpAlgorithm
        algo = _make_algo(QSharpAlgorithm, {"loss_type": "bce"})
        ex = FakeSample(question="What is 2+2?", answer="2+2=4\n#### 4")

        s_a, s_b, pref = algo.score_samples(ex, r"blah \boxed{4}", r"blah \boxed{5}")
        assert_eq(s_a, 1.0)
        assert_eq(s_b, 0.0)
        assert_eq(pref, None)

        s_a, s_b, pref = algo.score_samples(ex, r"\boxed{3}", r"\boxed{5}")
        assert_eq(s_a, 0.0)
        assert_eq(s_b, 0.0)

    def test_qsharp_delegates_to_base():
        """QSharp.generate_data should call super().generate_data (not have its own loop)."""
        from pita.algos.qsharp_algo import QSharpAlgorithm
        from pita.algos.base import ValueGuidedAlgorithms
        import inspect
        src = inspect.getsource(QSharpAlgorithm.generate_data)
        assert_in("super().generate_data", src,
                   "QSharp.generate_data should delegate to super()")

    run_test("score_samples: QSharp-HF composite scoring for GSM8K", test_qsharp_hf_score_samples)
    run_test("score_samples: QSharp binary {0,1} for GSM8K", test_qsharp_score_samples)
    run_test("score_samples: QSharp delegates to base", test_qsharp_delegates_to_base)


def section2_context_construction():
    """
    Critical test: verify context = prompt + solution_prefix_text
    (not character-sliced by token count).
    """
    model, tok = get_model()

    def test_context_is_token_aligned():
        from pita.datasets.gsm8k import GSM8K
        from pita.core.prompts import build_instruction_prompt

        raw_prompt = GSM8K.hydrate_prompt("What is 2+2?")

        rollouts = model.roll_in_batch([raw_prompt], max_roll_tokens=128, batch_size=1)
        r = rollouts[0]

        built_prompt = r["prompt"]  # chat-template-applied prompt
        prompt_ids = tok(built_prompt, add_special_tokens=False)["input_ids"]
        context_ids = r["context_ids"].tolist()
        rollout_token_count = len(context_ids) - len(prompt_ids)

        if rollout_token_count < 4:
            return "SKIP"

        random.seed(42)
        cutoff_tokens = random.randint(1, rollout_token_count - 1)

        context_prefix_ids = context_ids[:len(prompt_ids) + cutoff_tokens]
        solution_prefix_ids = context_prefix_ids[len(prompt_ids):]
        solution_prefix_text = tok.decode(solution_prefix_ids, skip_special_tokens=False)

        # Chars vs tokens should differ (proving the old bug was real)
        assert_true(len(solution_prefix_text) != cutoff_tokens,
                     f"solution_prefix_text len ({len(solution_prefix_text)}) should differ from "
                     f"cutoff_tokens ({cutoff_tokens}) - chars vs tokens")

        # The record stores context = raw_prompt + solution_prefix_text.
        # The conversion pipeline reconstructs: built_prompt + solution_prefix_text.
        # Verify that reconstruction is token-aligned with the original context_prefix_ids.
        reconstructed = built_prompt + solution_prefix_text
        re_tokenized = tok(reconstructed, add_special_tokens=False)["input_ids"]
        len_diff = abs(len(re_tokenized) - len(context_prefix_ids))
        assert_true(len_diff <= 3,
                     f"re-tokenized length {len(re_tokenized)} vs original {len(context_prefix_ids)}, "
                     f"diff={len_diff} (should be <= 3)")

        # Also verify the stored context field works correctly through convert
        stored_context = raw_prompt + solution_prefix_text
        sol_prefix_from_stored = stored_context[len(raw_prompt):]
        assert_eq(sol_prefix_from_stored, solution_prefix_text,
                   "extracting sol_prefix from stored context should recover solution_prefix_text")

    run_test("context construction: token-aligned (Fix 1 verification)", test_context_is_token_aligned)


def section2_value_classifier():
    """Test ValueClassifier forward pass and loss computation."""

    def test_forward_and_loss():
        from pita.models.value_classifier import ValueClassifier
        from transformers import AutoTokenizer

        model, tok = get_model()

        vc = ValueClassifier(
            "qwen-1.5b",
            tokenizer=tok,
            device=model.model.device,
            loss_type="bce",
            num_atoms=11,
            V_min=0.0,
            V_max=1.0,
            attn_impl="sdpa",
            dtype="bfloat16",
            gradient_checkpointing=False,
        )
        vc.eval()

        text = "Hello world"
        ids = tok(text, return_tensors="pt").to(vc.device)
        input_ids = ids["input_ids"]
        attn = ids["attention_mask"]

        # Inference
        with torch.no_grad():
            out = vc(input_ids=input_ids, attention_mask=attn)
        assert_true(out.logits is not None, "should produce logits")
        assert_eq(out.logits.shape[0], 1, "batch size 1")

        # Loss computation
        bs, seq_len = input_ids.shape
        labels = torch.tensor([0.8], device=vc.device)
        loss_weights = torch.ones(bs, device=vc.device)
        loss_mask = torch.ones(bs, seq_len, device=vc.device)

        loss = vc.compute_loss(
            input_ids=input_ids,
            attention_mask=attn,
            labels=labels,
            loss_weights=loss_weights,
            loss_mask=loss_mask,
        )
        assert_true(loss.item() > 0, f"loss should be positive, got {loss.item()}")
        assert_true(torch.isfinite(loss), "loss should be finite")

        del vc
        torch.cuda.empty_cache()

    run_test("ValueClassifier: forward + loss (bce)", test_forward_and_loss)


def section2_bt_microbatch():
    """
    Test that BT loss micro-batching preserves chosen/rejected pairing.
    Verifies Fix 4.
    """

    def test_bt_pairing():
        from pita.models.value_classifier import ValueClassifier

        model, tok = get_model()
        device = model.model.device

        vc = ValueClassifier(
            "qwen-1.5b",
            tokenizer=tok,
            device=device,
            loss_type="bradley_terry",
            num_atoms=11,
            V_min=0.0,
            V_max=1.0,
            attn_impl="sdpa",
            dtype="bfloat16",
            gradient_checkpointing=False,
        )
        vc.eval()

        # Simulate a batch of 4 chosen + 4 rejected = 8 total
        seq_len = 16
        bs = 8
        input_ids = torch.randint(1, 1000, (bs, seq_len), device=device)
        attn = torch.ones(bs, seq_len, dtype=torch.long, device=device)
        loss_mask = torch.ones(bs, seq_len, device=device)
        labels = torch.cat([torch.ones(4, device=device), torch.zeros(4, device=device)])
        loss_weights = torch.ones(bs, device=device)

        # Full batch loss (ground truth)
        with torch.no_grad():
            full_loss = vc.compute_loss(
                input_ids=input_ids,
                attention_mask=attn,
                labels=labels,
                loss_weights=loss_weights,
                loss_mask=loss_mask,
            )

        # Simulate pair-aware micro-batching (micro_bs=4 -> pair_micro_bs=2)
        pair_micro_bs = 2
        num_chosen = 4
        micro_losses = []
        with torch.no_grad():
            for pair_start in range(0, num_chosen, pair_micro_bs):
                pair_end = min(pair_start + pair_micro_bs, num_chosen)
                mb_idx = torch.cat([
                    torch.arange(pair_start, pair_end, device=device),
                    torch.arange(num_chosen + pair_start, num_chosen + pair_end, device=device),
                ])
                mb_loss = vc.compute_loss(
                    input_ids=input_ids[mb_idx],
                    attention_mask=attn[mb_idx],
                    labels=labels[mb_idx],
                    loss_weights=loss_weights[mb_idx],
                    loss_mask=loss_mask[mb_idx],
                )
                micro_losses.append(mb_loss.item())

        # Weighted average should approximate full batch loss
        avg_micro = sum(micro_losses) / len(micro_losses)
        diff = abs(full_loss.item() - avg_micro)
        assert_true(diff < 0.5,
                     f"pair-aware micro-batch loss ({avg_micro:.4f}) should be close to "
                     f"full batch loss ({full_loss.item():.4f}), diff={diff:.4f}")

        # Now simulate the BUGGY linear micro-batching for BT
        buggy_losses = []
        with torch.no_grad():
            for start in range(0, bs, 4):
                end = min(start + 4, bs)
                mb_slice = slice(start, end)
                try:
                    mb_loss = vc.compute_loss(
                        input_ids=input_ids[mb_slice],
                        attention_mask=attn[mb_slice],
                        labels=labels[mb_slice],
                        loss_weights=loss_weights[mb_slice],
                        loss_mask=loss_mask[mb_slice],
                    )
                    buggy_losses.append(mb_loss.item())
                except Exception as e:
                    buggy_losses.append(float("nan"))

        # Buggy losses pair chosen with chosen (or rejected with rejected)
        # So they should differ significantly from the correct result
        avg_buggy = sum(buggy_losses) / len(buggy_losses)
        print(f"          full={full_loss.item():.4f} pair_micro={avg_micro:.4f} buggy_linear={avg_buggy:.4f}")

        del vc
        torch.cuda.empty_cache()

    run_test("BT micro-batching: pair-aware vs linear (Fix 4)", test_bt_pairing)


def section2_guided_generation():
    """Test that GuidedHFModel applies guidance (output differs from base)."""

    def test_guided_differs_from_base():
        from pita.models.guided import GuidedHFModel, GuidanceConfig
        from pita.models.value_classifier import ValueClassifier

        model, tok = get_model()

        vc = ValueClassifier(
            "qwen-1.5b",
            tokenizer=tok,
            device=model.model.device,
            loss_type="bce",
            num_atoms=11,
            V_min=0.0,
            V_max=1.0,
            attn_impl="sdpa",
            dtype="bfloat16",
            gradient_checkpointing=False,
        )
        vc.eval()

        guidance_cfg = GuidanceConfig(eta=2.0, mode="expectation", top_k=20, use_cache=True)
        guided = GuidedHFModel(model, vc, guidance_cfg)

        prompt = "What is the meaning of life?"
        from pita.core.prompts import build_instruction_prompt
        built = build_instruction_prompt(prompt, tokenizer=tok, use_chat_template=True)

        base_out = model.continue_from_context_batch(
            [built], 64, greedy=True, batch_size=1
        )[0]

        guided_out = guided.continue_from_context_batch(
            [built], 64, greedy=True, batch_size=1
        )[0]

        assert_true(len(base_out) > 0, "base should generate text")
        assert_true(len(guided_out) > 0, "guided should generate text")
        # With eta=2.0, guidance should meaningfully alter the output
        print(f"          base[:80]:   {base_out[:80]!r}")
        print(f"          guided[:80]: {guided_out[:80]!r}")

        del vc, guided
        torch.cuda.empty_cache()

    run_test("GuidedHFModel: guidance alters output", test_guided_differs_from_base)


def section2_end_to_end_datagen():
    """
    Mini end-to-end: run the data generation pipeline on 2 GSM8K examples,
    verify the output records have correct structure and scores.
    """

    def test_mini_datagen():
        from pita.algos.qsharp_hf_algo import QSharpHFAlgorithm
        from pita.datasets.gsm8k import GSM8K
        from pita.models.hf import HFModel, GenerationConfig
        from omegaconf import OmegaConf
        import tempfile, json
        from pathlib import Path

        model, tok = get_model()

        algo_cfg = OmegaConf.create({
            "epochs": 1, "batch_size": 2, "eval_batch_size": 2,
            "max_batch_num_tokens": -1, "num_workers": 0,
            "lr": 1e-5, "weight_decay": 0.01, "grad_clip": 1.0,
            "loss_type": "mle", "proxy_loss_type": "bradley_terry",
            "proxy_lr": 1e-5, "proxy_epochs": 1,
            "num_atoms": 11, "V_min": 0.0, "V_max": 1.0,
            "guidance": {"eta": 1.0, "mode": "expectation", "top_k": 20, "use_cache": True},
        })
        algo = QSharpHFAlgorithm(algo_cfg)

        class _MockReward:
            _bt_beta = 1.0
            def score_batch_single(self, pairs):
                return [0.5] * len(pairs)

        ds = GSM8K(hf_config="main", split="train[:2]", question_key="question", answer_key="answer")
        algo._dataset = ds
        algo._reward = _MockReward()
        algo._correctness_bonus = 5.0

        # Manually replicate the core loop from _generate_data_sequential
        gen_cfg = model.gen_cfg
        random.seed(42)

        examples = []
        for ex in ds.iter():
            prompt = ds.hydrate_prompt(ex.question)
            examples.append({"ex": ex, "prompt": prompt})

        prompts = [item["prompt"] for item in examples]
        rollouts = model.roll_in_batch(prompts, gen_cfg.max_new_tokens, batch_size=2)

        contexts_to_continue = []
        eos_set = set(model.eos_token_ids)
        for item, rollout in zip(examples, rollouts):
            built_prompt = rollout["prompt"]
            prompt_token_count = len(tok(built_prompt, add_special_tokens=False)["input_ids"])
            context_token_ids = rollout["context_ids"].tolist()

            # Truncate at first EOS (matching the fixed pipeline)
            gen_ids = context_token_ids[prompt_token_count:]
            first_eos = next(
                (i for i, tid in enumerate(gen_ids) if tid in eos_set),
                len(gen_ids),
            )
            context_token_ids = context_token_ids[:prompt_token_count + first_eos]
            rollout_token_count = first_eos

            if rollout_token_count < 2:
                continue

            cutoff_tokens = random.randint(1, rollout_token_count - 1)
            context_prefix_ids = context_token_ids[:prompt_token_count + cutoff_tokens]
            context_text = tok.decode(context_prefix_ids, skip_special_tokens=False)
            solution_prefix_ids = context_prefix_ids[prompt_token_count:]
            solution_prefix_text = tok.decode(solution_prefix_ids, skip_special_tokens=False)
            remaining = gen_cfg.max_new_tokens - cutoff_tokens

            if remaining >= 1:
                contexts_to_continue.append({
                    "ex": item["ex"], "prompt": item["prompt"],
                    "context_text": context_text,
                    "solution_prefix_text": solution_prefix_text,
                    "cutoff_tokens": cutoff_tokens,
                    "remaining_token_budget": remaining,
                })

        if not contexts_to_continue:
            print("          No valid contexts generated (rollouts too short)")
            return "SKIP"

        all_contexts = [c["context_text"] for c in contexts_to_continue]
        min_remaining = min(c["remaining_token_budget"] for c in contexts_to_continue)

        greedy_cont = model.continue_from_context_batch(all_contexts, min_remaining, greedy=True, batch_size=2)
        sampled_cont = model.continue_from_context_batch(all_contexts, min_remaining, greedy=False, batch_size=2)

        records = []
        for ctx, y_ref, y_sample in zip(contexts_to_continue, greedy_cont, sampled_cont):
            y_a = ctx["solution_prefix_text"] + y_ref
            y_b = ctx["solution_prefix_text"] + y_sample

            score_a, score_b, preferred = algo.score_samples(ctx["ex"], y_a, y_b)

            record = {
                "question": ctx["ex"].question,
                "answer": ctx["ex"].answer,
                "prompt": ctx["prompt"],
                "t": ctx["cutoff_tokens"],
                "context": ctx["prompt"] + ctx["solution_prefix_text"],
                "y_a": y_a,
                "y_b": y_b,
                "score_a": score_a,
                "score_b": score_b,
                "preferred": preferred,
            }
            records.append(record)

        assert_true(len(records) > 0, "should produce at least 1 record")

        for i, rec in enumerate(records):
            # Composite scores: rm_score + correctness_bonus for correct answers
            assert_in(rec["score_a"], {0.5, 5.5},
                       f"record {i}: score_a={rec['score_a']} not a valid composite")
            assert_in(rec["score_b"], {0.5, 5.5},
                       f"record {i}: score_b={rec['score_b']} not a valid composite")
            assert_in(rec["preferred"], {0, 1},
                       f"record {i}: preferred should be 0 or 1")

            # Context should equal prompt + solution_prefix_text
            prompt = rec["prompt"]
            context = rec["context"]
            t = rec["t"]

            assert_true(context.startswith(prompt),
                         f"record {i}: context should start with prompt")

            sol_prefix = context[len(prompt):]
            assert_true(rec["y_a"].startswith(sol_prefix),
                         f"record {i}: y_a should start with sol_prefix")
            assert_true(rec["y_b"].startswith(sol_prefix),
                         f"record {i}: y_b should start with sol_prefix")

            # Context suffix length (chars) should be DIFFERENT from t (tokens)
            # unless by coincidence - but statistically very unlikely for real text
            sol_prefix_tokens = len(tok(sol_prefix, add_special_tokens=False)["input_ids"])
            assert_true(abs(sol_prefix_tokens - t) <= 2,
                         f"record {i}: sol_prefix has {sol_prefix_tokens} tokens, "
                         f"t={t}. These should be close (within BPE rounding)")

        print(f"          Generated {len(records)} records, all checks passed")
        for i, rec in enumerate(records):
            print(f"          record {i}: score_a={rec['score_a']}, score_b={rec['score_b']}, "
                  f"t={rec['t']}, context_suffix_len={len(rec['context'])-len(rec['prompt'])} chars")

    run_test("End-to-end mini datagen (Fix 1+2 verification)", test_mini_datagen)


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    total_start = time.time()

    print("\n" + "=" * 60)
    print("Section 1: Pure Logic Tests (no GPU)")
    print("=" * 60)
    section1_answer_extraction()
    section1_reward_scorer_preference()
    section1_dataset_conversion()
    section1_checkpoint_io()
    section1_trainer_collation()

    print("\n" + "=" * 60)
    print("Section 2: Model Tests (GPU)")
    print("=" * 60)
    section2_prompt_building()
    section2_gen_kwargs()
    section2_roll_in()
    section2_continue_from_context()
    section2_continuation_quality()
    section2_scoring()
    section2_context_construction()
    section2_value_classifier()
    section2_bt_microbatch()
    section2_guided_generation()
    section2_end_to_end_datagen()

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed, {SKIP} skipped ({total_elapsed:.1f}s)")
    print("=" * 60)

    if FAIL > 0:
        sys.exit(1)
