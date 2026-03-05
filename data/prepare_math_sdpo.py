#!/usr/bin/env python3
"""
Prepare mathematical datasets for SDPO training and evaluation.

Training:  DAPO-Math-17k (BytedTsinghua-SIA/DAPO-Math-17k)
           OR OpenThoughts Math 30k OPSD (siyanzhao/Openthoughts_math_30k_opsd)
Eval:      MATH-500, AIME 2024, AIME 2025, GSM8k

Produces the directory structure expected by the SDPO pipeline:
    datasets/math_sdpo/
        train.parquet          # DAPO-Math-17k
        test.parquet           # concatenation of all eval sets
        train_example.json     # single example for sanity-checking
        test_example.json

Each row follows the standard schema:
    prompt, data_source, ability, reward_model, extra_info
"""

import argparse
import json
import os
import re
import sys

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MATH_PROMPT = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

# Map data_source values to the keys used by the feedback reward dispatcher
# (verl/utils/reward_score/feedback/__init__.py dispatches on data_source)
# "math", "math500", "dapo_math", "gsm8k" all go to math.compute_score

DATASETS_CONFIG = {
    "dapo_math": {
        "hf_path": "BytedTsinghua-SIA/DAPO-Math-17k",
        "split": "train",
        "problem_key": "prompt",  # DAPO already has chat-formatted prompts
        "answer_key": "reward_model",  # nested: reward_model.ground_truth
        "is_preformatted": True,
    },
    "openthoughts_math_30k_opsd": {
        "hf_path": "siyanzhao/Openthoughts_math_30k_opsd",
        "split": "train",
        "problem_key": "problem",
        "answer_key": "Answer",
        "source_key": "source",
    },
    "math500": {
        "hf_path": "math-ai/math500",
        "split": "test",
        "problem_key": "problem",
        "answer_key": "answer",
    },
    "aime24": {
        "hf_path": "math-ai/aime24",
        "split": "test",
        "problem_key": "problem",
        "answer_key": "solution",
        "answer_transform": "extract_boxed",
    },
    "aime25": {
        "hf_path": "math-ai/aime25",
        "split": "test",
        "problem_key": "problem",
        "answer_key": "answer",
    },
    "gsm8k": {
        "hf_path": "openai/gsm8k",
        "hf_config": "main",
        "split": "test",
        "problem_key": "question",
        "answer_key": "answer",
        "answer_transform": "extract_gsm8k",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str:
    """Remove \\boxed{} wrapper."""
    if text.startswith("\\boxed{") and text.endswith("}"):
        return text[7:-1]
    return text


def extract_gsm8k_answer(text: str) -> str:
    """Extract numeric answer after ####."""
    m = re.search(r"####\s*([0-9,]+(?:\.[0-9]+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    return ""


def _to_large(field: pa.Field) -> pa.Field:
    t = field.type
    if pa.types.is_string(t):
        return pa.field(field.name, pa.large_string(), field.nullable, field.metadata)
    if pa.types.is_binary(t):
        return pa.field(field.name, pa.large_binary(), field.nullable, field.metadata)
    if pa.types.is_list(t):
        return pa.field(
            field.name,
            pa.large_list(_to_large(pa.field("item", t.value_type)).type),
            field.nullable,
            field.metadata,
        )
    if pa.types.is_struct(t):
        return pa.field(
            field.name,
            pa.struct(
                [_to_large(pa.field(f.name, f.type, f.nullable, f.metadata)) for f in t]
            ),
            field.nullable,
            field.metadata,
        )
    return field


def _large_schema(schema: pa.Schema) -> pa.Schema:
    return pa.schema(
        [_to_large(pa.field(f.name, f.type, f.nullable, f.metadata)) for f in schema]
    )


def write_rowgrouped_large(ds, path: str, rows_per_group: int = 32):
    """Write parquet with LargeString types to avoid 32-bit overflow."""
    tbl: pa.Table = ds.data.table
    tbl = tbl.cast(_large_schema(tbl.schema))
    n = len(tbl)
    writer = None
    try:
        for start in range(0, n, rows_per_group):
            chunk = tbl.slice(start, min(rows_per_group, n - start))
            if writer is None:
                writer = pq.ParquetWriter(path, chunk.schema, compression="zstd")
            writer.write_table(chunk)
    finally:
        if writer is not None:
            writer.close()


# ---------------------------------------------------------------------------
# Dataset loading & formatting
# ---------------------------------------------------------------------------

def load_and_format_dapo(split_label: str = "train", num_samples: int = None) -> datasets.Dataset:
    """Load DAPO-Math-17k and format to the expected schema.

    DAPO already ships in the verl-compatible format (prompt as chat messages,
    reward_model dict, extra_info dict). We just normalise data_source and
    align the output formatting template with MATH_PROMPT.
    """
    cfg = DATASETS_CONFIG["dapo_math"]
    print(f"Loading {cfg['hf_path']} ...")
    ds = datasets.load_dataset(cfg["hf_path"], split=cfg["split"])
    print(f"  Loaded {len(ds)} examples")

    if num_samples is not None and num_samples < len(ds):
        print(f"  Sampling {num_samples} examples...")
        ds = ds.shuffle(seed=42).select(range(num_samples))

    # Prefix and suffix used in DAPO to specify its "Answer: $Answer" template
    dapo_prefix = (
        "Solve the following math problem step by step. "
        "The last line of your response should be of the form Answer: $Answer (without quotes) "
        "where $Answer is the answer to the problem.\n\n"
    )
    dapo_suffix = '\n\nRemember to put your answer on its own line after "Answer:".'

    def process_fn(example, idx):
        # DAPO already has: prompt, data_source, ability, reward_model, extra_info
        # Strip DAPO's template and apply our MATH_PROMPT
        raw_content = example["prompt"][-1]["content"]

        if raw_content.startswith(dapo_prefix):
            raw_content = raw_content[len(dapo_prefix) :]
        if raw_content.endswith(dapo_suffix):
            raw_content = raw_content[: -len(dapo_suffix)]

        formatted_content = MATH_PROMPT.format(problem=raw_content.strip())
        example["prompt"][-1]["content"] = formatted_content

        # Override data_source to match our reward dispatcher
        example["data_source"] = "dapo_math"
        # Ensure extra_info has split
        if "extra_info" not in example or example["extra_info"] is None:
            example["extra_info"] = {}
        example["extra_info"]["split"] = split_label
        example["extra_info"]["index"] = str(idx)
        example["extra_info"]["problem"] = raw_content.strip()
        return example

    ds = ds.map(process_fn, with_indices=True)
    return ds


def load_and_format_openthoughts_math_30k_opsd(
    split_label: str = "train", num_samples: int = None
) -> datasets.Dataset:
    """Load Openthoughts_math_30k_opsd and format to the expected schema.

    This dataset includes a dedicated short `Answer` field, which is directly
    compatible with the math reward function's ground-truth expectation.
    """
    cfg = DATASETS_CONFIG["openthoughts_math_30k_opsd"]
    print(f"Loading {cfg['hf_path']} ...")
    ds = datasets.load_dataset(cfg["hf_path"], split=cfg["split"])
    print(f"  Loaded {len(ds)} examples")

    ds = ds.filter(
        lambda ex: bool(str(ex.get(cfg["problem_key"], "")).strip())
        and bool(str(ex.get(cfg["answer_key"], "")).strip())
    )
    print(f"  Kept {len(ds)} examples with non-empty problem/Answer")

    if num_samples is not None and num_samples < len(ds):
        print(f"  Sampling {num_samples} examples...")
        ds = ds.shuffle(seed=42).select(range(num_samples))

    def process_fn(example, idx):
        problem = str(example[cfg["problem_key"]]).strip()
        answer = str(example[cfg["answer_key"]]).strip()

        # Normalize if a boxed answer appears in the Answer field.
        answer = extract_boxed(answer)

        out = {
            "data_source": "math",
            "prompt": [{"role": "user", "content": MATH_PROMPT.format(problem=problem)}],
            "ability": "math",
            "reward_model": {"style": "math", "ground_truth": answer},
            "extra_info": {
                "split": split_label,
                "index": str(idx),
                "description": problem,
                "problem": problem,
                "source": example.get(cfg.get("source_key", "source"), "openthoughts_math_30k_opsd"),
            },
        }
        if "solution" in example:
             out["reward_model"]["reference"] = example["solution"]
        return out

    ds = ds.map(process_fn, with_indices=True, remove_columns=ds.column_names)
    return ds


def load_and_format_eval(dataset_key: str, split_label: str = "test") -> datasets.Dataset:
    """Load an evaluation dataset and format it to the standard schema."""
    cfg = DATASETS_CONFIG[dataset_key]
    print(f"Loading {cfg['hf_path']} ...")

    load_kwargs = {"split": cfg["split"]}
    if "hf_config" in cfg:
        load_kwargs["name"] = cfg["hf_config"]
    ds = datasets.load_dataset(cfg["hf_path"], **load_kwargs)
    print(f"  Loaded {len(ds)} examples")

    def process_fn(example, idx):
        problem = example[cfg["problem_key"]]
        answer = str(example[cfg["answer_key"]])

        # Apply answer transformations
        transform = cfg.get("answer_transform")
        if transform == "extract_boxed":
            answer = extract_boxed(answer)
        elif transform == "extract_gsm8k":
            answer = extract_gsm8k_answer(answer)

        prompt_text = MATH_PROMPT.format(problem=problem)

        return {
            "data_source": dataset_key,  # e.g. "math500", "aime24", "gsm8k"
            "prompt": [{"role": "user", "content": prompt_text}],
            "ability": "math",
            "reward_model": {"style": "math", "ground_truth": answer},
            "extra_info": {
                "split": split_label,
                "index": str(idx),
                "description": problem,
                "problem": problem,
            },
        }

    ds = ds.map(process_fn, with_indices=True, remove_columns=ds.column_names)

    # if dataset is gsm8k, create a subset of 500 samples
    if "gsm8k" in dataset_key and len(ds) > 500:
        print(f"  Sampling 500 examples from GSM8k for evaluation...")
        ds = ds.shuffle(seed=42).select(range(500))

    return ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare math datasets for SDPO")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/math_sdpo",
        help="Output directory for the preprocessed data",
    )
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        default=["math500", "aime24", "aime25", "gsm8k"],
        help="Which evaluation datasets to include",
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=None,
        help="Number of training samples to include (default: all)",
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="dapo_math",
        choices=["dapo_math", "openthoughts_math_30k_opsd"],
        help="Training dataset to use: dapo_math or openthoughts_math_30k_opsd",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ---- Train ----
    print("\n" + "=" * 60)
    print(f"Preparing TRAINING data ({args.train_dataset})")
    print("=" * 60)

    if args.train_dataset == "dapo_math":
        train_ds = load_and_format_dapo(split_label="train", num_samples=args.train_samples)
        train_label = "DAPO-Math-17k"
    elif args.train_dataset == "openthoughts_math_30k_opsd":
        train_ds = load_and_format_openthoughts_math_30k_opsd(
            split_label="train", num_samples=args.train_samples
        )
        train_label = "Openthoughts_math_30k_opsd"
    else:
        raise ValueError(f"Unsupported train dataset: {args.train_dataset}")

    print(f"Train dataset: {len(train_ds)} examples")
    print(f"  Columns: {train_ds.column_names}")

    # ---- Eval: MATH-500 + AIME + GSM8k ----
    print("\n" + "=" * 60)
    print("Preparing EVALUATION data")
    print("=" * 60)
    eval_datasets = []
    for ds_key in args.eval_datasets:
        eval_ds = load_and_format_eval(ds_key, split_label="test")
        print(f"  {ds_key}: {len(eval_ds)} examples")
        eval_datasets.append(eval_ds)

    test_ds = datasets.concatenate_datasets(eval_datasets)
    print(f"Combined test dataset: {len(test_ds)} examples")

    # ---- Write parquet files ----
    print("\n" + "=" * 60)
    print("Writing parquet files")
    print("=" * 60)

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    write_rowgrouped_large(train_ds, train_path)
    print(f"  Written: {train_path}")

    write_rowgrouped_large(test_ds, test_path)
    print(f"  Written: {test_path}")

    # ---- Write example JSONs for sanity checking ----
    train_example = train_ds[0]
    with open(os.path.join(output_dir, "train_example.json"), "w") as f:
        json.dump(train_example, f, indent=2, default=str)

    test_example = test_ds[0]
    with open(os.path.join(output_dir, "test_example.json"), "w") as f:
        json.dump(test_example, f, indent=2, default=str)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Output directory: {output_dir}")
    print(f"  Train: {len(train_ds)} examples ({train_label})")
    print(f"  Test:  {len(test_ds)} examples ({', '.join(args.eval_datasets)})")
    # Breakdown by data_source
    for ds_key in args.eval_datasets:
        count = sum(1 for ex in test_ds if ex["data_source"] == ds_key)
        print(f"    - {ds_key}: {count}")
    print(f"\nFiles created:")
    print(f"  {train_path}")
    print(f"  {test_path}")
    print(f"  {os.path.join(output_dir, 'train_example.json')}")
    print(f"  {os.path.join(output_dir, 'test_example.json')}")


if __name__ == "__main__":
    main()
