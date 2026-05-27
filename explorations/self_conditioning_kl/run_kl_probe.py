#!/usr/bin/env python3
"""Probe token-level KL signals for SDPO-style self-conditioned teachers.

This is intentionally standalone with respect to training: it loads raw jsonl
rows, reuses the repository's feedback-producing reward modules, and writes
artifacts that are easy to inspect in pandas, a notebook, or a spreadsheet.
"""

import argparse
import csv
import importlib.util
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from visualizations import plot_traces, write_token_heatmap_report

KLDirections = Literal["teacher||base", "base||teacher", "js"]

@dataclass
class TokenKL:
    sample_id: int
    dataset_idx: str
    comparison: str
    target_name: str
    token_index: int
    token_id: int
    token_text: str
    kl: float
    base_logp_token: float
    teacher_logp_token: float
    logp_gap_teacher_minus_base: float


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def infer_split(dataset_path: Path, explicit_split: str | None) -> str:
    if explicit_split:
        return explicit_split
    if dataset_path.stem in {"train", "test", "val", "validation"}:
        return "test" if dataset_path.stem in {"test", "val", "validation"} else "train"
    return "test"


def normalize_data_source(row: dict[str, Any]) -> str:
    """Map raw rows to the feedback dispatcher's supported data_source keys."""
    data_source = row.get("dataset") or row.get("data_source") or row.get("kind")
    kind = row.get("kind") or row.get("ability")
    if data_source == "tooluse" or kind == "tooluse":
        return "tooluse"
    if data_source in {"code", "livecodebench", "humanevalplus"}:
        return data_source
    if kind == "code":
        return "code"
    raise ValueError(f"Unsupported row for feedback scoring: data_source={data_source!r}, kind={kind!r}")


def ground_truth_for_row(row: dict[str, Any], data_source: str) -> str:
    if data_source in {"code", "livecodebench", "humanevalplus"}:
        ground_truth = row.get("tests", row.get("answer"))
    else:
        ground_truth = row.get("answer")
    return ground_truth if isinstance(ground_truth, str) else json.dumps(ground_truth)


def extra_info_for_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    extra_info = dict(row.get("extra_info") or {})
    extra_info.setdefault("split", split)
    extra_info.setdefault("index", str(row.get("idx", "")))
    extra_info.setdefault("description", row.get("description", ""))
    extra_info.setdefault("problem", row.get("prompt", ""))
    extra_info.setdefault("elo", row.get("elo", None))
    extra_info.setdefault("achievement_prior", row.get("achievement_prior", 0))
    return extra_info


def score_response(
    *,
    data_source: str,
    solution: str,
    ground_truth: str,
    extra_info: dict[str, Any],
    max_code_test_cases: int | None,
) -> dict[str, Any]:
    """Use the same feedback-producing reward modules used by SDPO."""
    if data_source == "tooluse":
        from verl.utils.reward_score.feedback.tooluse import compute_score
        
        result = compute_score(solution, ground_truth)
    elif data_source in {"code", "livecodebench", "humanevalplus"}:
        from verl.utils.reward_score.feedback.code import compute_score
        
        result = compute_score(
            solution,
            ground_truth,
            extra_info,
            sparse_rewards=True,
            max_test_cases=max_code_test_cases,
        )
    else:
        raise ValueError(f"Unsupported feedback data_source: {data_source}")
    return dict(result)



def feedback_text(score: dict[str, Any]) -> str:
    feedback = str(score.get("feedback") or "").strip()
    if feedback:
        return feedback
    if float(score.get("score", 0.0)) >= 1.0:
        return "The environment judged the previous answer correct."
    return "The environment judged the previous answer incorrect, but did not provide details."


def base_messages(problem: str, system: str | None = None) -> list[dict[str, str]]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": problem})
    return messages


def feedback_only_messages(problem: str, feedback: str, system: str | None = None) -> list[dict[str, str]]:
    # There is no prior assistant turn for x,f, so keep this as one user
    # message rather than creating consecutive user turns.
    prompt = (
        f"{problem}\n\n"
        "Privileged environment feedback:\n"
        f"{feedback}\n\n"
        "Correctly answer the original question. Preserve the requested output format."
    )
    return base_messages(prompt, system)


def attempt_feedback_messages(
    problem: str,
    attempt: str,
    feedback: str,
    system: str | None = None,
) -> list[dict[str, str]]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(
        [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": attempt},
            {"role": "user", "content": environment_feedback_content(feedback)},
        ]
    )
    return messages


def environment_feedback_content(feedback: str) -> str:
    # Verl's interaction loop appends environment/interactor responses as a
    # user message. The "tool" role is reserved for actual tool-call outputs.
    return (
        "Environment feedback on your previous attempt:\n"
        f"{feedback}\n\n"
        "Try again and correctly answer the original question. Preserve the requested output format."
    )


def apply_chat_template(tokenizer, messages: list[dict[str, str]], enable_thinking: bool) -> torch.Tensor:
    kwargs = dict(
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            pass
    except Exception:
        pass

    # Fallback for base models or tokenizers without a chat template.
    text = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages) + "\n\nASSISTANT:\n"
    return tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"]


def maybe_truncate_context(input_ids: torch.Tensor, max_context_tokens: int | None) -> torch.Tensor:
    if max_context_tokens is not None and input_ids.shape[-1] > max_context_tokens:
        return input_ids[:, -max_context_tokens:]
    return input_ids


def decode_response(tokenizer, token_ids: torch.Tensor) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def trim_generated_ids(tokenizer, response_ids: torch.Tensor) -> torch.Tensor:
    ids = response_ids.detach().cpu()
    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        ids = ids[ids != pad_id]
    return ids


def generate_response(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    *,
    max_context_tokens: int | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
) -> tuple[str, list[int], list[int]]:
    input_ids = apply_chat_template(tokenizer, messages, enable_thinking=enable_thinking)
    input_ids = maybe_truncate_context(input_ids, max_context_tokens).to(model_input_device(model))
    attention_mask = torch.ones_like(input_ids)

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generate_kwargs.update({"temperature": temperature, "top_p": top_p})

    with torch.inference_mode():
        generation = model.generate(**generate_kwargs)
    response_ids = trim_generated_ids(tokenizer, generation[0, input_ids.shape[-1] :])
    return decode_response(tokenizer, response_ids), response_ids.tolist(), input_ids[0].detach().cpu().tolist()


def model_input_device(model):
    return model.get_input_embeddings().weight.device


def logprobs_for_target(model, prompt_ids: list[int], target_ids: list[int]) -> torch.Tensor:
    """Return log p(next token | prompt + target prefix) for every target position."""
    if not target_ids:
        return torch.empty(0, model.config.vocab_size)

    device = model_input_device(model)
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    target = torch.tensor(target_ids, dtype=torch.long, device=device)
    input_ids = torch.cat([prompt, target[:-1]], dim=0).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]

    start = len(prompt_ids) - 1
    target_logits = logits[start : start + len(target_ids)].float()
    return torch.log_softmax(target_logits, dim=-1)


def token_kl_values(
    base_logps: torch.Tensor,
    teacher_logps: torch.Tensor,
    direction: KLDirections,
) -> torch.Tensor:
    if direction == "teacher||base":
        return (teacher_logps.exp() * (teacher_logps - base_logps)).sum(dim=-1)
    if direction == "base||teacher":
        return (base_logps.exp() * (base_logps - teacher_logps)).sum(dim=-1)
    if direction == "js":
        base_p = base_logps.exp()
        teacher_p = teacher_logps.exp()
        mixture = 0.5 * (base_p + teacher_p)
        mixture_logps = torch.log(mixture.clamp_min(1e-45))
        return 0.5 * (base_p * (base_logps - mixture_logps)).sum(dim=-1) + 0.5 * (
            teacher_p * (teacher_logps - mixture_logps)
        ).sum(dim=-1)
    raise ValueError(f"Unknown KL direction: {direction}")


def compute_token_trace(
    *,
    model,
    tokenizer,
    sample_id: int,
    dataset_idx: str,
    comparison: str,
    target_name: str,
    base_prompt_ids: list[int],
    teacher_prompt_ids: list[int],
    target_ids: list[int],
    direction: KLDirections,
) -> list[TokenKL]:
    if not target_ids:
        return []

    base_logps = logprobs_for_target(model, base_prompt_ids, target_ids)
    teacher_logps = logprobs_for_target(model, teacher_prompt_ids, target_ids)
    kls = token_kl_values(base_logps, teacher_logps, direction).detach().cpu()
    token_tensor = torch.tensor(target_ids, dtype=torch.long)
    base_token_logps = base_logps.detach().cpu()[torch.arange(len(target_ids)), token_tensor]
    teacher_token_logps = teacher_logps.detach().cpu()[torch.arange(len(target_ids)), token_tensor]

    rows = []
    for i, token_id in enumerate(target_ids):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        rows.append(
            TokenKL(
                sample_id=sample_id,
                dataset_idx=dataset_idx,
                comparison=comparison,
                target_name=target_name,
                token_index=i,
                token_id=token_id,
                token_text=token_text,
                kl=float(kls[i].item()),
                base_logp_token=float(base_token_logps[i].item()),
                teacher_logp_token=float(teacher_token_logps[i].item()),
                logp_gap_teacher_minus_base=float((teacher_token_logps[i] - base_token_logps[i]).item()),
            )
        )
    return rows


def summarize_trace(rows: list[TokenKL], prefix: str) -> dict[str, Any]:
    if not rows:
        return {
            f"{prefix}_mean_kl": math.nan,
            f"{prefix}_max_kl": math.nan,
            f"{prefix}_argmax_kl": None,
            f"{prefix}_argmax_token": "",
        }
    max_row = max(rows, key=lambda item: item.kl)
    return {
        f"{prefix}_mean_kl": sum(item.kl for item in rows) / len(rows),
        f"{prefix}_max_kl": max_row.kl,
        f"{prefix}_argmax_kl": max_row.token_index,
        f"{prefix}_argmax_token": max_row.token_text,
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dtype_from_arg(name: str):
    if name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="HF model id or local model path.")
    parser.add_argument("--dataset-path", type=Path, default=Path("datasets/tooluse/test.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("explorations/self_conditioning_kl/runs/latest"))
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--split", choices=["train", "test", "val", "validation"], default=None)
    parser.add_argument(
        "--max-code-test-cases",
        type=int,
        default=None,
        help="Optional cap for code verifier test cases on non-test splits.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-context-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--kl-direction", choices=["teacher||base", "base||teacher", "js"], default="base||teacher")
    parser.add_argument("--torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true", help="Leave Qwen3 thinking mode on. Default is off.")
    parser.add_argument("--max-plots", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_from_arg(args.torch_dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    all_rows = load_jsonl(args.dataset_path)
    rows = all_rows[args.start_index : args.start_index + args.num_samples]
    split = infer_split(args.dataset_path, args.split)
    token_rows: list[TokenKL] = []
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for sample_id, row in enumerate(rows):
        dataset_idx = str(row.get("idx", args.start_index + sample_id))
        problem = row["prompt"]
        system = row.get("system")
        data_source = normalize_data_source(row)
        ground_truth = ground_truth_for_row(row, data_source)
        extra_info = extra_info_for_row(row, split)

        print(f"[{sample_id + 1}/{len(rows)}] generating first attempt for dataset idx={dataset_idx}")
        x_messages = base_messages(problem, system)
        y_text, y_ids, x_prompt_ids = generate_response(
            model,
            tokenizer,
            x_messages,
            max_context_tokens=args.max_context_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            enable_thinking=args.enable_thinking,
        )
        y_score = score_response(
            data_source=data_source,
            solution=y_text,
            ground_truth=ground_truth,
            extra_info=extra_info,
            max_code_test_cases=args.max_code_test_cases,
        )
        f_text = feedback_text(y_score)

        xf_messages = feedback_only_messages(problem, f_text, system)
        xyf_messages = attempt_feedback_messages(problem, y_text, f_text, system)
        xf_prompt_ids = maybe_truncate_context(
            apply_chat_template(tokenizer, xf_messages, args.enable_thinking), args.max_context_tokens
        )[0].tolist()
        xyf_prompt_ids = maybe_truncate_context(
            apply_chat_template(tokenizer, xyf_messages, args.enable_thinking), args.max_context_tokens
        )[0].tolist()

        print(f"[{sample_id + 1}/{len(rows)}] generating second attempt for dataset idx={dataset_idx}")
        y_prime_text, y_prime_ids, _ = generate_response(
            model,
            tokenizer,
            xyf_messages,
            max_context_tokens=args.max_context_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            enable_thinking=args.enable_thinking,
        )
        y_prime_score = score_response(
            data_source=data_source,
            solution=y_prime_text,
            ground_truth=ground_truth,
            extra_info=extra_info,
            max_code_test_cases=args.max_code_test_cases,
        )

        print(f"[{sample_id + 1}/{len(rows)}] computing KL traces for dataset idx={dataset_idx}")
        trace_y_xf = compute_token_trace(
            model=model,
            tokenizer=tokenizer,
            sample_id=sample_id,
            dataset_idx=dataset_idx,
            comparison="kl_y__x_vs_x_f",
            target_name="y",
            base_prompt_ids=x_prompt_ids,
            teacher_prompt_ids=xf_prompt_ids,
            target_ids=y_ids,
            direction=args.kl_direction,
        )
        trace_y_xyf = compute_token_trace(
            model=model,
            tokenizer=tokenizer,
            sample_id=sample_id,
            dataset_idx=dataset_idx,
            comparison="kl_y__x_vs_x_y_f",
            target_name="y",
            base_prompt_ids=x_prompt_ids,
            teacher_prompt_ids=xyf_prompt_ids,
            target_ids=y_ids,
            direction=args.kl_direction,
        )
        trace_yp_xyf = compute_token_trace(
            model=model,
            tokenizer=tokenizer,
            sample_id=sample_id,
            dataset_idx=dataset_idx,
            comparison="kl_yprime__x_vs_x_y_f",
            target_name="y_prime",
            base_prompt_ids=x_prompt_ids,
            teacher_prompt_ids=xyf_prompt_ids,
            target_ids=y_prime_ids,
            direction=args.kl_direction,
        )
        token_rows.extend(trace_y_xf)
        token_rows.extend(trace_y_xyf)
        token_rows.extend(trace_yp_xyf)

        summary = {
            "sample_id": sample_id,
            "dataset_idx": dataset_idx,
            "data_source": data_source,
            "score_y": y_score.get("score"),
            "score_y_prime": y_prime_score.get("score"),
            "acc_y": y_score.get("acc"),
            "acc_y_prime": y_prime_score.get("acc"),
            "incorrect_format_y": y_score.get("incorrect_format"),
            "incorrect_format_y_prime": y_prime_score.get("incorrect_format"),
            "len_y_tokens": len(y_ids),
            "len_y_prime_tokens": len(y_prime_ids),
            "feedback": f_text,
            **summarize_trace(trace_y_xf, "kl_y__x_vs_x_f"),
            **summarize_trace(trace_y_xyf, "kl_y__x_vs_x_y_f"),
            **summarize_trace(trace_yp_xyf, "kl_yprime__x_vs_x_y_f"),
        }
        summary_rows.append(summary)
        detail_rows.append(
            {
                **summary,
                "problem": problem,
                "ground_truth": ground_truth,
                "extra_info": extra_info,
                "y": y_text,
                "y_score": y_score,
                "y_prime": y_prime_text,
                "y_prime_score": y_prime_score,
                "messages_x": x_messages,
                "messages_x_f": xf_messages,
                "messages_x_y_f": xyf_messages,
                "prompt_x_f": xf_messages[-1]["content"],
                "prompt_x_y_f": xyf_messages[-1]["content"],
            }
        )

    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "token_kl.csv", [asdict(row) for row in token_rows])
    write_jsonl(args.output_dir / "details.jsonl", detail_rows)
    plot_traces(args.output_dir, token_rows, max_plots=args.max_plots)
    write_token_heatmap_report(args.output_dir, detail_rows, token_rows)

    print(f"Wrote {len(summary_rows)} samples to {args.output_dir}")
    print(f"Summary: {args.output_dir / 'summary.csv'}")
    print(f"Token traces: {args.output_dir / 'token_kl.csv'}")
    print(f"Token heatmap report: {args.output_dir / 'token_heatmap_report.html'}")


if __name__ == "__main__":
    main()
