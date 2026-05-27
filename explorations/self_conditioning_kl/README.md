# Self-Conditioning KL Probe

This folder contains an exploratory probe for comparing SDPO-style privileged
prompts with a second-attempt prompt that also includes the original attempt.
It is standalone with respect to training, but intentionally reuses the
repository's SDPO feedback scorers in `verl.utils.reward_score.feedback`.

It runs the following loop on a small ToolUse or code subset:

1. Generate `y ~ p(. | x)` with Qwen3-8B, with Qwen thinking disabled by default.
2. Score `y` with the same feedback-producing verifier used by SDPO.
3. Compare token-level KL traces for:
   - `kl_y__x_vs_x_f`: target `y`, original prompt `x` versus feedback-only prompt `x,f`.
   - `kl_y__x_vs_x_y_f`: target `y`, original prompt `x` versus second-attempt prompt `x,y,f`.
   - `kl_yprime__x_vs_x_y_f`: target `y'`, original prompt `x` versus second-attempt prompt `x,y,f`, where `y'` is generated from `p(. | x,y,f)`.
4. Score both `y` and `y'`.

The second-attempt prompt is represented as an actual chat history:
`user: x`, `assistant: y`, `user: environment feedback + retry request`.
This follows verl's interaction loop convention, where environment/interactor
responses are appended as user messages. The `tool` role is reserved for real
tool-call outputs, not verifier feedback. For the feedback-only `x,f` baseline,
there is no prior assistant turn, so the privileged feedback is kept inside a
single user message.

By default the KL direction is `KL(p_teacher || p_base)`, matching the usual
distillation orientation. Use `--kl-direction base||teacher` or `--kl-direction js`
for ablations.

Supported reward sources are the SDPO feedback modules used by this probe:
`tooluse`, `code`, `livecodebench`, and `humanevalplus`. Raw
rows with `kind == "code"` are routed to the code verifier if their dataset name
is not one of the explicit code keys.

## Run

From the repository root:

```bash
python3 explorations/self_conditioning_kl/run_kl_probe.py \
  --model Qwen/Qwen3-8B \
  --dataset-path datasets/tooluse/test.json \
  --num-samples 4 \
  --max-new-tokens 192 \
  --output-dir explorations/self_conditioning_kl/runs/qwen3_8b_tooluse
```

For code data:

```bash
python3 explorations/self_conditioning_kl/run_kl_probe.py \
  --model Qwen/Qwen3-8B \
  --dataset-path path/to/code/test.json \
  --num-samples 2 \
  --max-new-tokens 512 \
  --output-dir explorations/self_conditioning_kl/runs/qwen3_8b_code
```

On non-test splits, `--max-code-test-cases N` can cap the number of verifier
test cases per sample for faster iteration. On `test` splits the verifier follows
the existing SDPO logic and evaluates all tests.

If the model is stored locally, pass its local path to `--model`. The script does
not need training infrastructure, Ray, vLLM, or SDPO integration.

## Outputs

- `summary.csv`: one row per problem with `y`/`y'` scores and mean/max KLs.
- `token_kl.csv`: one row per target token per comparison, useful for plotting.
- `details.jsonl`: prompts, generations, feedback, and score metadata.
- `sample_XXX_kl.png`: quick line plots of token-level KL peaks.
- `token_heatmap_report.html`: browsable report with prompt, response,
  feedback, and three token-highlighted target columns.
- `sample_XXX_token_heatmap.html`: per-sample token KL heatmap page.

Useful quick inspection:

```python
import pandas as pd

summary = pd.read_csv("explorations/self_conditioning_kl/runs/qwen3_8b_tooluse/summary.csv")
tokens = pd.read_csv("explorations/self_conditioning_kl/runs/qwen3_8b_tooluse/token_kl.csv")

summary[[
    "sample_id",
    "score_y",
    "score_y_prime",
    "kl_y__x_vs_x_f_mean_kl",
    "kl_y__x_vs_x_y_f_mean_kl",
    "kl_yprime__x_vs_x_y_f_mean_kl",
]]

tokens.sort_values("kl", ascending=False).head(20)
```
