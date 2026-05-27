"""Visualization helpers for the self-conditioning KL probe."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

COMPARISON_TITLES = {
    "kl_y__x_vs_x_f": "Target y: p(.|x) vs p(.|x,f)",
    "kl_y__x_vs_x_y_f": "Target y: p(.|x) vs p(.|x,y,f)",
    "kl_yprime__x_vs_x_y_f": "Target y': p(.|x) vs p(.|x,y,f)",
}


def _row_dict(row: Any) -> dict[str, Any]:
    if is_dataclass(row):
        return asdict(row)
    return dict(row)


def plot_traces(output_dir: Path, token_rows: list[Any], max_plots: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots.")
        return

    rows = [_row_dict(row) for row in token_rows]
    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sample[int(row["sample_id"])].append(row)

    for sample_id in sorted(by_sample)[:max_plots]:
        plt.figure(figsize=(11, 4))
        sample_rows = by_sample[sample_id]
        for comparison in sorted({row["comparison"] for row in sample_rows}):
            series = [row for row in sample_rows if row["comparison"] == comparison]
            series.sort(key=lambda row: int(row["token_index"]))
            plt.plot([row["token_index"] for row in series], [row["kl"] for row in series], label=comparison)
        plt.xlabel("target token index")
        plt.ylabel("token KL")
        plt.title(f"Sample {sample_id}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"sample_{sample_id:03d}_kl.png", dpi=160)
        plt.close()


def write_token_heatmap_report(output_dir: Path, detail_rows: list[dict[str, Any]], token_rows: list[Any]) -> None:
    rows = [_row_dict(row) for row in token_rows]
    by_sample_and_comparison: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (int(row["sample_id"]), str(row["comparison"]))
        by_sample_and_comparison[key].append(row)

    for series in by_sample_and_comparison.values():
        series.sort(key=lambda row: int(row["token_index"]))

    pages = []
    for detail in detail_rows:
        sample_id = int(detail["sample_id"])
        page_name = f"sample_{sample_id:03d}_token_heatmap.html"
        pages.append((page_name, detail))
        html_text = render_sample_page(detail, by_sample_and_comparison)
        (output_dir / page_name).write_text(html_text)

    (output_dir / "token_heatmap_report.html").write_text(render_index_page(pages))


def render_index_page(pages: list[tuple[str, dict[str, Any]]]) -> str:
    items = []
    for page_name, detail in pages:
        label = (
            f"Sample {detail['sample_id']} | dataset idx {detail['dataset_idx']} | "
            f"score {detail.get('score_y')} -> {detail.get('score_y_prime')}"
        )
        items.append(f'<li><a href="{html.escape(page_name)}">{html.escape(label)}</a></li>')
    return html_document("Token KL Heatmap Report", f"<h1>Token KL Heatmap Report</h1><ul>{''.join(items)}</ul>")


def render_sample_page(
    detail: dict[str, Any],
    by_sample_and_comparison: dict[tuple[int, str], list[dict[str, Any]]],
) -> str:
    sample_id = int(detail["sample_id"])
    sample_rows = [
        row
        for (sid, _), rows in by_sample_and_comparison.items()
        if sid == sample_id
        for row in rows
    ]
    max_kl = max([float(row["kl"]) for row in sample_rows] + [1e-12])

    context = render_context(detail)
    columns = []
    for comparison in COMPARISON_TITLES:
        rows = by_sample_and_comparison.get((sample_id, comparison), [])
        columns.append(render_token_column(COMPARISON_TITLES[comparison], rows, max_kl))

    body = f"""
    <header>
      <a class="back" href="token_heatmap_report.html">Back to index</a>
      <h1>Sample {html.escape(str(detail["sample_id"]))}</h1>
      <div class="meta">
        dataset idx <strong>{html.escape(str(detail["dataset_idx"]))}</strong>
        | source <strong>{html.escape(str(detail.get("data_source", "")))}</strong>
        | score <strong>{html.escape(str(detail.get("score_y")))}</strong>
        -> <strong>{html.escape(str(detail.get("score_y_prime")))}</strong>
      </div>
    </header>
    {context}
    <section>
      <h2>Token-Level KL Targets</h2>
      <div class="legend"><span></span> lower KL <strong></strong> higher KL</div>
      <div class="columns">{''.join(columns)}</div>
    </section>
    """
    return html_document(f"Sample {detail['sample_id']} Token KL", body)


def render_context(detail: dict[str, Any]) -> str:
    return f"""
    <section class="context">
      <h2>Prompt, Response, Feedback</h2>
      <div class="context-grid">
        {render_text_block("Original Prompt", detail.get("problem", ""))}
        {render_text_block("Generated Response y", detail.get("y", ""))}
        {render_text_block("Environment Feedback f", detail.get("feedback", ""))}
      </div>
    </section>
    """


def render_text_block(title: str, text: Any) -> str:
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False, indent=2)
    return f"""
    <article class="text-block">
      <h3>{html.escape(title)}</h3>
      <pre>{html.escape(text)}</pre>
    </article>
    """


def render_token_column(title: str, rows: list[dict[str, Any]], max_kl: float) -> str:
    if not rows:
        tokens = '<p class="empty">No tokens generated for this target.</p>'
        mean_kl = "nan"
        max_col_kl = "nan"
    else:
        tokens = "".join(render_token(row, max_kl) for row in rows)
        mean_kl = f"{sum(float(row['kl']) for row in rows) / len(rows):.4g}"
        max_col_kl = f"{max(float(row['kl']) for row in rows):.4g}"

    return f"""
    <article class="token-column">
      <h3>{html.escape(title)}</h3>
      <div class="stats">mean KL {mean_kl} | max KL {max_col_kl}</div>
      <div class="tokens">{tokens}</div>
    </article>
    """


def render_token(row: dict[str, Any], max_kl: float) -> str:
    kl = float(row["kl"])
    intensity = min(max(kl / max_kl, 0.0), 1.0)
    alpha = 0.08 + 0.72 * intensity
    token = str(row["token_text"]).replace("\n", "\\n\n")
    title = (
        f"idx={row['token_index']} | token_id={row['token_id']} | KL={kl:.6g} | "
        f"base_logp={float(row['base_logp_token']):.6g} | "
        f"teacher_logp={float(row['teacher_logp_token']):.6g}"
    )
    return (
        f'<span class="tok" title="{html.escape(title)}" '
        f'style="background-color: rgba(207, 72, 54, {alpha:.3f});">'
        f"{html.escape(token)}</span>"
    )


def html_document(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --text: #202124;
      --muted: #666b73;
      --line: #dadce0;
      --accent: #cf4836;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header, section {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 20px 24px;
    }}
    h1, h2, h3 {{
      margin: 0;
      letter-spacing: 0;
    }}
    h1 {{ font-size: 28px; }}
    h2 {{ margin-bottom: 12px; font-size: 18px; }}
    h3 {{ font-size: 14px; }}
    .back, a {{ color: #2356a4; text-decoration: none; }}
    .meta, .stats, .legend {{ color: var(--muted); }}
    .context-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .text-block, .token-column {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 0;
    }}
    .text-block h3, .token-column h3, .stats {{
      padding: 10px 12px 0;
    }}
    pre {{
      margin: 0;
      padding: 10px 12px 12px;
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 300px;
      overflow: auto;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    .columns {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .tokens {{
      padding: 12px;
      white-space: pre-wrap;
      word-break: break-word;
      font: 13px/1.65 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    .tok {{
      border-radius: 3px;
      padding: 1px 0;
    }}
    .legend {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .legend span, .legend strong {{
      display: inline-block;
      width: 52px;
      height: 10px;
      border-radius: 999px;
      background: rgba(207, 72, 54, 0.1);
    }}
    .legend strong {{ background: rgba(207, 72, 54, 0.8); }}
    ul {{
      max-width: 900px;
      margin: 18px 0;
      padding-left: 28px;
    }}
    li {{ margin: 8px 0; }}
    .empty {{ margin: 12px; color: var(--muted); }}
    @media (max-width: 900px) {{
      .context-grid, .columns {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""
