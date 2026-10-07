#!/usr/bin/env python3
"""Persist full baseline-vs-fine-tune outputs for the report's qualitative cases."""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labkit import evaluate as ev, generate, report
from labkit.config import get_tier


def load_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    tier = get_tier()
    target = load_jsonl(ROOT / "data" / "eval_target.jsonl")
    regression = load_jsonl(ROOT / "data" / "eval_regression.jsonl")
    selected = [12, 18, 23, 48, 49]
    records = [target[i] for i in selected]
    prompts = [record["input"] for record in records]

    model, tok = generate.load_base(tier)
    baseline_preds, _ = generate.generate_batch(
        model, tok, prompts, system=generate.OPTIMIZED_PROMPT,
        label="qualitative/base-b",
    )
    baseline_regression_preds, _ = generate.generate_batch(
        model, tok, [record["instruction"] for record in regression], system=None,
        max_new_tokens=96, label="qualitative/base-regression",
    )

    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(ROOT / "adapters" / "correct"))
    model.eval()
    finetune_preds, _ = generate.generate_batch(
        model, tok, prompts, system=generate.NAIVE_PROMPT,
        label="qualitative/fine-tune",
    )
    finetune_regression_preds, _ = generate.generate_batch(
        model, tok, [record["instruction"] for record in regression], system=None,
        max_new_tokens=96, label="qualitative/ft-regression",
    )

    rows = []
    for index, record, baseline, finetune in zip(
        selected, records, baseline_preds, finetune_preds
    ):
        rows.append({
            "i": index,
            "ticket": record["input"],
            "label": record["label"],
            "baseline_b_pred": baseline,
            "baseline_b_score": ev.triage_field_accuracy(baseline, record["label"]),
            "finetune_pred": finetune,
            "finetune_score": ev.triage_field_accuracy(finetune, record["label"]),
        })

    report.write_json(rows, "qualitative_comparison.json", results_dir=ROOT / "results")
    print(report.markdown_table(
        rows,
        ["i", "baseline_b_score", "finetune_score", "baseline_b_pred", "finetune_pred"],
    ))

    regression_rows = []
    for index, record, baseline, finetune in zip(
        range(len(regression)), regression, baseline_regression_preds,
        finetune_regression_preds
    ):
        base_score = ev.keyword_recall(baseline, record["keywords"])
        ft_score = ev.keyword_recall(finetune, record["keywords"])
        regression_rows.append({
            "i": index,
            "instruction": record["instruction"],
            "keywords": record["keywords"],
            "baseline_pred": baseline,
            "baseline_score": base_score,
            "finetune_pred": finetune,
            "finetune_score": ft_score,
            "delta": ft_score - base_score,
        })
    regression_rows.sort(key=lambda row: row["delta"])
    report.write_json(
        regression_rows,
        "qualitative_regression_comparison.json",
        results_dir=ROOT / "results",
    )
    print("\nWorst regression cases:")
    print(report.markdown_table(
        regression_rows[:5],
        ["i", "baseline_score", "finetune_score", "instruction",
         "baseline_pred", "finetune_pred"],
    ))


if __name__ == "__main__":
    main()
