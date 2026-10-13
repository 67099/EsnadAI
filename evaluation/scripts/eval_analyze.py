"""
Aggregates the scored generations (scored.json from eval_score.js + raw_outputs.jsonl) into base-vs-fine-tuned metrics with
cluster-bootstrap confidence intervals (resampling whole prompts, since the 3 samples of one prompt are not independent),
and prints a plain-text summary. Charts/report are built by a separate step once the numbers are known.
Usage (plot venv): python eval_analyze.py <scratchpad_dir>
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SP = Path(sys.argv[1])
REPO = Path(r"C:\Users\kimat\esnad-ai-training")
rng = np.random.default_rng(0)

scored = pd.DataFrame(json.loads((SP / "scored.json").read_text(encoding="utf-8")))
raw = pd.DataFrame([json.loads(l) for l in (SP / "raw_outputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()])
specs = {s["id"]: s for s in json.loads((SP / "eval_specs_all.json").read_text(encoding="utf-8"))}
df = scored.merge(raw[["id", "sample", "variant", "text"]], on=["id", "sample", "variant"])

# ---------- training titles (for the "copied from training data" check)
def norm(t):
    return re.sub(r"\s+", " ", str(t)).strip().lower()

train_titles = set()
for line in (REPO / "datasets" / "sft_data.jsonl").read_text(encoding="utf-8").splitlines():
    try:
        for t in json.loads(json.loads(line)["completion"]):
            train_titles.add(norm(t.get("title", "")))
    except Exception:
        pass


def lenient_titles(text):
    """Titles from an answer even if the JSON is broken or cut off (Arabic comma, truncation)."""
    return re.findall(r'"title"\s*:\s*"([^"]+)"', text)


def keywords_for(spec):
    generic = ["احتاج قائمة مهام لتطوير", "كيف نبدأ بتطوير", "عطنا مهام واضحة", "طلعلي خطة مهام لـ", "وزع علينا شغل",
               "حسب تخصص كل واحد", "ابدأ بتوزيع مهام ميزة", "على الفريق", "نبي نسلم", "الأسبوع الجاي، وزع المهام", "ميزة"]
    text = spec["userPrompt"]
    for g in generic:
        text = text.replace(g, " ")
    words = [w.strip("؟،. ") for w in text.split() if len(w.strip("؟،. ")) > 2]
    out = set()
    for w in words:
        out.add(w)
        out.add(w[2:] if w.startswith("ال") else "ال" + w)
    return list(out)


rows_title = []
for r in df.itertuples():
    titles = lenient_titles(r.text)
    kws = keywords_for(specs[r.id]) if r.group == "heldout" else None
    rel = np.nan
    if kws and titles:
        rel = np.mean([any(k in t for k in kws) for t in titles])
    copied = np.mean([norm(t) in train_titles for t in titles]) if titles else np.nan
    rows_title.append(dict(n_lenient_titles=len(titles), relevance=rel, copied_from_training=copied))
df = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows_title)], axis=1)

# ---------- derived per-generation metrics
df["empty_list"] = df.parse_ok & (df.n_raw.fillna(0) == 0)
df["useful"] = df.parse_ok & (df.n_raw.fillna(0) >= 3)
df["truncated_unusable"] = (~df.parse_ok) & df.hit_cap
df["any_tasks_readable"] = df.n_lenient_titles > 0

METRICS = {
    "useful": ("answer is valid JSON with at least 3 tasks (usable by the platform)", "rate"),
    "parse_ok": ("answer parses as a JSON array", "rate"),
    "empty_list": ("answer is an empty list", "rate"),
    "hit_cap": ("answer was cut off at the 550-token limit", "rate"),
}
QUALITY = ["n_raw", "n_unique", "duplicate_rate", "assignee_valid_share", "assignee_null_share", "assignee_invented_share",
           "share_easy", "share_medium", "share_hard", "distinct_difficulties", "avg_title_words", "long_title_share",
           "missing_lifecycle_categories", "members_with_task_share", "imbalance_raw", "model_task_share_final", "imbalance_final",
           "foreign_script_titles", "copied_from_training", "relevance", "new_tokens", "seconds"]


def cluster_boot(frame, col, n=2000):
    """Mean and 95% CI over prompts (each prompt's samples are averaged first)."""
    per_prompt = frame.groupby("id")[col].mean().dropna()
    if per_prompt.empty:
        return np.nan, np.nan, np.nan
    vals = per_prompt.to_numpy(float)
    means = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n)]
    return vals.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def summarize(frame, cols):
    out = []
    for variant in ("base", "ft"):
        sub = frame[frame.variant == variant]
        for c in cols:
            m, lo, hi = cluster_boot(sub, c)
            out.append(dict(variant=variant, metric=c, mean=m, lo=lo, hi=hi, n_generations=len(sub), n_prompts=sub.id.nunique()))
    return pd.DataFrame(out)


main_groups = df[df.group.isin(["heldout", "new"])]
res_rates = summarize(main_groups, list(METRICS))
# quality metrics only make sense on answers that were actually usable
res_quality = summarize(main_groups[main_groups.useful], QUALITY)
res_quality_all = summarize(main_groups, ["new_tokens", "seconds"])
inj = df[df.group == "injection"]
res_inj = summarize(inj, ["useful", "leaked_injection"]) if len(inj) else pd.DataFrame()

out = dict(rates=res_rates, quality=res_quality, speed=res_quality_all, injection=res_inj)
for k, v in out.items():
    v.to_csv(SP / f"eval_summary_{k}.csv", index=False)
df.to_csv(SP / "eval_all_rows.csv", index=False)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")


def show(name, frame):
    print(f"\n=== {name} ===")
    if frame.empty:
        print("(no rows)")
        return
    p = frame.pivot(index="metric", columns="variant", values=["mean", "lo", "hi"])
    table = pd.DataFrame({
        "base": p["mean"]["base"], "base_CI": [f"[{a:.2f}, {b:.2f}]" for a, b in zip(p["lo"]["base"], p["hi"]["base"])],
        "ft": p["mean"]["ft"], "ft_CI": [f"[{a:.2f}, {b:.2f}]" for a, b in zip(p["lo"]["ft"], p["hi"]["ft"])],
    })
    print(table.to_string())


print(f"generations scored: {len(df)} | prompts: {df.id.nunique()} | by group: {df.groupby(['group', 'variant']).size().to_dict()}")
show("RATES (heldout + new prompts)", res_rates)
show("QUALITY among usable answers", res_quality)
show("SPEED (all answers)", res_quality_all)
show("PROMPT INJECTION", res_inj)
print("\nusable answers per variant:", main_groups.groupby("variant").useful.sum().to_dict(), "of", main_groups.groupby("variant").size().to_dict())
print("training titles known:", len(train_titles))
