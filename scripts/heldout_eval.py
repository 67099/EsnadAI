"""
Rigorous, objective generalization test: runs the 5 held-out project ideas
(NEVER seen during training — see HELDOUT_INDICES in generate_synthetic_data.py)
through the base model (before) and the v2 fine-tuned adapter (after), then
scores topical relevance automatically using distinctive per-project keywords
— not a subjective eyeball read.
"""
import json
import re
import subprocess

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = "models/allam-7b-instruct"
ADAPTER_PATH = "models/allam-7b-esnad-lora-v5/final_adapter"
HELDOUT_PATH = "datasets/heldout_eval_specs.json"
REPORT_PATH = "heldout_generalization_report_v5.md"

# Generic phrasing that wraps the actual feature name in USER_PROMPTS — stripped
# out so keyword scoring is derived from the SPECIFIC feature actually asked
# about in each case, not a static per-project guess (a v3-run bug: the random
# feature picked differs run to run, so a fixed keyword list can silently score
# the wrong thing — e.g. scoring a "billing" response against "tables" keywords).
_GENERIC_PHRASES = [
    "احتاج قائمة مهام لتطوير", "كيف نبدأ بتطوير", "عطنا مهام واضحة", "طلعلي خطة مهام لـ",
    "وزع علينا شغل", "حسب تخصص كل واحد", "ابدأ بتوزيع مهام ميزة", "على الفريق",
    "نبي نسلم", "الأسبوع الجاي، وزع المهام", "ميزة",
]


def keywords_for(spec):
    text = spec["userPrompt"]
    for phrase in _GENERIC_PHRASES:
        text = text.replace(phrase, " ")
    words = [w.strip("؟،. ") for w in text.split() if len(w.strip("؟،. ")) > 2]
    # also add each word with/without the leading "ال" so morphological
    # variants (السعرات vs للسعرات vs سعرات) still match as substrings
    expanded = set()
    for w in words:
        expanded.add(w)
        expanded.add(w[2:] if w.startswith("ال") else "ال" + w)
    return list(expanded)


def build_prompt_via_node(spec):
    payload = json.dumps({
        "userPrompt": spec["userPrompt"],
        "projectIdea": spec["projectIdea"],
        "teamMembers": spec["teamMembers"],
    })
    result = subprocess.run(
        ["node", "-e", f"""
        const {{ buildPrompt }} = require('C:/Users/kimat/GP2-Cl/services/aiEngine');
        process.stdout.write(buildPrompt({payload}));
        """],
        capture_output=True, text=True, encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout


def generate(model, tokenizer, prompt, max_new_tokens=350):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.4, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def score_relevance(response_text, keywords):
    try:
        cleaned = response_text.replace("،", ",")
        match = re.search(r"\[.*\]", cleaned, re.S)
        tasks = json.loads(match.group(0)) if match else []
    except Exception:
        return None, 0, 0  # unparseable JSON

    if not tasks:
        return [], 0, 0

    matched = 0
    for t in tasks:
        title = t.get("title", "") if isinstance(t, dict) else ""
        if any(kw in title for kw in keywords):
            matched += 1
    return tasks, matched, len(tasks)


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb_config, device_map="auto"
    )
    print("Attaching v5 fine-tuned adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    heldout = json.load(open(HELDOUT_PATH, encoding="utf-8"))

    report = ["# تقرير التعميم على مشاريع لم يشاهدها الموديل إطلاقًا (v5)\n"]
    total_before_matched, total_before_all = 0, 0
    total_after_matched, total_after_all = 0, 0
    total_after_null = 0

    for spec in heldout:
        idea = spec["projectIdea"]
        keywords = keywords_for(spec)
        prompt = build_prompt_via_node(spec)

        print(f"\n{'=' * 70}\n{idea}\n{'=' * 70}")

        with model.disable_adapter():
            before = generate(model, tokenizer, prompt)
        before_tasks, b_matched, b_total = score_relevance(before, keywords)
        print(f"BEFORE: {b_matched}/{b_total} tasks topically relevant")
        print(before)

        after = generate(model, tokenizer, prompt)
        after_tasks, a_matched, a_total = score_relevance(after, keywords)
        after_nulls = sum(1 for t in (after_tasks or []) if isinstance(t, dict) and t.get("suggestedAssignee") is None)
        print(f"AFTER:  {a_matched}/{a_total} tasks topically relevant, {after_nulls} null assignees")
        print(after)

        total_before_matched += b_matched
        total_before_all += b_total
        total_after_matched += a_matched
        total_after_all += a_total
        total_after_null += after_nulls

        report.append(f"## {idea}\n")
        report.append(f"**كلمات التقييم المستخدمة:** {', '.join(keywords)}\n")
        report.append(f"**قبل:** {b_matched}/{b_total} صحيح\n```\n{before}\n```\n")
        report.append(f"**بعد (v5):** {a_matched}/{a_total} صحيح، {after_nulls} بدون مسؤول\n```\n{after}\n```\n")
        report.append("---\n")

    summary = (
        f"\n## الخلاصة الرقمية\n"
        f"- قبل التدريب: {total_before_matched}/{total_before_all} "
        f"({100*total_before_matched/max(total_before_all,1):.0f}%)\n"
        f"- بعد التدريب (v5): {total_after_matched}/{total_after_all} "
        f"({100*total_after_matched/max(total_after_all,1):.0f}%)\n"
        f"- مهام بدون مسؤول بعد التدريب (المفروض ٠): {total_after_null}\n"
    )
    report.append(summary)
    print(summary)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
