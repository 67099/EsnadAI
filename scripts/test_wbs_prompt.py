"""
Tests the NEW WBS-style prompt (12-20 tasks, full lifecycle, weighted-fair
distribution) against the CURRENT v4 model — no retraining. Picks 5 diverse
projects/teams and scores each response objectively: task count in range,
zero nulls, weighted workload balance (hard=3, medium=2, easy=1), and
whether all 6 WBS phases (management/design/backend/frontend/qa/deployment)
are represented via skillCategory-ish keyword matching on titles.
"""
import json
import subprocess

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = "models/allam-7b-instruct"
ADAPTER_PATH = "models/allam-7b-esnad-lora-v4/final_adapter"
REPORT_PATH = "wbs_prompt_test_report.md"

DIFFICULTY_WEIGHT = {"easy": 1, "medium": 2, "hard": 3}

TEST_CASES = [
    {
        "projectIdea": "منصة لحجز مواعيد صالونات الحلاقة والتجميل",
        "userPrompt": "وزع علينا خطة العمل الكاملة للمشروع",
        "teamMembers": [
            {"name": "مشعل", "about": "أجيد React وTailwind، أفضل شغل الواجهات"},
            {"name": "عبدالرحمن", "about": "خبرة Node.js وMongoDB"},
            {"name": "سارة", "about": "أصمم واجهات المستخدم على Figma"},
        ],
    },
    {
        "projectIdea": "تطبيق لتوصيل طلبات البقالة من السوبرماركت للمنازل",
        "userPrompt": "نبي خطة عمل كاملة للمشروع",
        "teamMembers": [
            {"name": "فيصل", "about": "خبرة بـ Django وPostgreSQL"},
            {"name": "نورة", "about": "أختبر الأداء والاختبارات الآلية"},
            {"name": "بدر", "about": "أطور تطبيقات جوال بـ Flutter"},
        ],
    },
    {
        "projectIdea": "منصة لتعليم البرمجة للأطفال بأسلوب تفاعلي وألعاب",
        "userPrompt": "وزع خطة العمل الكاملة على الفريق",
        "teamMembers": [
            {"name": "ريم", "about": "خبرة بإعداد السيرفرات ومراقبة الأداء والنشر الآلي"},
            {"name": "خالد", "about": "أجيد Vue.js وCSS"},
            {"name": "لينا", "about": ""},
        ],
    },
    {
        "projectIdea": "نظام لإدارة نوبات عمل الموظفين بمحلات السلاسل التجارية",
        "userPrompt": "ابدأ بتوزيع خطة العمل الكاملة للمشروع",
        "teamMembers": [
            {"name": "عمر", "about": "خبرة بتصميم قواعد البيانات العلائقية"},
            {"name": "تركي", "about": "أهتم بالـUI/UX وعمل تصاميم Figma"},
        ],
    },
    {
        "projectIdea": "منصة لبيع وشراء المنتجات المستعملة بين الطلاب",
        "userPrompt": "طلعلي خطة العمل الكاملة",
        "teamMembers": [
            {"name": "سلطان", "about": "خبرة بمعالجة اللغة الطبيعية وبناء نماذج تصنيف"},
            {"name": "فهد", "about": "أطور تطبيقات جوال بـ Flutter"},
            {"name": "وليد", "about": "أفضل كتابة الاختبارات وضمان جودة الكود"},
        ],
    },
]

PHASE_KEYWORDS = {
    "إدارة مشروع": ["متطلبات", "تخطيط", "خطة", "توثيق", "نطاق"],
    "تصميم": ["تصميم", "واجهة المستخدم", "UX", "UI", "Figma", "prototype", "برو"],
    "باك إند": ["API", "خادم", "خلفية", "منطق", "سيرفر"],
    "قاعدة بيانات": ["قاعدة بيانات", "جدول", "مخطط", "استعلام", "سجلات"],
    "فرونت إند": ["واجهة", "شاشة", "صفحة", "جوال", "تطبيق"],
    "جودة": ["اختبار", "فحص", "QA", "الحواف"],
    "نشر": ["نشر", "Deployment", "CI/CD", "استضافة", "إطلاق", "release"],
}


def build_prompt_via_node(spec):
    payload = json.dumps(spec)
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


def generate(model, tokenizer, prompt, max_new_tokens=900):
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


def analyze(tasks, team_names):
    n = len(tasks)
    in_range = 12 <= n <= 20
    nulls = sum(1 for t in tasks if t.get("suggestedAssignee") is None)

    weighted_load = {name: 0 for name in team_names}
    raw_count = {name: 0 for name in team_names}
    for t in tasks:
        who = t.get("suggestedAssignee")
        w = DIFFICULTY_WEIGHT.get(t.get("difficulty"), 1)
        if who in weighted_load:
            weighted_load[who] += w
            raw_count[who] += 1
    loads = list(weighted_load.values())
    balance_spread = max(loads) - min(loads) if loads else None

    phases_found = []
    for phase, kws in PHASE_KEYWORDS.items():
        if any(any(kw in t.get("title", "") for kw in kws) for t in tasks):
            phases_found.append(phase)

    return {
        "count": n, "in_range": in_range, "nulls": nulls,
        "weighted_load": weighted_load, "raw_count": raw_count,
        "balance_spread": balance_spread, "phases_found": phases_found,
    }


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
    print("Attaching v4 adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    report = ["# اختبار البرومبت الجديد (WBS) على v4 — بدون إعادة تدريب\n"]

    for i, case in enumerate(TEST_CASES, 1):
        team_names = [m["name"] for m in case["teamMembers"]]
        prompt = build_prompt_via_node(case)
        print(f"\n{'='*70}\n{i}. {case['projectIdea']}\n{'='*70}")

        response = generate(model, tokenizer, prompt)
        try:
            cleaned = response.replace("،", ",")
            tasks = json.loads(cleaned[cleaned.index("["):cleaned.rindex("]")+1])
        except Exception as e:
            print("PARSE ERROR:", e)
            report.append(f"## {i}. {case['projectIdea']}\n\n**فشل التحليل (JSON غير صالح)**\n```\n{response}\n```\n---\n")
            continue

        analysis = analyze(tasks, team_names)
        print(f"عدد المهام: {analysis['count']} ({'ضمن النطاق' if analysis['in_range'] else 'خارج النطاق!'})")
        print(f"null: {analysis['nulls']}")
        print(f"الحمل الموزون: {analysis['weighted_load']}")
        print(f"فرق التوازن (0=مثالي): {analysis['balance_spread']}")
        print(f"المراحل المغطاة ({len(analysis['phases_found'])}/7): {analysis['phases_found']}")

        team_desc = ", ".join(
            m["name"] + " (" + (m["about"] or "بدون وصف") + ")" for m in case["teamMembers"]
        )
        report.append(f"## {i}. {case['projectIdea']}\n")
        report.append(f"**الفريق:** {team_desc}\n")
        report.append(f"**عدد المهام:** {analysis['count']} — {'✅ ضمن ١٢-٢٠' if analysis['in_range'] else '❌ خارج النطاق'}\n")
        report.append(f"**مهام بدون مسؤول:** {analysis['nulls']} {'✅' if analysis['nulls']==0 else '❌'}\n")
        report.append(f"**الحمل الموزون بالصعوبة لكل عضو:** {analysis['weighted_load']}\n")
        report.append(f"**فرق التوازن (الأقل=الأفضل):** {analysis['balance_spread']}\n")
        report.append(f"**مراحل WBS المغطاة ({len(analysis['phases_found'])}/7):** {', '.join(analysis['phases_found']) or 'لا شيء'}\n")
        report.append(f"**الرد الكامل:**\n```json\n{json.dumps(tasks, ensure_ascii=False, indent=2)}\n```\n")
        report.append("---\n")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
