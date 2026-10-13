"""
Stage 2: turn the stage-1 measurements into seaborn charts and an Arabic RTL report (HTML, then PDF via headless Chrome).
Run with the scratchpad plotting venv (seaborn, matplotlib, pandas).
"""
import base64
import re
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OUT = Path(sys.argv[1])
FIGS = OUT / "figs"
FIGS.mkdir(exist_ok=True)
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

df = pd.read_csv(OUT / "per_module.csv")
sv = pd.read_csv(OUT / "singular_values.csv")
summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
sample = np.load(OUT / "sample_q_proj_layer15.npz")

MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
PAL = sns.color_palette("deep", 7)
PAL_BY_MODULE = dict(zip(MODULES, PAL))


def save(fig, name):
    path = FIGS / name
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------- 1. heatmap: layer x module, relative change
pivot = df.pivot(index="layer", columns="module", values="rel_change_pct")[MODULES]
fig, ax = plt.subplots(figsize=(8.2, 11))
sns.heatmap(pivot, cmap="rocket_r", annot=True, fmt=".2f", annot_kws={"size": 7}, linewidths=0.4, linecolor="white",
            cbar_kws={"label": "size of the change relative to the weight (%)", "shrink": 0.6}, ax=ax)
ax.set_xlabel("")
ax.set_ylabel("transformer layer (0 = closest to the input, 31 = closest to the output)")
ax.set_title("How much each layer/module moved after fine-tuning  (||delta W|| / ||W||, %)", fontsize=11, pad=12)
plt.setp(ax.get_xticklabels(), rotation=0)
save(fig, "fig1_heatmap.png")

# ---------- 2. line plot: change by layer
fig, ax = plt.subplots(figsize=(9, 4.6))
sns.lineplot(data=df, x="layer", y="rel_change_pct", hue="module", hue_order=MODULES, palette=PAL, marker="o", markersize=4, linewidth=1.4, ax=ax)
mean_by_layer = df.groupby("layer").rel_change_pct.mean()
ax.plot(mean_by_layer.index, mean_by_layer.values, color="black", linestyle="--", linewidth=2, label="average of the 7 modules")
ax.set_xlabel("transformer layer")
ax.set_ylabel("relative change (%)")
ax.set_title("The upper layers changed more than the lower ones", fontsize=11)
ax.legend(ncol=4, fontsize=8, loc="upper left")
save(fig, "fig2_by_layer.png")

# ---------- 3. box + strip by module
order = df.groupby("module").rel_change_pct.median().sort_values(ascending=False).index.tolist()
fig, ax = plt.subplots(figsize=(9, 4.4))
sns.boxplot(data=df, x="module", y="rel_change_pct", order=order, palette=PAL_BY_MODULE, hue="module", legend=False, width=0.6, fliersize=0, ax=ax)
sns.stripplot(data=df, x="module", y="rel_change_pct", order=order, color="black", size=2.6, alpha=0.55, ax=ax)
ax.set_xlabel("")
ax.set_ylabel("relative change (%)")
ax.set_title("Which kind of layer moved most (each dot = one of the 32 layers)", fontsize=11)
save(fig, "fig3_by_module.png")

# ---------- 4. weight vs delta magnitude distributions (one representative module)
w, d = sample["w"], sample["delta"]
dist = pd.DataFrame({"abs_value": np.concatenate([np.abs(w), np.abs(d)]),
                     "which": ["original weight  W"] * len(w) + ["training change  delta W"] * len(d)})
dist = dist[dist.abs_value > 0]
fig, ax = plt.subplots(figsize=(9, 4.2))
sns.histplot(data=dist, x="abs_value", hue="which", log_scale=True, bins=90, stat="density", common_norm=False, element="step",
             palette=["#4c72b0", "#dd8452"], ax=ax)
ratio = float(np.median(np.abs(w)) / np.median(np.abs(d)))
ax.set_xlabel("absolute value of a single number (log scale)")
ax.set_title(f"Layer 15 q_proj: a typical weight is about {ratio:.0f}x larger than its typical change", fontsize=11)
save(fig, "fig4_distribution.png")

# ---------- 5. how many directions the change really uses (cumulative energy)
sv["energy"] = sv.sv ** 2
sv["cum_energy_pct"] = 100 * sv.groupby(["layer", "module"]).energy.cumsum() / sv.groupby(["layer", "module"]).energy.transform("sum")
fig, ax = plt.subplots(figsize=(9, 4.6))
sns.lineplot(data=sv, x="k", y="cum_energy_pct", hue="module", hue_order=MODULES, palette=PAL, errorbar="sd", linewidth=1.8, ax=ax)
ax.set_xlabel("number of directions used (LoRA rank is 16)")
ax.set_ylabel("share of the change explained (%)")
ax.set_xticks(range(1, 17))
ax.set_ylim(0, 101)
ax.set_title("The change is concentrated in a few directions (mean over 32 layers, band = std)", fontsize=11)
ax.legend(ncol=4, fontsize=8, loc="lower right")
save(fig, "fig5_rank_usage.png")

# ---------- 6. what survives when merged into bf16 weights
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
sns.barplot(data=df, x="module", y="bf16_elements_changed_pct", order=MODULES, palette=PAL_BY_MODULE, hue="module", legend=False, errorbar="sd", ax=axes[0])
axes[0].set_xlabel("")
axes[0].set_ylabel("numbers that actually changed (%)")
axes[0].set_title("Numbers that change after merging into bf16", fontsize=10)
axes[0].set_ylim(0, 100)
sns.barplot(data=df, x="module", y="bf16_merge_cosine", order=MODULES, palette=PAL_BY_MODULE, hue="module", legend=False, errorbar="sd", ax=axes[1])
axes[1].set_xlabel("")
axes[1].set_ylabel("similarity to the exact change (cosine)")
axes[1].set_title("How faithfully bf16 keeps the change", fontsize=10)
axes[1].set_ylim(0.9, 1.0)
for a in axes:
    plt.setp(a.get_xticklabels(), rotation=35, ha="right")
fig.tight_layout()
save(fig, "fig6_bf16_merge.png")

# ---------- 7. parameter budget
g = summary["base_params_by_group"]
budget = pd.DataFrame({
    "group": ["MLP projections", "Attention projections", "Embeddings", "LM head", "LoRA adapters (trained)", "RMSNorm weights"],
    "params": [g["mlp_proj"], g["attention_proj"], g["embeddings"], g["lm_head"], summary["lora_params"], g["rmsnorm"]],
})
fig, ax = plt.subplots(figsize=(9, 3.6))
colors = ["#9aa7b4"] * 4 + ["#dd8452", "#9aa7b4"]
sns.barplot(data=budget, y="group", x="params", palette=colors, hue="group", legend=False, ax=ax)
ax.set_xscale("log")
ax.set_xlabel("number of parameters (log scale)")
ax.set_ylabel("")
for i, p in enumerate(budget.params):
    ax.text(p * 1.15, i, f"{p / 1e6:,.1f}M", va="center", fontsize=9)
ax.set_xlim(1e5, 3e10)
ax.set_title("Only the orange bar was trained; everything grey is untouched", fontsize=11)
save(fig, "fig7_params.png")

# ---------- numbers for the text
by_mod = df.groupby("module").agg(
    rel=("rel_change_pct", "mean"), top1=("energy_top1_pct", "mean"), erank=("effective_rank", "mean"),
    changed=("bf16_elements_changed_pct", "mean"), cos=("bf16_merge_cosine", "mean"),
    dmean=("delta_mean_abs", "mean"), wmean=("w_mean_abs", "mean"),
).reindex(MODULES)
glob_rel = 100 * np.sqrt((df.delta_fro ** 2).sum() / (df.w_fro ** 2).sum())
top5 = df.nlargest(5, "rel_change_pct")[["layer", "module", "rel_change_pct"]]
bot5 = df.nsmallest(5, "rel_change_pct")[["layer", "module", "rel_change_pct"]]
early = df[df.layer < 8].rel_change_pct.mean()
late = df[df.layer >= 24].rel_change_pct.mean()
attn_rel = df[df.block == "self_attn"].rel_change_pct.mean()
mlp_rel = df[df.block == "mlp"].rel_change_pct.mean()
total = summary["base_total_params"]
lora = summary["lora_params"]
ta = summary["training_args"]
n_train = int(4800 * 0.92)


def fig_b64(name):
    return base64.b64encode((FIGS / name).read_bytes()).decode()


def table_rows(frame):
    return "".join(f"<tr><td dir='ltr'>layer {int(r.layer)} · {r.module}</td><td dir='ltr'>{r.rel_change_pct:.2f}%</td></tr>" for r in frame.itertuples())


mod_rows = "".join(
    f"<tr><td dir='ltr'>{m}</td><td dir='ltr'>{r.rel:.2f}%</td><td dir='ltr'>{r.top1:.0f}%</td><td dir='ltr'>{r.erank:.1f}</td>"
    f"<td dir='ltr'>{r.changed:.0f}%</td><td dir='ltr'>{r.cos:.3f}</td></tr>"
    for m, r in by_mod.iterrows()
)

html = r"""<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><title>مقارنة أوزان النموذج قبل وبعد التدريب</title>
<style>
@page { size: A4; margin: 14mm 13mm 16mm 13mm; }
body { font-family: 'Segoe UI', Tahoma, Arial, sans-serif; color: #17222b; font-size: 11pt; line-height: 1.75; }
h1 { font-size: 21pt; margin: 0 0 2mm; color: #0e3b36; }
.sub { color: #55655f; margin: 0 0 6mm; font-size: 10.5pt; }
h2 { font-size: 14pt; margin: 8mm 0 3mm; color: #0e3b36; border-bottom: 2px solid #0e3b36; padding-bottom: 1mm; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 5mm 0 1.5mm; break-after: avoid; }
.box { background: #f3f5f1; border: 1px solid #d5dccf; border-radius: 8px; padding: 3mm 4mm; margin: 0 0 3mm; break-inside: avoid; }
.key { background: #eaf4ee; border: 1px solid #b9d9c4; border-radius: 8px; padding: 3mm 4mm; margin: 0 0 3mm; break-inside: avoid; }
.warn { background: #fff6e0; border: 1px solid #ecd9a0; border-radius: 8px; padding: 3mm 4mm; margin: 3mm 0; break-inside: avoid; font-size: 10.5pt; }
table { border-collapse: collapse; width: 100%; font-size: 10pt; margin: 2mm 0 3mm; }
td, th { border: 1px solid #c9d1c4; padding: 1.6mm 2.6mm; text-align: start; vertical-align: top; }
th { background: #e6ece1; }
figure { margin: 2mm 0 4mm; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; height: auto; }
figcaption { font-size: 10pt; color: #55655f; margin-top: 1mm; text-align: start; }
ul { margin: 1mm 0; padding-inline-start: 6mm; }
code, .en { direction: ltr; unicode-bidi: embed; font-family: Consolas, monospace; font-size: 9.5pt; }
.pb { break-before: page; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 4mm; }
</style></head><body>

<h1>مقارنة أوزان النموذج قبل وبعد التدريب</h1>
<p class="sub">قبل التدريب: ALLaM-7B-Instruct من سدايا (SDAIA) &nbsp;|&nbsp; بعد التدريب: آخر تدريب لنا (إسناد v5) &nbsp;|&nbsp; القياس: كل الأوزان في ٢٢٤ طبقة مدرَّبة</p>

<h2>الخلاصة</h2>
<div class="key"><ul>
<li><b>التدريب غيّر __LORA_PCT__٪ فقط من النموذج.</b> درّبنا __LORA_M__ مليون رقم من أصل __TOTAL_B__ مليار. الباقي (__REST_PCT__٪) هو نفس أوزان سدايا حرفيًا، بدون أي تغيير.</li>
<li><b>الـ Bias: ما فيه أي bias في النموذج أصلًا، لا قبل ولا بعد.</b> ALLaM بنية Llama بدون bias (<code>attention_bias=false</code>)، وفحصت كل أرقام النموذج (٢٩١ مصفوفة): صفر مصفوفات bias. وإعداد التدريب <code>bias="none"</code>، وملف التدريب (٤٤٨ مصفوفة) ما فيه أي bias. فالفرق في الـ bias = صفر. وأوزان التطبيع (RMSNorm) ما اتغيّرت (مجمّدة).</li>
<li><b>حجم التغيير في الأوزان:</b> متوسط __REL_MEAN__٪ من حجم الوزن نفسه (أصغر تغيير __REL_MIN__٪ وأكبر __REL_MAX__٪). تغيير صغير جدًا، لكنه كافٍ يغيّر سلوك النموذج في المهمة اللي دربناه عليها.</li>
<li><b>وين صار التغيير؟</b> الطبقات العليا (٢٤–٣١) تغيّرت أكثر (__LATE__٪) من الطبقات الأولى (٠–٧: __EARLY__٪). وأكثر نوع تغيّر هو <code>gate_proj</code> (__GATE__٪)، وأقله <code>down_proj</code> (__DOWN__٪).</li>
<li><b>التغيير يتركز في اتجاهات قليلة:</b> في <code>gate_proj</code> حوالي __GATE_TOP1__٪ من التغيير في اتجاه واحد فقط، مع إن التدريب سمح بـ ١٦ اتجاه. معناه ممكن نجرّب رتبة (rank) أصغر مستقبلًا بدون خسارة كبيرة (يحتاج تجربة تأكيد).</li>
</ul></div>

<h2>١. ماذا قارنّا بالضبط؟</h2>
<table>
<tr><th></th><th>قبل التدريب</th><th>بعد التدريب</th></tr>
<tr><td>النموذج</td><td class="en">ALLaM-7B-Instruct (SDAIA)</td><td>نفسه + LoRA adapter (إسناد v5)</td></tr>
<tr><td>الأوزان</td><td>__TOTAL_PARAMS__ رقم (bf16)</td><td>نفس الأوزان + تعديل على __LORA_PARAMS__ رقم إضافي</td></tr>
<tr><td>الأجزاء اللي تتدرب</td><td>-</td><td>٧ مصفوفات في كل طبقة × ٣٢ طبقة = ٢٢٤ مصفوفة: <span class="en">q, k, v, o, gate, up, down</span></td></tr>
<tr><td>الـ Bias</td><td>لا يوجد</td><td>لا يوجد (<code>bias="none"</code>)</td></tr>
<tr><td>إعدادات التدريب</td><td>-</td><td dir="ltr" style="text-align:left">r=16, alpha=32, dropout=0.05, lr=__LR__, __EPOCHS__ epoch, effective batch 16, __N_TRAIN__ train examples (__N_STEPS__ steps), 4-bit QLoRA</td></tr>
</table>
<div class="box">
<b>كيف نحسب "الأوزان بعد التدريب"؟</b> LoRA ما يعدّل الوزن مباشرة، يضيف عليه تعديلًا صغيرًا:
<div class="en" style="text-align:center;margin:2mm 0;">W<sub>after</sub> = W<sub>before</sub> + (alpha / r) · B · A &nbsp;&nbsp; (alpha / r = __SCALE__)</div>
فالفرق بين قبل وبعد هو بالضبط <span class="en">ΔW = (alpha/r)·B·A</span>، وهذا اللي قسناه لكل مصفوفة. قبل التدريب B تساوي صفر، فالنموذج قبل التدريب = نموذج سدايا تمامًا.
</div>

<figure><img src="data:image/png;base64,__FIG7__"><figcaption>الشكل ١: ميزانية الأرقام. اللي تدرّب هو الأعمدة البرتقالية فقط (<span class="en">__LORA_M__M</span>)، وكل الباقي لم يُمس.</figcaption></figure>

<h2>٢. الـ Bias</h2>
<div class="box">
<ul>
<li>الـ bias رقم إضافي يُضاف بعد كل ضرب مصفوفات (<span class="en">y = Wx + b</span>). نماذج Llama، ومنها ALLaM، تعمل بدونه.</li>
<li>فحص أسماء كل المصفوفات في ملفات سدايا الثلاثة: <b>٢٩١ مصفوفة، صفر منها اسمه ينتهي بـ <span class="en">bias</span></b>.</li>
<li>فحص ملف التدريب (<span class="en">adapter_model.safetensors</span>): ٤٤٨ مصفوفة (A و B لـ ٢٢٤ طبقة)، <b>صفر منها bias</b>.</li>
<li>أوزان RMSNorm (__RMS__ رقم) كانت مجمّدة، فما اتغيّرت أيضًا.</li>
<li>الخلاصة: <b>الفرق في الـ bias قبل وبعد = صفر</b>، وكل التغيير صار في الأوزان.</li>
</ul>
</div>
<div class="warn">لو تقصد بـ "bias" الانحياز في المخرجات (مثل تكرار نفس المهام أو الأسماء)، هذا يُقاس بتجربة النموذج على أمثلة، مو من الأوزان. هذا موجود في تقارير الاختبار على أمثلة جديدة (<span class="en">heldout_generalization_report_v5.md</span>). أقدر أجمعه في ملف مقارنة لو تبي.</div>

<h2 class="pb">٣. الأوزان: وين وكم تغيّرت؟</h2>
<h3>٣.١ خريطة حرارية (heatmap): كل طبقة وكل مصفوفة</h3>
<p>كل خانة = مصفوفة وحدة. اللون الأغمق = تغيّرت أكثر نسبةً لحجمها. القراءة السريعة: عمود <span class="en">gate_proj</span> (الخامس من اليسار) أغمق من البقية، والصفوف السفلية (الطبقات العليا) أغمق من العلوية.</p>
<figure><img src="data:image/png;base64,__FIG1__" style="max-height:188mm;"><figcaption>الشكل ٢: نسبة تغيّر كل مصفوفة (<span class="en">‖ΔW‖ / ‖W‖</span>، بالمئة).</figcaption></figure>

<h3 class="pb">٣.٢ التغيير حسب العمق</h3>
<figure><img src="data:image/png;base64,__FIG2__"><figcaption>الشكل ٣: التغيير يزيد تدريجيًا كل ما اقتربنا من مخرجات النموذج (الطبقات العليا)، وغالبًا لأن الطبقات القريبة من المخرجات مرتبطة أكثر بصياغة الإجابة (تفسير شائع في هذا النوع من التدريب، ما اختبرناه هنا).</figcaption></figure>

<h3>٣.٣ حسب نوع المصفوفة</h3>
<figure><img src="data:image/png;base64,__FIG3__"><figcaption>الشكل ٤: توزيع نسبة التغيير لكل نوع (كل نقطة = طبقة). متوسط الانتباه (attention) __ATTN__٪ مقابل الشبكة الداخلية (MLP) __MLP__٪.</figcaption></figure>

<table>
<tr><th>النوع</th><th>متوسط التغيير</th><th>طاقة أهم اتجاه</th><th>الرتبة الفعلية (من ١٦)</th><th>أرقام تغيّرت بعد الدمج في bf16</th><th>تطابق الاتجاه بعد الدمج</th></tr>
__MOD_ROWS__
</table>

<div class="two">
<div><h3>أكثر ٥ مصفوفات تغيّرًا</h3><table>__TOP5__</table></div>
<div><h3>أقل ٥ مصفوفات تغيّرًا</h3><table>__BOT5__</table></div>
</div>

<h3 class="pb">٣.٤ حجم التغيير مقابل حجم الوزن</h3>
<figure><img src="data:image/png;base64,__FIG4__"><figcaption>الشكل ٥: التغيير النموذجي أصغر من الوزن النموذجي بمقدار <b>__RATIO__ مرة</b> تقريبًا (مثال: طبقة ١٥، <span class="en">q_proj</span>). يعني التدريب "هزّ" الأوزان هزّة خفيفة، ما غيّرها.</figcaption></figure>

<h3>٣.٥ كم اتجاه استخدم التدريب؟</h3>
<figure><img src="data:image/png;base64,__FIG5__"><figcaption>الشكل ٦: سمحنا بـ ١٦ اتجاه (rank). لكن في <span class="en">gate_proj</span> اتجاه واحد يفسّر أغلب التغيير، وفي <span class="en">q/k_proj</span> حوالي النصف. الاستنتاج: الرتبة ١٦ أكبر من المطلوب لبعض المصفوفات.</figcaption></figure>

<h2 class="pb">٤. ماذا يصير عند دمج التدريب داخل أوزان bf16؟</h2>
<p>لو دمجنا التدريب في الأوزان لينتج نموذج واحد (بدل ملف منفصل)، الأرقام تنقرّب لدقة bf16 (حوالي ٢–٣ خانات عشرية). التغيير أصغر من الأرقام نفسها، فبعضه يضيع بالتقريب.</p>
<figure><img src="data:image/png;base64,__FIG6__"><figcaption>الشكل ٧: بعد الدمج، من __CH_MIN__٪ إلى __CH_MAX__٪ من الأرقام تتغير فعلًا. واتجاه التغيير الأصلي يبقى محفوظ بنسبة (cosine) من __COS_MIN__ إلى __COS_MAX__، يعني الدمج في bf16 <b>مقبول</b> وما يخرّب التدريب بشكل مؤثر.</figcaption></figure>
<div class="box">ملاحظة: سيرفر الاستضافة يحمّل التدريب كملف منفصل فوق النموذج، فما يتأثر بهذا التقريب أصلًا. هذا القسم يهمنا فقط لو قررنا ندمج ونصدّر نموذج واحد.</div>

<h2>٥. هل seaborn مفيدة هنا؟</h2>
<div class="key">
<b>نعم، وانستخدمت فعلًا في كل الرسوم.</b> نصّبتها في بيئة منفصلة (<span class="en">seaborn 0.13.2</span>) عشان ما نلمس بيئة التدريب.
<ul>
<li><b>الخريطة الحرارية (<span class="en">heatmap</span>)</b>: أنفع استخدام. ٢٢٤ رقم (٣٢ طبقة × ٧ أنواع) ما تنقرا في جدول، لكن بالألوان تشوف النمط في ثانية. عملناها بسطر واحد.</li>
<li><b>الصندوق مع النقاط (<span class="en">boxplot + stripplot</span>)</b> والخط مع الشريط (<span class="en">lineplot</span> مع الانحراف المعياري): seaborn تحسبهم وحدها من الجدول، بدون كود إحصاء يدوي.</li>
<li><b>المدرّج التكراري (<span class="en">histplot</span>)</b> بمقياس لوغاريتمي: خلّى مقارنة حجم الوزن بحجم التغيير ممكنة على رسم واحد.</li>
<li>ما احتجناها لـ: الحساب نفسه (يتم بـ <span class="en">torch</span>)، ولا للصور المعقدة. هي فوق <span class="en">matplotlib</span>، فنفس النتيجة ممكنة بدونها لكن بكود أطول وألوان أسوأ.</li>
</ul>
</div>

<h2>٦. حدود هذي المقارنة</h2>
<ul>
<li>هذا يقيس <b>كم</b> و<b>وين</b> تغيّرت الأوزان، مو <b>جودة</b> النموذج. الجودة تُقاس باختبارات الأمثلة الجديدة (تقارير <span class="en">heldout</span>).</li>
<li>التدريب والتشغيل على Modal يستخدمون نسخة مضغوطة بـ ٤-bit (QLoRA) من نموذج سدايا ويضيفون التدريب فوقها، لكن المقارنة هنا مع أوزان سدايا الأصلية بدقة bf16. فالأوزان اللي تشتغل فعليًا تختلف عنها قليلًا بسبب الضغط، وهذا الفرق مو من التدريب.</li>
<li>النسبة المئوية صغيرة لأن LoRA يحدّث جزء صغير جدًا، فلا تعني إن النموذج "تغيّر قليلًا" في السلوك. أثر التدريب على المخرجات يُحكم عليه من الاختبارات.</li>
<li>قارنّا آخر تدريب (v5) فقط مع نموذج سدايا. v4 وبقية المحاولات ما دخلت هنا.</li>
</ul>

<h2>٧. إعادة التشغيل</h2>
<div class="box en" style="font-size:9.5pt;">
weights_stage1.py &nbsp;(project venv: torch + safetensors) &rarr; per_module.csv, singular_values.csv, summary.json<br>
weights_stage2.py &nbsp;(plot venv: seaborn + matplotlib + pandas) &rarr; figs/*.png + this report<br>
Total run time: about 20 seconds for stage 1 on an RTX 4070 Ti.
</div>
</body></html>"""

repl = {
    "__LORA_PCT__": f"{summary['lora_fraction_pct']:.2f}",
    "__REST_PCT__": f"{100 - summary['lora_fraction_pct']:.2f}",
    "__LORA_M__": f"{lora / 1e6:.1f}",
    "__TOTAL_B__": f"{total / 1e9:.2f}",
    "__TOTAL_PARAMS__": f"{total:,}",
    "__LORA_PARAMS__": f"{lora:,}",
    "__REL_MEAN__": f"{df.rel_change_pct.mean():.2f}",
    "__REL_MIN__": f"{df.rel_change_pct.min():.2f}",
    "__REL_MAX__": f"{df.rel_change_pct.max():.2f}",
    "__EARLY__": f"{early:.2f}",
    "__LATE__": f"{late:.2f}",
    "__GATE__": f"{by_mod.loc['gate_proj', 'rel']:.2f}",
    "__DOWN__": f"{by_mod.loc['down_proj', 'rel']:.2f}",
    "__GATE_TOP1__": f"{by_mod.loc['gate_proj', 'top1']:.0f}",
    "__ATTN__": f"{attn_rel:.2f}",
    "__MLP__": f"{mlp_rel:.2f}",
    "__SCALE__": f"{summary['lora']['scale']:.0f}",
    "__RMS__": f"{summary['base_params_by_group']['rmsnorm']:,}",
    "__LR__": f"{float(ta['learning_rate']):g}",
    "__EPOCHS__": ta["num_train_epochs"],
    "__N_TRAIN__": f"{n_train:,}",
    "__N_STEPS__": "276",
    "__RATIO__": f"{ratio:.0f}",
    "__CH_MIN__": f"{by_mod.changed.min():.0f}",
    "__CH_MAX__": f"{by_mod.changed.max():.0f}",
    "__COS_MIN__": f"{by_mod.cos.min():.2f}",
    "__COS_MAX__": f"{by_mod.cos.max():.2f}",
    "__MOD_ROWS__": mod_rows,
    "__TOP5__": table_rows(top5),
    "__BOT5__": table_rows(bot5),
}
for i, n in enumerate(["fig1_heatmap", "fig2_by_layer", "fig3_by_module", "fig4_distribution", "fig5_rank_usage", "fig6_bf16_merge", "fig7_params"], 1):
    repl[f"__FIG{i}__"] = fig_b64(n + ".png")
# fig numbering in the text follows the report order, filenames follow creation order
html = re.sub("(__[A-Z0-9_]+__)" + chr(0x66A), lambda m: "<span class='en'>" + m.group(1) + "%</span>", html)
for token, value in repl.items():
    html = html.replace(token, value)

html_path = OUT / "esnad_model_weights_comparison.html"
html_path.write_text(html, encoding="utf-8")
pdf_path = OUT / "esnad_model_weights_comparison.pdf"
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", html_path.as_uri()],
               check=True, capture_output=True, timeout=240)
print("html", html_path.stat().st_size // 1024, "KB | pdf", pdf_path.stat().st_size // 1024, "KB")
print("global rel change %:", round(glob_rel, 3), "| ratio:", round(ratio))
