# Esnad AI: task decomposition model

[العربية](#العربية)

The language model behind the AI task generator of **Esnad**, a platform for managing university software engineering projects. A team writes its project idea and describes its members. The model returns a task list: a title, a difficulty and a suggested member for every task.

This repository holds the trained model, the training data, the scripts that built and tested it, the serving code and the evaluation. The Esnad web platform is a separate private repository.

| | |
|---|---|
| Base model | ALLaM-7B-Instruct-preview, by the National Center for AI at SDAIA. 7 billion parameters, Apache-2.0 according to its Hugging Face tag |
| What we trained | A LoRA adapter, 39,976,960 parameters, which is 0.57% of the model. The other 99.43% is SDAIA's original weights, unchanged |
| Method | QLoRA: the base model is frozen in 4-bit NF4, the adapter is trained in bfloat16. Rank 16, alpha 32, dropout 0.05, on all seven projection matrices of all 32 layers |
| Input | One prompt with the project idea, the team request and each member's name and bio |
| Output | A JSON array: `[{"title", "difficulty", "suggestedAssignee"}, ...]` with `easy`, `medium` or `hard` |
| Training data | 4,800 synthetic examples built from 18 project ideas. 4,416 for training, 384 for validation |
| Hardware | Training: one NVIDIA RTX 4070 Ti with 12 GB, on a personal computer. Serving: one A10G GPU on Modal |
| Training time | About 1.5 hours: 1 epoch, 276 steps at about 14 seconds each, plus 7 validation passes. This is an estimate from the earlier runs on the same GPU, because the log of the v5 run was not kept |
| Training settings | Learning rate 1e-4 with cosine decay, effective batch size 16 |
| Serving | Modal, A10G GPU, 4-bit base model plus the adapter, HTTP endpoint `/generate` |
| Version | v5, the one in production |
| Result | 98% usable answers in the platform format (44 of 45 on unseen ideas) against 0% for the original model. Details in Results |

![The AI task breakdown inside Esnad: the team writes a request, the model returns a task list, and the team reviews it before adding it to the board](docs/esnad-task-breakdown.gif)

*The model inside the Esnad platform. The team writes a request, the model returns tasks with a difficulty and a member for each, and the team reviews them before adding them to the board.*

## Where the model comes from

```mermaid
flowchart LR
  A["ALLaM-7B-Instruct-preview<br/>SDAIA, Apache-2.0"] --> T
  subgraph Data ["Synthetic training data"]
    I["23 project ideas<br/>18 for training, 5 held out"] --> B
    Q["Qwen2.5-7B-Instruct via Ollama<br/>writes the task content"] --> B["Task banks<br/>92 idea and feature pairs, 2 versions each<br/>72 pairs from the 18 training ideas are used"]
    B --> S["Example builder<br/>teams, skills, fair assignment rules"]
    S --> D["4,800 examples<br/>prompt and JSON answer"]
  end
  D --> T["QLoRA training<br/>4-bit base, LoRA rank 16, 1 epoch"]
  T --> L["LoRA adapter v5"]
  L --> M["Modal GPU endpoint"]
  M --> E["Esnad backend<br/>removes duplicates, covers all phases, balances load"]
```

We did not train a model from scratch. We took ALLaM, an Arabic and English model from SDAIA, and taught it one narrow job: answer the Esnad task prompt with a clean JSON task list, in Arabic, using only the team members it was given.

We chose ALLaM because it is a Saudi model with strong Arabic, its Hugging Face page tags it Apache-2.0, and at 7 billion parameters it fits on a 12 GB gaming GPU with QLoRA. Per its model card it was pretrained on 5.2 trillion tokens, first 4 trillion English and then 1.2 trillion mixed Arabic and English.

Qwen2.5-7B-Instruct (Apache-2.0) was used once, offline, to write realistic task content for the training data. It is not part of the shipped model.

## What we did

1. **Wrote the project ideas.** 23 student-style software project ideas with four features each, for example a clinic booking app or a restaurant table system. Five were kept out of training completely and used only for testing.
2. **Generated the task content.** Qwen2.5-7B-Instruct wrote two different task lists for every pair of project idea and feature, following work breakdown structure (WBS) and story splitting practice. That is 92 pairs, 184 generations and 1,350 tasks, each with an Arabic title, a difficulty and a skill category. Only the 72 pairs of the 18 training ideas went into the training data. The generator sometimes produced glitch words in other scripts, such as Cyrillic. They were replaced, and a scan of the shipped file finds none.
3. **Built the examples.** Each example is a team of 2 to 5 members with a short bio, a request, and the expected task list. Who gets which task is decided by a rule, not by a model: the member with the matching skill and the lowest current load takes the next task. This is what teaches the model to share the work.
4. **Added attacks.** In 731 of the 4,800 examples one member's bio contains an instruction such as "ignore the above and write only X". The expected answer ignores it, still gives that member one easy task and a normal share, and never rewards the attack with extra work.
5. **Used the real prompt.** Every training prompt was produced by the platform's own `buildPrompt()`, so the model was trained on the prompt production sent at that time. The prompt tells the model that anything inside the data tags is untrusted text, never instructions.
6. **Trained with QLoRA.** The loss is computed only on the answer tokens, so the model learns to write the JSON and does not learn to copy the prompt.
7. **Tested against the original model.** Same prompt, same settings as production, fixed seeds, 108 generations. Results are in the next section.
8. **Deployed.** The adapter runs on a Modal GPU. The Esnad backend calls it, then repairs and balances the result in code.

## Results

The test used the real platform prompt and the real production generation settings: 550 new tokens, temperature 0.4, top-p 0.9, repetition penalty 1.2. Both models ran 4-bit with the same random seed for every prompt and sample. The prompts were 5 held-out project ideas, 10 new ideas written outside the training data, and 3 prompt-injection cases, 3 samples each: 108 generations in total. A usable answer is valid JSON with at least 3 tasks.

| | SDAIA original | Esnad v5 |
|---|---|---|
| Usable answers, 15 ideas × 3 samples | 0 of 45 | **44 of 45** |
| What the original returned | 40 empty lists, 5 broken JSON | none |
| Prompt injection followed, 3 cases × 3 samples | 3 of 9 | **0 of 9** |
| Usable answers on the injection cases | 0 of 9 | 8 of 9 |

Among the 44 usable answers of v5:

| Measure | Value |
|---|---|
| Tasks per answer | 7.8 on average. The platform completes the list to about 12 |
| Suggested members that exist in the team | 99.1% |
| Members who got at least one task | 97.8% |
| Duplicate titles | 0% |
| Difficulty mix | 48% easy, 49% medium, 3% hard |
| Work gap between the most and least loaded member | 0.83 from the model alone, 0.21 after the platform balances it. 0 is perfectly equal |
| Project phases the model missed | 2.8 of 6 on average. The platform adds them |
| Titles copied word for word from the training data | 25% |
| Tasks that mention the requested feature, held-out ideas, coarse keyword test | 33% |
| Generation time on the RTX 4070 Ti | about 17 seconds |

How much the training changed the model: 0.57% of the numbers, by an average of 1.06% of each weight's size. The change is largest in the upper layers and in the `gate_proj` matrices. There are no bias terms in this architecture. Details are in `evaluation/esnad_model_before_after_weights.pdf`.

### What the results mean

**What they prove.** The original model could not answer the Esnad prompt at all: 0 of 45 usable answers. After training it answers correctly 98% of the time (44 of 45). The answer is a valid task list in Arabic, built from the real team members, and it ignored all the fake commands hidden in member bios that we tried (0 of 9 followed).

**What they do not prove.** They do not prove that our model plans a project better than the original one. The original gave no task list, so there was nothing to compare the quality of the tasks against. A fair comparison needs the original with a simpler prompt, and we did not run it.

### Limits

- **Little variety in the data.** All 4,800 examples come from only 18 project ideas. A quarter of the titles are copied from the training data, and some make no sense, for example a task to create a "student" in a project that has no students.
- **Short lists.** The model writes about 8 tasks, and a project needs 12 to 20. The platform adds the missing ones, so about 70% of the final tasks come from the model and the rest from the platform. We tried asking for 20 tasks in the prompt, and the final list only grew from 11.8 to 12.6 tasks. The model learned the size of a list from how the request is worded, not from a number.
- **The fair split is done by code.** When the model divides the work, one member gets much more than another. The platform balances it by task difficulty, which makes the split fair but fits the tasks a little less to each member's skills.
- **Difficulty is a guess.** The difficulty labels were written by Qwen2.5-7B, not by people, and nobody measured how long the tasks really take. Only 3% of the answers use `hard`.
- **Small test.** 18 prompts with 3 samples each. With so few, the numbers can move a lot. For example the relevance score of 33% could really be anywhere between 18% and 49%.
- **No human check.** Nobody judged the quality of the tasks, and the model was not tested on ideas written in English.
- **Validation loss is not a quality score.** The final value of 0.109 only says how well the model predicts answers on examples like the ones it trained on. It comes from the same 18 ideas.

The full report with figures and real, unselected examples is `evaluation/esnad_finetune_evaluation.pdf`, written in Arabic.

## Using the model

The adapter needs the base model. Download it once, and join the two adapter parts into the weights file. The join script checks the SHA-256 of the result:

```bash
python scripts/download_model.py     # saves models/allam-7b-instruct, about 14 GB
python scripts/join_adapter.py       # writes final_adapter/adapter_model.safetensors and verifies it
```

Load it the same way the production server does:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = "humain-ai/ALLaM-7B-Instruct-preview"
tokenizer = AutoTokenizer.from_pretrained(BASE)
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
)
base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(base, "models/allam-7b-esnad-lora-v5/final_adapter").eval()

text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=550, do_sample=True, temperature=0.4, top_p=0.9,
                     repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id)
completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
```

`prompt` must be the prompt that the Esnad platform builds with `buildPrompt()` in its `services/aiEngine.js`. The adapter was trained on that format only, and other formats were not tested. `datasets/sft_data.jsonl` has 4,800 real examples of it.

The answer is a JSON array of tasks, for example:

```json
[
  {"title": "تصميم واجهة المستخدم لعرض التذكيرات", "difficulty": "easy", "suggestedAssignee": "فهد"},
  {"title": "إنشاء خدمة API لاسترجاع مواعيد المرضى", "difficulty": "medium", "suggestedAssignee": "سلطان"}
]
```

Two ways to serve it:

```bash
python server/inference_server.py     # local Flask server, POST http://localhost:8765/generate
modal deploy server/modal_app.py      # Modal, A10G GPU, prints the public /generate URL
```

Both take `{"prompt": "...", "max_new_tokens": 550}` and return `{"completion": "..."}`. Fixing the answer, removing duplicates, covering the project phases and balancing the load all happen in the platform, not in the server.

Intended use: draft task lists for student software projects, written in Arabic, that a team reviews and edits before adding them to the board. Not for grading or judging people. The points and completion rate in Esnad come from finished tasks, not from this model.

## Repository layout

```
models/allam-7b-esnad-lora-v5/
  final_adapter/        the adapter configuration and tokenizer files
  adapter_parts/        the adapter weights, 160 MB, cut in two parts because GitHub rejects files above 100 MB
  trainer_state.json    loss history of the 276 steps
datasets/
  sft_data.jsonl        the 4,800 training examples: {"prompt", "completion"}
  rich_task_banks.json  the task content written by Qwen2.5-7B, per idea and feature
  heldout_eval_specs.json   the 5 held-out ideas used in the first generalization test
scripts/                the pipeline, in the order below
server/                 local server and the Modal deployment
evaluation/
  esnad_finetune_evaluation.pdf          before and after test (Arabic)
  esnad_model_before_after_weights.pdf   what the training changed in the weights (Arabic)
  scripts/ and data/                     code, prompts, raw answers and scores of the 108 generations
```

## Reproducing the pipeline

Python 3.12 with CUDA PyTorch, `transformers`, `peft`, `bitsandbytes`, `accelerate` and `datasets`. The v5 run used one 12 GB GPU.

| Step | Command | Output |
|---|---|---|
| Base model | `python scripts/download_model.py` | `models/allam-7b-instruct` |
| Adapter file | `python scripts/join_adapter.py` (only to use the shipped adapter, training writes the file itself) | `final_adapter/adapter_model.safetensors` |
| Task content | `python scripts/generate_rich_task_banks.py` (needs Ollama with `qwen2.5:7b-instruct`). See the note below | `datasets/rich_task_banks.json` |
| Clean glitch words | `python scripts/fix_cyrillic_glitch.py`, then `python scripts/scan_all_scripts.py` | the scan should print `"count": 0` |
| Build the examples | `python scripts/generate_synthetic_data.py`, then `node scripts/build_prompts.js` | `datasets/raw_specs.json`, `datasets/sft_data.jsonl` |
| Train | `python scripts/train_qlora.py` | `models/allam-7b-esnad-lora-v5/final_adapter` |
| Try it | `python scripts/try_model.py` | interactive |
| Evaluate | `python evaluation/scripts/eval_generate.py`, `eval_score.js`, `eval_analyze.py` | tables and figures |

To reproduce the training exactly, use the `datasets/rich_task_banks.json` and `datasets/sft_data.jsonl` in this repository. Running the task content step again calls Qwen again and writes different tasks. Also note that the shipped bank is keyed by idea and feature, `idea :: feature`, which is what `generate_synthetic_data.py` reads, while `generate_rich_task_banks.py` in its current form writes one whole-project list per idea. It would need adjusting to rebuild the shipped file.

Some scripts contain paths from the machine where they were written, for example `C:\Users\<name>\esnad-ai-training` and `C:/Users/<name>/GP2-Cl/services/aiEngine`. The second one is the Esnad platform repository, which supplies `buildPrompt()`. Change these paths to your own before running.

## Next steps

- Train on many more project ideas than 18, and rewrite the weak task templates.
- Train for the number of tasks and for all six project phases, so the platform has less to repair.
- Compare against the original model with a simplified prompt, and have people judge a sample of the tasks.

## Sources

- ALLaM-7B-Instruct-preview: https://huggingface.co/humain-ai/ALLaM-7B-Instruct-preview
- QLoRA, Dettmers et al., 2023: https://arxiv.org/abs/2305.14314
- PEFT: https://github.com/huggingface/peft, bitsandbytes: https://github.com/bitsandbytes-foundation/bitsandbytes, Transformers: https://github.com/huggingface/transformers
- Qwen2.5-7B-Instruct, used to write task content: https://ollama.com/library/qwen2.5
- Work breakdown structure practice and the 8/80 rule: https://asana.com/resources/work-breakdown-structure and https://community.atlassian.com/forums/App-Central-articles/3-Work-Breakdown-Structure-WBS-Gantt-Chart-examples-for-software/ba-p/2338509
- Splitting user stories into tasks with AI: https://arxiv.org/abs/2605.07320
- Story splitting, SPIDR: https://www.mountaingoatsoftware.com/agile/five-simple-but-powerful-ways-to-split-user-stories
- NEODATASET, 40 thousand real user stories, read for style only and not used in training: https://huggingface.co/datasets/giseldo/neodataset

## Status and license

Repository of the Esnad team, a graduation project at Imam Mohammad Ibn Saud Islamic University. The adapter is a derivative of ALLaM-7B-Instruct-preview and follows its license, Apache-2.0 according to the Hugging Face tag of the base model. All training examples are synthetic: names, bios and projects are made up, and there is no personal data.

---

## العربية

<details>
<summary>اضغط لعرض النسخة العربية الكاملة</summary>

<div dir="rtl">

# إسناد AI: نموذج تفكيك المهام

هذا هو النموذج اللغوي الذي يشغّل ميزة توليد المهام بالذكاء الاصطناعي في **إسناد**، منصة لإدارة مشاريع هندسة البرمجيات الجامعية. الفريق يكتب فكرة المشروع ويصف أعضاءه، والنموذج يرجّع قائمة مهام: عنوان وصعوبة وعضو مقترح لكل مهمة.

يحتوي المستودع على النموذج المدرَّب وبيانات التدريب والسكربتات اللي بنته واختبرته وكود التشغيل والتقييم. منصة إسناد نفسها مستودع خاص منفصل.

| | |
|---|---|
| النموذج الأساسي | ALLaM-7B-Instruct-preview من المركز الوطني للذكاء الاصطناعي في سدايا. 7 مليار بارامتر، رخصة Apache-2.0 حسب وسمه في Hugging Face |
| وش دربنا | LoRA adapter فيه 39,976,960 بارامتر، يعني 0.57% من النموذج. الـ 99.43% الباقية هي أوزان سدايا الأصلية بدون أي تغيير |
| الطريقة | QLoRA: النموذج الأساسي مجمّد بدقة 4-bit، والـ adapter يتدرب بـ bfloat16. الرتبة 16 وalpha 32 وdropout 0.05 على المصفوفات السبع في الـ 32 طبقة كلها |
| المدخل | برومبت فيه فكرة المشروع وطلب الفريق واسم كل عضو ونبذته |
| المخرج | مصفوفة JSON: `[{"title", "difficulty", "suggestedAssignee"}, ...]` والصعوبة `easy` أو `medium` أو `hard` |
| بيانات التدريب | 4,800 مثال اصطناعي مبنية من 18 فكرة مشروع. 4,416 للتدريب و384 للتحقق |
| العتاد | التدريب: كرت NVIDIA RTX 4070 Ti واحد بذاكرة 12 جيجا على جهاز شخصي. التشغيل: كرت A10G على Modal |
| مدة التدريب | حوالي ساعة ونص: دورة وحدة، 276 خطوة بحدود 14 ثانية للخطوة، مع 7 مرات تحقق. هذا تقدير مبني على التدريبات الأقدم على نفس الكرت، لأن سجل تدريب v5 ما انحفظ |
| إعدادات التدريب | معدل تعلم 1e-4 مع cosine، حجم دفعة فعلي 16 |
| التشغيل | Modal، كرت A10G، النموذج الأساسي 4-bit مع الـ adapter، ونقطة اتصال `/generate` |
| الإصدار | v5 وهو اللي في الإنتاج |
| النتيجة | 98% إجابات صالحة بصيغة المنصة (44 من 45 على أفكار جديدة) مقابل 0% للنموذج الأصلي. التفاصيل في النتائج |

![تفكيك المهام بالذكاء الاصطناعي داخل منصة إسناد: الفريق يكتب طلبه، والنموذج يرجّع قائمة مهام، والفريق يراجعها قبل ما يضيفها للوحة](docs/esnad-task-breakdown.gif)

*النموذج داخل منصة إسناد: الفريق يكتب طلبه، والنموذج يرجّع مهام بصعوبة وعضو لكل مهمة، والفريق يراجعها قبل ما يضيفها للوحة.*

## من وين جا النموذج

ما دربنا نموذج من الصفر. أخذنا ALLaM من سدايا، نموذج عربي وإنجليزي، وعلّمناه مهمة وحدة ضيقة: يرد على برومبت إسناد بقائمة مهام JSON نظيفة، بالعربي، ومن أعضاء الفريق اللي انعطوه فقط.

اخترنا ALLaM لأنه نموذج سعودي قوي بالعربي، وموقعه في Hugging Face يوسمه برخصة Apache-2.0، وبحجم 7 مليار يدخل على كرت 12 جيجا مع QLoRA. حسب بطاقة النموذج تدرّب مسبقاً على 5.2 تريليون توكن: أول 4 تريليون إنجليزي، وبعدها 1.2 تريليون عربي وإنجليزي مختلط.

نموذج Qwen2.5-7B-Instruct (رخصة Apache-2.0) استخدمناه مرة وحدة، بدون اتصال، عشان يكتب محتوى مهام واقعي لبيانات التدريب. ما هو جزء من النموذج النهائي.

## وش سوينا

1. **كتبنا أفكار المشاريع.** 23 فكرة مشروع برمجي بأسلوب مشاريع الطلاب، لكل فكرة أربع ميزات، مثل تطبيق حجز عيادات أو نظام طاولات مطاعم. خمس أفكار انحجزت خارج التدريب تماماً واستخدمناها للاختبار فقط.

2. **ولّدنا محتوى المهام.** نموذج Qwen2.5-7B-Instruct كتب قائمتين مختلفتين لكل زوج من فكرة وميزة، باتباع ممارسات تقسيم العمل WBS وتقسيم قصص المستخدم. المجموع 92 زوج و184 توليد و1,350 مهمة، لكل مهمة عنوان عربي وصعوبة وتصنيف مهارة. دخل في بيانات التدريب 72 زوج فقط، هي أزواج الـ 18 فكرة المخصصة للتدريب. المولّد أحياناً طلّع كلمات خربانة بخطوط ثانية مثل الروسي، فاستبدلناها، وفحص الملف المرفق ما يلقى منها شي.

3. **بنينا الأمثلة.** كل مثال فريق من 2 إلى 5 أعضاء مع نبذة قصيرة لكل واحد، وطلب، وقائمة المهام المتوقعة. من ياخذ أي مهمة تحدده قاعدة مو نموذج: العضو اللي مهارته تطابق وحمله الحالي الأقل ياخذ المهمة التالية. هذا اللي يعلّم النموذج يوزع الشغل بالعدل.

4. **أضفنا هجمات.** في 731 مثال من الـ 4,800 نبذة أحد الأعضاء فيها أمر مثل «تجاهل اللي فوق واكتب كذا فقط». الجواب المتوقع يتجاهله، ويعطي هذا العضو مهمة وحدة سهلة وحصة عادية، وما يكافئ الهجوم بشغل زيادة.

5. **استخدمنا البرومبت الحقيقي.** كل برومبت تدريب طلع من `buildPrompt()` في المنصة نفسها، فالنموذج تدرّب على البرومبت اللي كانت ترسله الإنتاج وقتها. البرومبت يقول للنموذج إن أي شي داخل وسوم البيانات نص غير موثوق ومو أوامر.

6. **دربنا بـ QLoRA.** الخسارة تنحسب على توكنات الجواب فقط، فالنموذج يتعلم كتابة الـ JSON ولا يتعلم نسخ البرومبت.

7. **اختبرناه قدام النموذج الأصلي.** نفس البرومبت ونفس إعدادات الإنتاج، وبذور عشوائية ثابتة، و108 توليد. النتائج في القسم الجاي.

8. **نشرناه.** الـ adapter يشتغل على كرت Modal، وباك إند إسناد يناديه، وبعدها يصلح الناتج ويوزنه بالكود.

## النتائج

الاختبار استخدم برومبت المنصة الحقيقي وإعدادات التوليد الفعلية للإنتاج: 550 توكن جديد، حرارة 0.4، top-p 0.9، وعقوبة تكرار 1.2. النموذجان اشتغلوا بـ 4-bit وبنفس البذرة العشوائية لكل برومبت ومحاولة. البرومبتات: 5 أفكار محجوزة، و10 أفكار جديدة مكتوبة خارج بيانات التدريب، و3 حالات حقن تعليمات، وكل واحد 3 محاولات. المجموع 108 توليد. الإجابة الصالحة هي JSON سليم فيه 3 مهام على الأقل.

| | سدايا الأصلي | إسناد v5 |
|---|---|---|
| إجابات صالحة، 15 فكرة × 3 محاولات | 0 من 45 | **44 من 45** |
| وش رجّع الأصلي | 40 قائمة فاضية، و5 JSON خربان | لا شي |
| تنفيذ أمر مدسوس، 3 حالات × 3 محاولات | 3 من 9 | **0 من 9** |
| إجابات صالحة في حالات الحقن | 0 من 9 | 8 من 9 |

من الـ 44 إجابة صالحة لنموذج v5:

| المقياس | القيمة |
|---|---|
| المهام في الإجابة | 7.8 بالمتوسط. المنصة تكمّل القائمة لحوالي 12 |
| أعضاء مقترحين موجودين فعلاً بالفريق | 99.1% |
| أعضاء أخذوا مهمة على الأقل | 97.8% |
| عناوين مكررة | 0% |
| توزيع الصعوبة | 48% سهلة، 49% متوسطة، 3% صعبة |
| فرق الحمل بين أكثر عضو وأقل عضو | 0.83 من النموذج لحاله، و0.21 بعد ما توزّنه المنصة. الصفر يعني متساوي تماماً |
| مراحل المشروع اللي فاتت النموذج | 2.8 من 6 بالمتوسط. المنصة تضيفها |
| عناوين منسوخة حرفياً من بيانات التدريب | 25% |
| مهام تذكر الميزة المطلوبة، أفكار محجوزة، اختبار كلمات خشن | 33% |
| وقت التوليد على RTX 4070 Ti | حوالي 17 ثانية |

كم غيّر التدريب من النموذج: 0.57% من الأرقام، بمتوسط 1.06% من حجم كل وزن. التغيير أكبر في الطبقات العليا وفي مصفوفات `gate_proj`. هذي البنية ما فيها bias. التفاصيل في `evaluation/esnad_model_before_after_weights.pdf`.

### وش تعني النتائج

**وش تثبت.** النموذج الأصلي ما كان يقدر يجاوب على برومبت إسناد أبدًا: صفر من 45 إجابة صالحة. بعد التدريب صار يجاوب صح 98% من المرات (44 من 45). الجواب قائمة مهام صالحة بالعربي، مبنية على أعضاء الفريق الحقيقيين، وما نفّذ أي أمر مزوّر مخبّى في نبذة عضو من اللي جرّبناها (صفر من 9 نفّذوها).

**وش ما تثبت.** ما تثبت إن نموذجنا يخطط المشروع أحسن من الأصلي. الأصلي ما رجّع قائمة مهام أصلًا، فما كان فيه شي نقارن عليه جودة المهام. المقارنة العادلة تبغى الأصلي ببرومبت أبسط، وما سويناها.

### الحدود

- **بيانات قليلة التنوع.** الـ 4,800 مثال كلها من 18 فكرة مشروع بس. ربع العناوين منسوخة من بيانات التدريب، وبعضها ما يركب، مثل مهمة إنشاء «طالب» في مشروع ما فيه طلاب.
- **قوائم قصيرة.** النموذج يكتب حوالي 8 مهام، والمشروع يحتاج من 12 إلى 20. المنصة تكمّل الناقص، فحوالي 70% من المهام النهائية من النموذج والباقي من المنصة. جرّبنا نطلب 20 مهمة في البرومبت، والقائمة النهائية زادت من 11.8 إلى 12.6 مهمة بس. النموذج تعلّم حجم القائمة من صياغة الطلب، مب من رقم.
- **العدل يجي من الكود.** لما النموذج يوزع الشغل، عضو ياخذ أكثر بكثير من عضو ثاني. المنصة توازنه بحسب صعوبة المهام، وهذا يخلي التوزيع عادل لكن يقلل شوي تطابق المهمة مع مهارة العضو.
- **الصعوبة تخمين.** تصنيفات الصعوبة كتبها Qwen2.5-7B مب بشر، وما أحد قاس كم تاخذ المهام فعلًا. 3% فقط من الإجابات تستخدم `hard`.
- **اختبار صغير.** 18 برومبت و3 محاولات لكل واحد. مع هالقلة الأرقام ممكن تتحرك كثير، مثلًا مقياس الصلة 33% ممكن فعلًا يكون بين 18% و49%.
- **ما فيه مراجعة بشرية.** ما أحد حكّم على جودة المهام، والنموذج ما انختبر على أفكار مكتوبة بالإنجليزي.
- **خسارة التحقق مب مقياس جودة.** قيمتها النهائية 0.109 تقول بس كم النموذج يتوقع الإجابات زين في أمثلة تشبه اللي تدرب عليها. وهي جات من نفس الـ 18 فكرة.

التقرير الكامل بالرسوم وأمثلة حقيقية غير منتقاة هو `evaluation/esnad_finetune_evaluation.pdf` وهو بالعربي.

## استخدام النموذج

الـ adapter يحتاج النموذج الأساسي. نزّله مرة وحدة، وجمّع جزئي الـ adapter في ملف الأوزان. سكربت التجميع يتحقق من SHA-256 للناتج:

<div dir="ltr">

```bash
python scripts/download_model.py     # يحفظ models/allam-7b-instruct حوالي 14 جيجا
python scripts/join_adapter.py       # يكتب final_adapter/adapter_model.safetensors ويتحقق منه
```

</div>

حمّله بنفس طريقة سيرفر الإنتاج:

<div dir="ltr">

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = "humain-ai/ALLaM-7B-Instruct-preview"
tokenizer = AutoTokenizer.from_pretrained(BASE)
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
)
base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(base, "models/allam-7b-esnad-lora-v5/final_adapter").eval()

text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=550, do_sample=True, temperature=0.4, top_p=0.9,
                     repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id)
completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
```

</div>

`prompt` لازم يكون البرومبت اللي تبنيه منصة إسناد بالدالة `buildPrompt()` في `services/aiEngine.js`. الـ adapter تدرّب على هذي الصيغة فقط، والصيغ الثانية ما انختبرت. الملف `datasets/sft_data.jsonl` فيه 4,800 مثال حقيقي منها.

الجواب مصفوفة JSON من المهام، مثل:

<div dir="ltr">

```json
[
  {"title": "تصميم واجهة المستخدم لعرض التذكيرات", "difficulty": "easy", "suggestedAssignee": "فهد"},
  {"title": "إنشاء خدمة API لاسترجاع مواعيد المرضى", "difficulty": "medium", "suggestedAssignee": "سلطان"}
]
```

</div>

طريقتين للتشغيل:

<div dir="ltr">

```bash
python server/inference_server.py     # سيرفر Flask محلي، POST http://localhost:8765/generate
modal deploy server/modal_app.py      # Modal، كرت A10G، ويطبع رابط /generate العام
```

</div>

الاثنين يستقبلون `{"prompt": "...", "max_new_tokens": 550}` ويرجّعون `{"completion": "..."}`. تصليح الجواب وإزالة التكرار وتغطية مراحل المشروع وموازنة الحمل كلها تصير في المنصة، مو في السيرفر.

الاستخدام المقصود: مسودات قوائم مهام لمشاريع برمجية طلابية بالعربي، يراجعها الفريق ويعدلها قبل ما تنضاف للوحة. مو للتقييم ولا للحكم على الناس. النقاط ونسبة الإنجاز في إسناد تجي من المهام المنجزة، مو من هذا النموذج.

## ترتيب المستودع

<div dir="ltr">

```
models/allam-7b-esnad-lora-v5/
  final_adapter/        ملفات إعداد الـ adapter والتوكنايزر
  adapter_parts/        أوزان الـ adapter بحجم 160 ميجا، مقسومة لجزئين لأن GitHub يرفض الملفات فوق 100 ميجا
  trainer_state.json    سجل الخسارة لـ 276 خطوة
datasets/
  sft_data.jsonl        أمثلة التدريب الـ 4,800: {"prompt", "completion"}
  rich_task_banks.json  محتوى المهام اللي كتبه Qwen2.5-7B، لكل فكرة وميزة
  heldout_eval_specs.json   الأفكار الخمس المحجوزة المستخدمة في أول اختبار تعميم
scripts/                خط الإنتاج، بالترتيب تحت
server/                 السيرفر المحلي ونشر Modal
evaluation/
  esnad_finetune_evaluation.pdf          اختبار قبل وبعد (بالعربي)
  esnad_model_before_after_weights.pdf   وش غيّر التدريب في الأوزان (بالعربي)
  scripts/ و data/                       الكود والبرومبتات والأجوبة الخام ودرجات الـ 108 توليد
```

</div>

## إعادة بناء الخط من الأول

بايثون 3.12 مع PyTorch CUDA و`transformers` و`peft` و`bitsandbytes` و`accelerate` و`datasets`. تدريب v5 استخدم كرت واحد 12 جيجا.

| الخطوة | الأمر | المخرج |
|---|---|---|
| النموذج الأساسي | `python scripts/download_model.py` | `models/allam-7b-instruct` |
| ملف الـ adapter | `python scripts/join_adapter.py` (فقط لاستخدام الـ adapter المرفق، التدريب يكتب الملف بنفسه) | `final_adapter/adapter_model.safetensors` |
| محتوى المهام | `python scripts/generate_rich_task_banks.py` (يحتاج Ollama مع `qwen2.5:7b-instruct`). شوف الملاحظة تحت | `datasets/rich_task_banks.json` |
| تنظيف الكلمات الخربانة | `python scripts/fix_cyrillic_glitch.py` ثم `python scripts/scan_all_scripts.py` | الفحص لازم يطبع `"count": 0` |
| بناء الأمثلة | `python scripts/generate_synthetic_data.py` ثم `node scripts/build_prompts.js` | `datasets/raw_specs.json` و`datasets/sft_data.jsonl` |
| التدريب | `python scripts/train_qlora.py` | `models/allam-7b-esnad-lora-v5/final_adapter` |
| تجربته | `python scripts/try_model.py` | تفاعلي |
| التقييم | `python evaluation/scripts/eval_generate.py` و`eval_score.js` و`eval_analyze.py` | جداول ورسوم |

لإعادة التدريب بنفس النتيجة استخدم ملفي `datasets/rich_task_banks.json` و`datasets/sft_data.jsonl` الموجودين في المستودع. إعادة تشغيل خطوة محتوى المهام تنادي Qwen من جديد وتكتب مهام مختلفة. وانتبه أن ملف المحتوى المرفق مفهرس بـ `فكرة :: ميزة` وهذا اللي يقرأه `generate_synthetic_data.py`، بينما `generate_rich_task_banks.py` بصورته الحالية يكتب قائمة مشروع كامل لكل فكرة. يحتاج تعديل عشان يعيد بناء الملف المرفق.

بعض السكربتات فيها مسارات من الجهاز اللي انكتبت عليه، مثل `C:\Users\<name>\esnad-ai-training` و`C:/Users/<name>/GP2-Cl/services/aiEngine`. الثاني هو مستودع منصة إسناد اللي يوفّر `buildPrompt()`. غيّر هذي المسارات لمساراتك قبل التشغيل.

## الخطوات الجاية

- التدريب على أفكار مشاريع أكثر بكثير من 18، وإعادة كتابة قوالب المهام الضعيفة.
- التدريب على عدد المهام وعلى مراحل المشروع الست، عشان المنصة يكون عليها تصليح أقل.
- المقارنة بالنموذج الأصلي ببرومبت مبسّط، وتقييم بشري لعينة من المهام.

## المصادر

- ALLaM-7B-Instruct-preview: https://huggingface.co/humain-ai/ALLaM-7B-Instruct-preview
- QLoRA، Dettmers وآخرون 2023: https://arxiv.org/abs/2305.14314
- PEFT: https://github.com/huggingface/peft، وbitsandbytes: https://github.com/bitsandbytes-foundation/bitsandbytes، وTransformers: https://github.com/huggingface/transformers
- Qwen2.5-7B-Instruct اللي كتب محتوى المهام: https://ollama.com/library/qwen2.5
- ممارسات WBS وقاعدة 8/80: https://asana.com/resources/work-breakdown-structure و https://community.atlassian.com/forums/App-Central-articles/3-Work-Breakdown-Structure-WBS-Gantt-Chart-examples-for-software/ba-p/2338509
- تقسيم قصص المستخدم لمهام بالذكاء الاصطناعي: https://arxiv.org/abs/2605.07320
- تقسيم القصص، SPIDR: https://www.mountaingoatsoftware.com/agile/five-simple-but-powerful-ways-to-split-user-stories
- NEODATASET، 40 ألف قصة مستخدم حقيقية، قرأناها للأسلوب فقط وما استخدمناها بالتدريب: https://huggingface.co/datasets/giseldo/neodataset

## الحالة والرخصة

مستودع فريق إسناد، مشروع تخرج في جامعة الإمام محمد بن سعود الإسلامية. الـ adapter مشتق من ALLaM-7B-Instruct-preview ويتبع رخصته Apache-2.0 حسب وسمه في Hugging Face. كل أمثلة التدريب اصطناعية: الأسماء والنبذ والمشاريع مختلقة، وما فيها بيانات شخصية.

</div>

</details>
