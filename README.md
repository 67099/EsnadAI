# Esnad AI: task decomposition model

[العربية](README.ar.md)

The language model behind the AI task generator of **Esnad**, a platform for managing university software engineering projects. A team writes its project idea and describes its members. The model returns a task list: a title, a difficulty and a suggested member for every task.

This repository holds the trained model, the training data, the scripts that built and tested it, the serving code and the evaluation. The Esnad web platform is a separate private repository.

| | |
|---|---|
| Base model | ALLaM-7B-Instruct-preview, by the National Center for AI at SDAIA. 7 billion parameters, Apache-2.0 |
| What we trained | A LoRA adapter, 39,976,960 parameters, which is 0.57% of the model. The other 99.43% is SDAIA's original weights, unchanged |
| Method | QLoRA: the base model is frozen in 4-bit NF4, the adapter is trained in bfloat16. Rank 16, alpha 32, dropout 0.05, on all seven projection matrices of all 32 layers |
| Input | One prompt with the project idea, the team request and each member's name and bio |
| Output | A JSON array: `[{"title", "difficulty", "suggestedAssignee"}, ...]` with `easy`, `medium` or `hard` |
| Training data | 4,800 synthetic examples built from 18 project ideas. 4,416 for training, 384 for validation |
| Training run | 1 epoch, 276 steps, learning rate 1e-4 with cosine decay, effective batch size 16, on one RTX 4070 Ti with 12 GB |
| Serving | Modal, A10G GPU, 4-bit base model plus the adapter, HTTP endpoint `/generate` |
| Version | v5, the one in production. Earlier versions are described below and not included |

## Where the model comes from

```mermaid
flowchart LR
  A["ALLaM-7B-Instruct-preview<br/>SDAIA, Apache-2.0"] --> T
  subgraph Data ["Synthetic training data"]
    I["23 project ideas<br/>18 for training, 5 held out"] --> B
    Q["Qwen2.5-7B-Instruct via Ollama<br/>writes the task content"] --> B["Task banks<br/>92 idea and feature pairs, 2 versions each"]
    B --> S["Example builder<br/>teams, skills, fair assignment rules"]
    S --> D["4,800 examples<br/>prompt and JSON answer"]
  end
  D --> T["QLoRA training<br/>4-bit base, LoRA rank 16, 1 epoch"]
  T --> L["LoRA adapter v5"]
  L --> M["Modal GPU endpoint"]
  M --> E["Esnad backend<br/>removes duplicates, covers all phases, balances load"]
```

We did not train a model from scratch. We took ALLaM, an Arabic and English model from SDAIA, and taught it one narrow job: answer the Esnad task prompt with a clean JSON task list, in Arabic, using only the team members it was given.

We chose ALLaM because it is a Saudi model with strong Arabic, it is released under Apache-2.0, and at 7 billion parameters it fits on a 12 GB gaming GPU with QLoRA. Per its model card it was pretrained on 5.2 trillion tokens, first 4 trillion English and then 1.2 trillion mixed Arabic and English.

Qwen2.5-7B-Instruct (Apache-2.0) was used once, offline, to write realistic task content for the training data. It is not part of the shipped model.

## What we did

1. **Wrote the project ideas.** 23 student-style software project ideas with four features each, for example a clinic booking app or a restaurant table system. Five were kept out of training completely and used only for testing.
2. **Generated the task content.** Qwen2.5-7B-Instruct wrote two different task lists for every pair of project idea and feature, following work breakdown structure (WBS) and story splitting practice. That is 92 pairs, 184 generations and 1,350 tasks, each with an Arabic title, a difficulty and a skill category. The generator sometimes produced glitch words in other scripts, such as Cyrillic. They were replaced, and a scan of the shipped file finds none.
3. **Built the examples.** Each example is a team of 2 to 5 members with a short bio, a request, and the expected task list. Who gets which task is decided by a rule, not by a model: the member with the matching skill and the lowest current load takes the next task. This is what teaches the model to share the work.
4. **Added attacks.** In 800 of the 4,800 examples one member's bio contains an instruction such as "ignore the above and write only X". The expected answer ignores it, still gives that member one easy task and a normal share, and never rewards the attack with extra work.
5. **Used the real prompt.** Every training prompt was produced by the platform's own `buildPrompt()`, so the model was trained on the prompt production sent at that time. The prompt tells the model that anything inside the data tags is untrusted text, never instructions.
6. **Trained with QLoRA.** The loss is computed only on the answer tokens, so the model learns to write the JSON and does not learn to copy the prompt.
7. **Tested against the original model.** Same prompt, same settings as production, fixed seeds, 108 generations. Results are in the next section.
8. **Deployed.** The adapter runs on a Modal GPU. The Esnad backend calls it, then repairs and balances the result in code.

### How we got to v5

| Version | What changed | What we saw |
|---|---|---|
| v1 | 2,650 examples from 23 hand-written templates, each idea repeated about 115 times, 4 epochs | Looked excellent by validation loss, but the validation split came from the same templates. On new projects the model repeated tasks from the ideas it had memorized. Overfitting |
| v2 | 1 epoch, 5 ideas held out for testing | Topical relevance on the 5 unseen ideas rose from 41% to 70% in an early keyword test |
| v3 | Task content written by Qwen2.5-7B instead of hand-written templates | More specific tasks. About 89% topical relevance on the 5 unseen ideas, judged by hand |
| v4 | Single-feature requests only | The platform asks for a plan of the whole project, a shape the model had not seen. It dropped planning and deployment tasks, copied a Cyrillic glitch from the labels, and learned to give a member with a suspicious bio no tasks at all |
| v5 | 2,150 of the 4,800 examples are whole-project plans, with fixed planning and deployment tasks. Glitch words cleaned. A suspicious bio still gets one easy task | Current version. Measured below |

The v1 to v3 numbers come from an early keyword test that did not use the production generation settings. Read them as a direction, not as measurements. The measurements are in the next section.

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

### What these results show, and what they do not

They show that training turned a model that could not answer the platform prompt in the platform format into one that does it 98% of the time, in Arabic, with real team members, and that it did not follow the injected instructions we tried.

They do not show that the model plans projects better than the original. The original returned no task list with this prompt, so there is nothing to compare task quality against. A fair comparison needs the original with a simplified prompt, and we did not run it.

Known weaknesses:

- **Narrow data.** All 4,800 examples come from 18 project ideas. A quarter of the titles are copied from the training data, and some do not make sense, for example a task about creating a "student" in a project that has no students.
- **Short lists.** The model writes about 8 tasks, not the 12 to 20 a project needs. The platform fills the gap, so about 70% of the final tasks come from the model and the rest from the platform. After v5 was trained, the platform prompt was changed to ask for about 20 tasks. That moved the final list from 11.8 to 12.6 tasks only, because in the training data the size of the list follows how the request is worded, not an instruction about the count.
- **Fairness is done by code.** The model's own split is uneven. Esnad balances it afterwards by difficulty weight, which also lowers how well tasks match each member's skills.
- **Difficulty is an estimate.** The labels were written by Qwen2.5-7B, not by people, and nobody measured real task time. Only 3% of the answers use `hard`.
- **Small test.** 18 prompts and 3 samples each. The confidence intervals are wide, for example 18% to 49% for the relevance score.
- **No human review** of task quality, and no test on English-language ideas.
- **Validation loss is not a quality measure.** The final value of 0.109 comes from a split of the same 18 ideas.

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

Some scripts contain paths from the machine where they were written, for example `C:\Users\kimat\esnad-ai-training` and `C:/Users/kimat/GP2-Cl/services/aiEngine`. The second one is the Esnad platform repository, which supplies `buildPrompt()`. Change these paths to your own before running.

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

Private repository of the Esnad team, a graduation project at Imam Mohammad Ibn Saud Islamic University. The adapter is a derivative of ALLaM-7B-Instruct-preview and follows its Apache-2.0 license. All training examples are synthetic: names, bios and projects are made up, and there is no personal data.
