"""
Fair before/after generation test. Both variants (SDAIA base model with the adapter switched off, and the same model with
our v5 adapter) answer the exact production prompts (services/aiEngine.js buildPrompt) with the exact production sampling
settings (server/modal_app.py: max_new_tokens=550, temperature=0.4, top_p=0.9, repetition_penalty=1.2), using the same
random seed for each (prompt, sample) pair. Results are appended to raw_outputs.jsonl as they finish, so the run can resume.

Usage (project venv):  python eval_generate.py <scratchpad_dir> [samples_per_prompt]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO = Path(r"C:\Users\kimat\esnad-ai-training")
SP = Path(sys.argv[1])
N_SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 3
# optional third argument "a:b" runs only prompts a..b-1 (used to split the test in halves)
SPEC_RANGE = tuple(int(x) for x in sys.argv[3].split(":")) if len(sys.argv) > 3 else None
MODEL_PATH = str(REPO / "models" / "allam-7b-instruct")
ADAPTER_PATH = str(REPO / "models" / "allam-7b-esnad-lora-v5" / "final_adapter")
OUT = SP / "raw_outputs.jsonl"
MAX_NEW_TOKENS = 550

heldout = json.loads((REPO / "datasets" / "heldout_eval_specs.json").read_text(encoding="utf-8"))
for i, s in enumerate(heldout):
    s["id"], s["group"] = f"h{i + 1}", "heldout"
specs = heldout + json.loads((SP / "eval_specs_new.json").read_text(encoding="utf-8"))
(SP / "eval_specs_all.json").write_text(json.dumps(specs, ensure_ascii=False, indent=1), encoding="utf-8")
subprocess.run(["node", str(SP / "build_prompts.js"), str(SP / "eval_specs_all.json"), str(SP / "eval_prompts.json")], check=True)
prompts = {p["id"]: p["prompt"] for p in json.loads((SP / "eval_prompts.json").read_text(encoding="utf-8"))}

done = set()
if OUT.exists():
    for line in OUT.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        done.add((r["id"], r["sample"], r["variant"]))
lo, hi = SPEC_RANGE or (0, len(specs))
n_run = hi - lo
print(f"prompts {lo}..{hi - 1}: {n_run} prompts x {N_SAMPLES} samples x 2 variants = {n_run * N_SAMPLES * 2} generations ({len(done)} already saved overall)", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER_PATH)
model.eval()
print("model loaded", flush=True)


def generate(prompt, seed):
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.4, top_p=0.9,
                             repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), int(len(new_tokens)), time.time() - t0


t_start = time.time()
count = 0
total = n_run * N_SAMPLES * 2
with OUT.open("a", encoding="utf-8") as f:
    for sample in range(N_SAMPLES):
        for idx, spec in enumerate(specs):
            if not (lo <= idx < hi):
                continue
            seed = 7919 * (sample + 1) + idx
            for variant in ("base", "ft"):
                key = (spec["id"], sample, variant)
                if key in done:
                    continue
                if variant == "base":
                    with model.disable_adapter():
                        text, n_tok, secs = generate(prompts[spec["id"]], seed)
                else:
                    text, n_tok, secs = generate(prompts[spec["id"]], seed)
                rec = dict(id=spec["id"], group=spec["group"], sample=sample, variant=variant, seed=seed, text=text,
                           new_tokens=n_tok, hit_cap=n_tok >= MAX_NEW_TOKENS, seconds=round(secs, 1))
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                count += 1
                elapsed = time.time() - t_start
                print(f"[{count}/{total}] {spec['id']} s{sample} {variant}: {n_tok} tok, {secs:.0f}s  (elapsed {elapsed / 60:.1f} min)", flush=True)
print("DONE", flush=True)
