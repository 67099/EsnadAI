"""
Interactive CLI to try the fine-tuned model yourself.
Run: venv\\Scripts\\python.exe scripts\\try_model.py
Paste a full JSON test case ({"projectIdea", "userPrompt", "teamMembers"}),
then press Enter on an empty line to submit. The model prints its response.
Paste empty input to quit.
"""
import json
import subprocess

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = "models/allam-7b-instruct"
ADAPTER_PATH = "models/allam-7b-esnad-lora-v3/final_adapter"


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


def main():
    print("Loading model (may take ~15 seconds)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    print("Ready.\n")

    while True:
        print("Paste one JSON test case (projectIdea, userPrompt, teamMembers), then press Enter on an empty line. Empty input to quit:")
        lines = []
        while True:
            line = input()
            if not line.strip():
                break
            lines.append(line)
        if not lines:
            break

        try:
            spec = json.loads("\n".join(lines))
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}\n")
            continue

        prompt = build_prompt_via_node({
            "userPrompt": spec.get("userPrompt", ""),
            "projectIdea": spec.get("projectIdea", ""),
            "teamMembers": spec.get("teamMembers", []),
        })

        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=400,
                do_sample=True, temperature=0.4, top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print("\n--- Model response ---")
        print(response.strip())
        print()


if __name__ == "__main__":
    main()
