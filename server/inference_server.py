"""
Local inference server for the fine-tuned Esnad task-decomposition model
(ALLaM-7B-Instruct + v5 LoRA adapter). Loads the model ONCE at startup and
stays resident, serving generation requests over HTTP — this is what makes
it fast enough to actually call from Esnad's Node backend per-request
(loading the model fresh each time would take ~15-20 seconds).

This is a thin "prompt in, completion out" service on purpose: Esnad's own
aiEngine.js remains the single source of truth for prompt construction
(buildPrompt()) — this server does not rebuild or duplicate that logic.

Run: venv\\Scripts\\python.exe server\\inference_server.py
Esnad backend should POST to http://localhost:8765/generate
"""
import torch
from flask import Flask, request, jsonify
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = "models/allam-7b-instruct"
ADAPTER_PATH = "models/allam-7b-esnad-lora-v5/final_adapter"
PORT = 8765

app = Flask(__name__)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, quantization_config=bnb_config, device_map="auto"
)
print("Attaching v4 adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
print(f"Ready. Listening on http://localhost:{PORT}")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "missing 'prompt'"}), 400

    max_new_tokens = int(data.get("max_new_tokens", 900))

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return jsonify({"completion": completion})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT, threaded=False)
