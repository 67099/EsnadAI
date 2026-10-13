"""
Modal deployment for the Esnad task-decomposition model (ALLaM-7B-Instruct + LoRA v5).

Same "prompt in, completion out" contract as inference_server.py — aiEngine.js
does not need to change at all once AI_INFERENCE_URL points here.

Setup (one-time, on your machine):
    pip install modal
    modal setup                      # opens a browser to log in / create an account

Deploy:
    modal deploy server/modal_app.py

The command prints a public HTTPS URL ending in /generate — put that exact
URL in Render's AI_INFERENCE_URL environment variable.

Notes:
  - Modal's SDK evolves; if a decorator/parameter name below has changed
    (e.g. fastapi_endpoint / web_endpoint, scaledown_window / container_idle_timeout),
    `modal deploy` will error and name the new one — this file is the one
    that was deployed as the live `esnad-inference` app (A10G GPU).
  - First request after a cold start takes longer (downloading + loading the
    base model into the persistent volume); subsequent requests on a warm
    container are fast. scaledown_window keeps a container warm for a while
    after each request so back-to-back generations don't all pay that cost.
"""
import modal

app = modal.App("esnad-inference")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
        "fastapi[standard]",
        "tiktoken",
        "sentencepiece",
    )
    .add_local_dir(
        "models/allam-7b-esnad-lora-v5/final_adapter",
        remote_path="/adapter",
    )
)

# Persists the downloaded base model across cold starts so you don't
# re-download ~15GB from Hugging Face on every new container.
hf_cache = modal.Volume.from_name("esnad-hf-cache", create_if_missing=True)

MODEL_ID = "humain-ai/ALLaM-7B-Instruct-preview"
GENERATION_TIMEOUT_S = 180


@app.cls(
    image=image,
    gpu="A10G",
    memory=24576,  # 24GB CPU RAM — loading a 7B model's safetensors before they
    # move to the GPU got OOM-killed (exit code 137) on Modal's default allocation
    volumes={"/cache": hf_cache},
    timeout=GENERATION_TIMEOUT_S,
    scaledown_window=180,  # keep a container warm for 3 min after last request
    max_containers=3,  # up to 3 concurrent users generating at once before requests queue
)
class Model:
    @modal.enter()
    def load(self):
        import os

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        os.environ["HF_HOME"] = "/cache"

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir="/cache")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading base model in 4-bit...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=bnb_config, device_map="auto", cache_dir="/cache"
        )
        print("Attaching v5 adapter...")
        self.model = PeftModel.from_pretrained(base_model, "/adapter")
        self.model.eval()
        print("Ready.")

    @modal.fastapi_endpoint(method="POST", docs=True)
    def generate(self, item: dict):
        import torch

        prompt = item.get("prompt", "")
        if not prompt:
            return {"error": "missing 'prompt'"}

        max_new_tokens = int(item.get("max_new_tokens", 550))

        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        completion = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return {"completion": completion}

    @modal.fastapi_endpoint(method="GET")
    def health(self):
        return {"status": "ok"}
