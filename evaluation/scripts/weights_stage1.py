"""
Stage 1: measure how much the latest fine-tune (v5 LoRA adapter) changed the SDAIA ALLaM-7B-Instruct weights.

For every LoRA-targeted linear layer (32 layers x 7 modules) the fine-tuned weight is
    W_after = W_before + (lora_alpha / r) * B @ A
so the change is exactly delta = scale * B @ A. This script reads the untouched base weights
and the adapter, and writes one row of statistics per layer/module, plus a summary
(parameter counts, bias check, training settings).

Run with the project venv (needs torch + safetensors), reads only, writes to the scratchpad.
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open

REPO = Path(r"C:\Users\kimat\esnad-ai-training")
BASE = REPO / "models" / "allam-7b-instruct"
ADAPTER_DIR = REPO / "models" / "allam-7b-esnad-lora-v5" / "final_adapter"
CKPT_DIR = REPO / "models" / "allam-7b-esnad-lora-v5" / "checkpoint-276"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
adapter_cfg = json.loads((ADAPTER_DIR / "adapter_config.json").read_text())
scale = adapter_cfg["lora_alpha"] / adapter_cfg["r"]
index = json.loads((BASE / "model.safetensors.index.json").read_text())
weight_map = index["weight_map"]

# ---------- open the adapter and find every lora_A / lora_B pair
adapter = safe_open(str(ADAPTER_DIR / "adapter_model.safetensors"), framework="pt", device="cpu")
adapter_keys = list(adapter.keys())
bias_keys_adapter = [k for k in adapter_keys if "bias" in k]
pat = re.compile(r"layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.lora_A\.weight$")
modules = []
for key in adapter_keys:
    m = pat.search(key)
    if m:
        layer, block, name = int(m.group(1)), m.group(2), m.group(3)
        modules.append((layer, block, name, key, key.replace("lora_A", "lora_B")))
modules.sort()
print(f"adapter tensors: {len(adapter_keys)} | LoRA-targeted modules: {len(modules)} | scale (alpha/r): {scale}")
print("sample adapter key:", adapter_keys[0])

# ---------- open every base shard lazily
shards = {}
def base_tensor(name):
    shard = weight_map[name]
    if shard not in shards:
        shards[shard] = safe_open(str(BASE / shard), framework="pt", device="cpu")
    return shards[shard].get_tensor(name)

rows = []
sv_rows = []
t0 = time.time()
for i, (layer, block, name, key_a, key_b) in enumerate(modules):
    base_name = f"model.layers.{layer}.{block}.{name}.weight"
    W_bf16 = base_tensor(base_name).to(device)
    A = adapter.get_tensor(key_a).to(device).float()   # r x in
    B = adapter.get_tensor(key_b).to(device).float()   # out x r
    W = W_bf16.float()
    delta = (B @ A) * scale                              # out x in

    # singular values of delta without forming an SVD of the big matrix: delta = scale * B A,
    # QR both factors and take the SVD of the small r x r core.
    _, Rb = torch.linalg.qr(B)
    _, Ra = torch.linalg.qr(A.T)
    sv = torch.linalg.svdvals(Rb @ Ra.T) * scale
    sv_np = sv.cpu().numpy()
    energy = sv_np ** 2
    p = sv_np / sv_np.sum()
    eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))

    # what survives when the adapter is merged into the bf16 base weights
    merged = (W + delta).to(torch.bfloat16)
    realised = merged.float() - W
    changed_frac = float((merged != W_bf16).float().mean())
    cos_merge = float((realised.flatten() @ delta.flatten()) / (realised.norm() * delta.norm() + 1e-30))

    fro_W, fro_D = float(W.norm()), float(delta.norm())
    rows.append(
        dict(
            layer=layer,
            block=block,
            module=name,
            out_features=int(W.shape[0]),
            in_features=int(W.shape[1]),
            w_fro=fro_W,
            delta_fro=fro_D,
            rel_change_pct=100 * fro_D / fro_W,
            w_mean_abs=float(W.abs().mean()),
            delta_mean_abs=float(delta.abs().mean()),
            delta_max_abs=float(delta.abs().max()),
            w_std=float(W.std()),
            delta_std=float(delta.std()),
            lora_A_fro=float(A.norm()),
            lora_B_fro=float(B.norm()),
            sv1=float(sv_np[0]),
            energy_top1_pct=100 * float(energy[0] / energy.sum()),
            energy_top4_pct=100 * float(energy[:4].sum() / energy.sum()),
            stable_rank=float(energy.sum() / energy[0]),
            effective_rank=eff_rank,
            bf16_elements_changed_pct=100 * changed_frac,
            bf16_merge_cosine=cos_merge,
        )
    )
    for k, s in enumerate(sv_np):
        sv_rows.append(dict(layer=layer, module=name, block=block, k=k + 1, sv=float(s), sv_norm=float(s / sv_np[0])))

    # keep a flat sample of both distributions for one representative module for the histogram
    if layer == 15 and name == "q_proj":
        rng = np.random.default_rng(0)
        idx = rng.choice(W.numel(), size=400_000, replace=False)
        np.savez(OUT / "sample_q_proj_layer15.npz", w=W.flatten()[idx].cpu().numpy(), delta=delta.flatten()[idx].cpu().numpy())

    del W_bf16, W, A, B, delta, merged, realised
    if (i + 1) % 28 == 0:
        print(f"  {i + 1}/{len(modules)} modules  ({time.time() - t0:.0f}s)")

pd.DataFrame(rows).to_csv(OUT / "per_module.csv", index=False)
pd.DataFrame(sv_rows).to_csv(OUT / "singular_values.csv", index=False)

# ---------- whole-model facts: parameter counts and bias check
total_params = 0
group_params = {"embeddings": 0, "lm_head": 0, "rmsnorm": 0, "attention_proj": 0, "mlp_proj": 0}
for shard_name in sorted(set(weight_map.values())):
    f = safe_open(str(BASE / shard_name), framework="pt", device="cpu")
    for tname in f.keys():
        n = int(np.prod(f.get_slice(tname).get_shape()))
        total_params += n
        if "embed_tokens" in tname:
            group_params["embeddings"] += n
        elif "lm_head" in tname:
            group_params["lm_head"] += n
        elif "norm" in tname:
            group_params["rmsnorm"] += n
        elif "self_attn" in tname:
            group_params["attention_proj"] += n
        elif "mlp" in tname:
            group_params["mlp_proj"] += n

lora_params = sum(int(np.prod(adapter.get_slice(k).get_shape())) for k in adapter_keys)
bias_keys_base = [k for k in weight_map if k.endswith(".bias")]

train_args = {}
try:
    import transformers  # noqa: F401  (needed to unpickle TrainingArguments)
    ta = torch.load(CKPT_DIR / "training_args.bin", weights_only=False)
    for field in (
        "learning_rate", "num_train_epochs", "per_device_train_batch_size", "gradient_accumulation_steps",
        "lr_scheduler_type", "warmup_ratio", "warmup_steps", "weight_decay", "max_grad_norm", "optim", "seed", "bf16", "fp16",
    ):
        train_args[field] = str(getattr(ta, field, None))
except Exception as exc:  # the summary is still useful without it
    train_args["error"] = str(exc)

summary = dict(
    base_model="ALLaM-7B-Instruct (SDAIA), bf16",
    adapter="allam-7b-esnad-lora-v5/final_adapter",
    lora=dict(r=adapter_cfg["r"], alpha=adapter_cfg["lora_alpha"], scale=scale, dropout=adapter_cfg["lora_dropout"],
              bias_setting=adapter_cfg["bias"], target_modules=sorted(adapter_cfg["target_modules"])),
    base_total_params=total_params,
    base_params_by_group=group_params,
    lora_params=lora_params,
    lora_fraction_pct=100 * lora_params / total_params,
    n_lora_modules=len(modules),
    bias_keys_in_base=bias_keys_base,
    bias_keys_in_adapter=bias_keys_adapter,
    adapter_tensors=len(adapter_keys),
    training_args=train_args,
    training_examples=4800,
    adapter_file_mb=round((ADAPTER_DIR / "adapter_model.safetensors").stat().st_size / 1e6, 1),
)
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary, indent=2, ensure_ascii=False)[:2500])
print(f"done in {time.time() - t0:.0f}s")
