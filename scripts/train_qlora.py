"""
QLoRA fine-tuning of ALLaM-7B-Instruct on the synthetic task-decomposition
dataset. Loads the base model in 4-bit (frozen), attaches small trainable
LoRA adapters, and trains only those adapters. Loss is masked on the prompt
tokens so the model only learns to produce the expected JSON completion.
"""
import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

MODEL_PATH = "models/allam-7b-instruct"
DATA_PATH = "datasets/sft_data.jsonl"
OUTPUT_DIR = "models/allam-7b-esnad-lora-v5"
MAX_LENGTH = 1600

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
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("Building tokenized dataset...")
raw = [json.loads(l) for l in open(DATA_PATH, encoding="utf-8")]


def tokenize_example(ex):
    messages_full = [
        {"role": "user", "content": ex["prompt"]},
        {"role": "assistant", "content": ex["completion"]},
    ]
    messages_prompt_only = [{"role": "user", "content": ex["prompt"]}]

    full_text = tokenizer.apply_chat_template(messages_full, tokenize=False)
    prompt_text = tokenizer.apply_chat_template(messages_prompt_only, tokenize=False, add_generation_prompt=True)

    full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)["input_ids"]

    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(full_ids))
    for i in range(prompt_len):
        labels[i] = -100

    return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}


dataset = Dataset.from_list(raw).map(tokenize_example, remove_columns=["prompt", "completion"])
split = dataset.train_test_split(test_size=0.08, seed=42)
train_ds, eval_ds = split["train"], split["test"]
print(f"train examples: {len(train_ds)}, eval examples: {len(eval_ds)}")


def collate(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, attention_mask, labels = [], [], []
    pad_id = tokenizer.pad_token_id
    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(b["attention_mask"] + [0] * pad_len)
        labels.append(b["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=40,
    save_strategy="epoch",
    save_total_limit=2,
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collate,
)

print("Starting training...")
trainer.train()

print("Saving LoRA adapter...")
model.save_pretrained(OUTPUT_DIR + "/final_adapter")
tokenizer.save_pretrained(OUTPUT_DIR + "/final_adapter")
print("Done. Adapter saved to", OUTPUT_DIR + "/final_adapter")
