from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="humain-ai/ALLaM-7B-Instruct-preview",
    local_dir="models/allam-7b-instruct",
    allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*"],
)
print("Downloaded to:", path)
