from pathlib import Path
import torch
from transformers import ModernBertConfig, ModernBertForSequenceClassification, AutoTokenizer

# ─── Edit these if needed ──────────────────────────────────────────────────────
CKPT_PATH  = "/checkpoint/ram/tianjian/math-classifier/qwen3-4b-emb-finetuned/model-70.pt"                         # your .pt file
TARGET_DIR = Path("/checkpoint/ram/tianjian/math-classifier-hf")
# ───────────────────────────────────────────────────────────────────────────────


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # 1 ▸ Load the base model’s config
    config = ModernBertConfig.from_pretrained("answerdotai/ModernBERT-base")

    # 2 ▸ Build an empty classification model (change class if you used a
    #       different head—e.g., BertForTokenClassification, BertModel, etc.)
    model = ModernBertForSequenceClassification(config)

    # 3 ▸ Load fine-tuned weights
    state_dict = torch.load(CKPT_PATH, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("⚠️  Missing keys:", missing)
    if unexpected:
        print("⚠️  Unexpected keys:", unexpected)

    # 4 ▸ Save model & tokenizer in HF format
    model.save_pretrained(TARGET_DIR)
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    tokenizer.save_pretrained(TARGET_DIR)

    print(f"✅  Hugging Face model saved to → {TARGET_DIR.resolve()}")


if __name__ == "__main__":
    main()
