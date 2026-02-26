#!/usr/bin/env python3
import os
import sys
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from peft.tuners.lora import LoraConfig

MODEL = "teapotai/tinyteapot"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER = os.path.join(BASE_DIR, "epstein_lora_teapotai_tinyteapot")

BANNER = """
╔═══════════════════════════════════════════════════╗
║         🤖 Epstein Files Inference Bot            ║
╚═══════════════════════════════════════════════════╝
"""

def find_latest_checkpoint():
    adapter_dir = os.path.join(BASE_DIR, "epstein_lora_teapotai_tinyteapot")
    if not os.path.isdir(adapter_dir):
        return None
    
    checkpoints = []
    for item in os.listdir(adapter_dir):
        if item.startswith("checkpoint-"):
            step = item.split("-")[-1]
            if step.isdigit():
                checkpoints.append((int(step), os.path.join(adapter_dir, item)))
    
    if checkpoints:
        checkpoints.sort(reverse=True)
        return checkpoints[0][1]
    return None

def is_valid_adapter(path):
    return os.path.isfile(os.path.join(path, "adapter_config.json"))

def load_model():
    print(f"{BANNER}")
    print(f"Loading base model: {MODEL}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    base = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL,
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    checkpoint_path = find_latest_checkpoint()
    
    if is_valid_adapter(ADAPTER):
        print(f"Loading adapter: {ADAPTER}...")
        model = PeftModel.from_pretrained(base, ADAPTER)
        print("✓ Loaded with LoRA adapter")
    elif checkpoint_path and is_valid_adapter(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}...")
        model = PeftModel.from_pretrained(base, checkpoint_path)
        print("✓ Loaded from checkpoint")
    else:
        print(f"⚠ No adapter or checkpoint found")
        print("  Using base model only (no fine-tuning)")
        model = base
    
    model.eval()
    print("\n" + "─" * 50)
    print("Ready! Type your questions below.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    return model, tokenizer


def chat():
    model, tokenizer = load_model()
    
    while True:
        try:
            q = input("\n\033[1mYou:\033[0m ").strip()
            if q.lower() in ("quit", "exit", "q"):
                print("\nGoodbye!")
                break
            if not q:
                continue
            
            prompt = f"""You are a knowledgeable assistant with access to the Epstein files documents. Your task is to answer questions about this information accurately and factually based ONLY on the provided documents. Never fabricate information - only answer based on the provided context. If you don't know the answer based on the documents, say so.

### Document Content:
[Context would be here from training data]

### Question: {q}

### Answer:"""
            
            print("\n\033[90mThinking...\033[0m", end="\r")
            
            with torch.no_grad():
                out = model.generate(
                    **tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048),
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            ans = tokenizer.decode(out[0], skip_special_tokens=True)
            if prompt in ans:
                ans = ans[len(prompt):].strip()
            
            print(f"\n\033[1;36mAssistant:\033[0m {ans}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n\033[91mError:\033[0m {e}")


if __name__ == "__main__":
    chat()
