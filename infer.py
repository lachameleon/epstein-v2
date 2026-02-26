#!/usr/bin/env python3
import os
import sys
import subprocess
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel, PeftConfig
from peft.tuners.lora import LoraConfig
from threading import Thread

MODEL = "teapotai/tinyteapot"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER = os.path.join(BASE_DIR, "epstein_lora_teapotai_tinyteapot")
subprocess.run("clear",text=True,shell=True)
BANNER = """
Jefrrey Inference
"""

LORA_SCALING = 0.3  # 0.0 = no LORA impact, 1.0 = full impact
REPETITION_PENALTY = 1.5
NO_REPEAT_NGRAM_SIZE = 3

def find_all_checkpoints():
    adapter_dir = os.path.join(BASE_DIR, "epstein_lora_teapotai_tinyteapot")
    if not os.path.isdir(adapter_dir):
        return []
    
    checkpoints = []
    for item in os.listdir(adapter_dir):
        if item.startswith("checkpoint-"):
            step = item.split("-")[-1]
            if step.isdigit():
                checkpoints.append((int(step), os.path.join(adapter_dir, item)))
    
    checkpoints.sort(reverse=True)
    return checkpoints

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
    
    checkpoints = find_all_checkpoints()
    
    print("\n╔═══════════════════════════════════════════════════╗")
    print("║              Available Checkpoints                 ║")
    print("╚═══════════════════════════════════════════════════╝")
    
    print("  [0] Base model only (no LoRA)")
    for i, (step, path) in enumerate(checkpoints, 1):
        print(f"  [{i}] Checkpoint {step}")
    
    while True:
        try:
            choice = input("\nSelect checkpoint [0]: ").strip()
            if choice == "":
                checkpoint_path, selected_step = None, None
                break
            idx = int(choice)
            if idx == 0:
                checkpoint_path, selected_step = None, None
                break
            if 1 <= idx <= len(checkpoints):
                selected_step, checkpoint_path = checkpoints[idx - 1]
                break
            print(f"Invalid selection. Choose 0-{len(checkpoints)}")
        except ValueError:
            print("Please enter a number")
    
    if selected_step:
        print(f"\nLoading checkpoint-{selected_step}...")
        model = PeftModel.from_pretrained(base, checkpoint_path, scaling_factor=LORA_SCALING)
        print(f"✓ Loaded from checkpoint-{selected_step} (scaling={LORA_SCALING})")
    elif is_valid_adapter(ADAPTER):
        print(f"\nLoading adapter: {ADAPTER}...")
        model = PeftModel.from_pretrained(base, ADAPTER, scaling_factor=LORA_SCALING)
        print(f"✓ Loaded with LoRA adapter (scaling={LORA_SCALING})")
    else:
        print(f"\n⚠ No adapter or checkpoint found")
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
            
            print("\n\033[90mThinking...\033[0m")
            print("\n\033[1;36mAssistant:\033[0m ", end="", flush=True)
            
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            with torch.no_grad():
                gen_kwargs = dict(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=REPETITION_PENALTY,
                    no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer
                )
                thread = Thread(target=model.generate, kwargs=gen_kwargs)
                thread.start()
                
                generated_text = ""
                for text in streamer:
                    print(text, end="", flush=True)
                    generated_text += text
                thread.join()
            
            print("\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n\033[91mError:\033[0m {e}")


if __name__ == "__main__":
    chat()
