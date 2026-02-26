#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import warnings
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel, PeftConfig
from peft.tuners.lora import LoraConfig
from threading import Thread

warnings.filterwarnings("ignore")

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

parser = argparse.ArgumentParser(description="Epstein Inference")
parser.add_argument("--lora-scale", type=float, default=None, help="LoRA scaling factor (0.0-1.0)")
parser.add_argument("--repetition-penalty", type=float, default=1.5, help="Repetition penalty")
args, _ = parser.parse_known_args()

if args.lora_scale is None:
    while True:
        try:
            lora_input = input(f"\n{BOLD}LoRA Scaling (0.0-1.0){RESET} [0.3]: ").strip()
            if lora_input == "":
                args.lora_scale = 0.3
                break
            val = float(lora_input)
            if 0.0 <= val <= 1.0:
                args.lora_scale = val
                break
            print(f"{RED}Please enter a value between 0.0 and 1.0{RESET}")
        except ValueError:
            print(f"{RED}Please enter a valid number{RESET}")

MODEL = "teapotai/tinyteapot"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER = os.path.join(BASE_DIR, "epstein_lora_teapotai_tinyteapot")
subprocess.run("clear",text=True,shell=True)
BANNER = f"""
{BOLD}{MAGENTA}╔═══════════════════════════════════════════════════════════╗
║  ██████╗ ███████╗███████╗██╗     ██╗███╗   ██╗███████╗               ║
║  ██╔══██╗██╔════╝██╔════╝██║     ██║████╗  ██║██╔════╝               ║
║  ██║  ██║█████╗  █████╗  ██║     ██║██╔██╗ ██║█████╗                 ║
║  ██║  ██║██╔══╝  ██╔══╝  ██║     ██║██║╚██╗██║██╔══╝                 ║
║  ██████╔╝███████╗███████╗███████╗██║██║ ╚████║███████╗               ║
║  ╚═════╝ ╚══════╝╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝               ║
║          {CYAN}Inference Engine v2.0{RESET}                              {MAGENTA}║
╚═══════════════════════════════════════════════════════════╝{RESET}
"""

LORA_SCALING: float = args.lora_scale
REPETITION_PENALTY = args.repetition_penalty
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
    print(f"{DIM}Loading base model: {MODEL}...{RESET}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    base = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL,
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    checkpoints = find_all_checkpoints()
    
    print(f"\n{BOLD}{CYAN}╔═══════════════════════════════════════════════════════════╗")
    print("║              Available Checkpoints                 ║")
    print("╚═══════════════════════════════════════════════════════════╝{RESET}")
    
    print(f"  {GREEN}[0]{RESET}  Base model only (no LoRA)")
    for i, (step, path) in enumerate(checkpoints, 1):
        print(f"  {GREEN}[{i}]{RESET}  Checkpoint {step}")
    
    print(f"\n  {YELLOW}LoRA Scaling:{RESET} {LORA_SCALING}  {YELLOW}Rep Penalty:{RESET} {REPETITION_PENALTY}")
    
    while True:
        try:
            choice = input(f"\n{BOLD}Select checkpoint{RESET} [0]: ").strip()
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
            print(f"{RED}Invalid selection. Choose 0-{len(checkpoints)}{RESET}")
        except ValueError:
            print(f"{RED}Please enter a number{RESET}")
    
    if selected_step:
        print(f"\n{GREEN}Loading checkpoint-{selected_step}...{RESET}")
        model = PeftModel.from_pretrained(base, checkpoint_path, scaling_factor=LORA_SCALING)
        print(f"{GREEN}✓{RESET} Loaded from checkpoint-{selected_step} (scaling={LORA_SCALING})")
    elif is_valid_adapter(ADAPTER):
        print(f"\n{GREEN}Loading adapter: {ADAPTER}...{RESET}")
        model = PeftModel.from_pretrained(base, ADAPTER, scaling_factor=LORA_SCALING)
        print(f"{GREEN}✓{RESET} Loaded with LoRA adapter (scaling={LORA_SCALING})")
    else:
        print(f"\n{YELLOW}⚠ No adapter or checkpoint found{RESET}")
        print(f"  {DIM}Using base model only (no fine-tuning){RESET}")
        model = base
    
    model.eval()
    print(f"\n{BOLD}{MAGENTA}{'─' * 56}{RESET}")
    print(f"{BOLD}{GREEN}Ready!{RESET} Type your questions below.")
    print(f"{DIM}Type 'quit' or 'exit' to stop.{RESET}\n")
    
    return model, tokenizer


def chat():
    model, tokenizer = load_model()
    
    while True:
        try:
            q = input(f"\n{BOLD}{CYAN}You:{RESET} ").strip()
            if q.lower() in ("quit", "exit", "q"):
                print(f"{YELLOW}Goodbye!{RESET}")
                break
            if not q:
                continue
            
            prompt = f"""You are a knowledgeable assistant with access to the Epstein files documents. Your task is to answer questions about this information accurately and factually based ONLY on the provided documents. Never fabricate information - only answer based on the provided context. If you don't know the answer based on the documents, say so.

### Document Content:
[Context would be here from training data]

### Question: {q}

### Answer:"""
            
            print(f"\n{DIM}Thinking...{RESET}")
            print(f"\n{BOLD}{MAGENTA}Assistant:{RESET} ", end="", flush=True)
            
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
            print(f"\n\n{YELLOW}Goodbye!{RESET}")
            break
        except Exception as e:
            print(f"\n{RED}Error: {e}{RESET}")


if __name__ == "__main__":
    chat()
