#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
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

parser = argparse.ArgumentParser(description="LoRA Inference")
parser.add_argument("--base-model", type=str, default=None, help="Base model (e.g. meta-llama/Llama-2-7b-hf)")
parser.add_argument("--adapter", type=str, default=None, help="LoRA adapter (local path or hf://username/adapter)")
parser.add_argument("--lora-scale", type=float, default=None, help="LoRA scaling factor (0.0-1.0)")
parser.add_argument("--repetition-penalty", type=float, default=1.5, help="Repetition penalty")
parser.add_argument("--max-tokens", type=int, default=512, help="Max new tokens")
args, _ = parser.parse_known_args()

DEFAULT_BASE_MODEL = "teapotai/tinyteapot"

subprocess.run("clear", text=True, shell=True)
BANNER = f"""
{BOLD}{MAGENTA}╔═══════════════════════════════════════════════════════════╗
║  ██████╗ ███████╗███████╗██╗     ██╗███╗   ██╗███████╗               ║
║  ██╔══██╗██╔════╝██╔════╝██║     ██║████╗  ██║██╔════╝               ║
║  ██║  ██║█████╗  █████╗  ██║     ██║██╔██╗ ██║█████╗                 ║
║  ██║  ██║██╔══╝  ██╔══╝  ██║     ██║██║╚██╗██║██╔══╝                 ║
║  ██████╔╝███████╗███████╗███████╗██║██║ ╚████║███████╗               ║
║  ╚═════╝ ╚══════╝╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝               ║
║          {CYAN}LoRA Inference Engine{RESET}                               {MAGENTA}║
╚═══════════════════════════════════════════════════════════╝{RESET}
"""

def prompt_with_default(prompt_text, default=None, color=CYAN):
    if default:
        user_input = input(f"{prompt_text} [{color}{default}{RESET}]: ").strip()
        return user_input if user_input else default
    return input(f"{prompt_text}: ").strip()

def get_model_choice():
    print(f"{BANNER}")
    
    print(f"\n{BOLD}Step 1: Select Base Model{RESET}")
    print(f"  {DIM}Enter a HuggingFace model ID or press Enter for default{RESET}")
    print(f"  {GREEN}Examples:{RESET}")
    print(f"    - teapotai/tinyteapot")
    print(f"    - meta-llama/Llama-2-7b-hf")
    print(f"    - mistralai/Mistral-7B-Instruct-v0.1")
    print(f"    - Qwen/Qwen2-0.5B-Instruct")
    
    base_model = args.base_model
    if not base_model:
        base_model = prompt_with_default(f"\n{BOLD}Base model", DEFAULT_BASE_MODEL)
    
    print(f"\n{GREEN}Using base model: {base_model}{RESET}")
    return base_model

def get_adapter_choice():
    print(f"\n{BOLD}Step 2: Select LoRA Adapter{RESET}")
    print(f"  {DIM}Enter adapter source:{RESET}")
    print(f"    {GREEN}[1]{RESET} Local adapter (in epstein_lora_teapotai_tinyteapot/)")
    print(f"    {GREEN}[2]{RESET} HuggingFace adapter (hf://username/adapter-name)")
    print(f"    {GREEN}[3]{RESET} No adapter (base model only)")
    
    while True:
        choice = prompt_with_default(f"\n{BOLD}Adapter source", "1")
        if choice in ("1", "2", "3"):
            break
        print(f"{RED}Invalid choice{RESET}")
    
    if choice == "1":
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        adapter_path = os.path.join(BASE_DIR, "epstein_lora_teapotai_tinyteapot")
        if os.path.isdir(adapter_path) and os.path.isfile(os.path.join(adapter_path, "adapter_config.json")):
            return adapter_path
        else:
            print(f"{YELLOW}No local adapter found, using base model only{RESET}")
            return None
    
    elif choice == "2":
        print(f"\n  {DIM}Enter HuggingFace adapter ID{RESET}")
        print(f"  {GREEN}Examples:{RESET}")
        print(f"    - vinesharma/qlora-adapter-Mistral-7B-gsm8k")
        print(f"    - alignment-handbook/zephyr-7b-dpo-lora")
        print(f"    - TheBananaMan/epstein-full-merged")
        
        adapter_id = args.adapter
        if not adapter_id:
            adapter_id = prompt_with_default(f"\n{BOLD}HuggingFace adapter (hf://)")
        
        if adapter_id.startswith("hf://"):
            adapter_id = adapter_id[5:]
        
        return f"https://huggingface.co/{adapter_id}"
    
    return None

def get_lora_scale():
    if args.lora_scale is not None:
        return args.lora_scale
    
    print(f"\n{BOLD}Step 3: LoRA Scaling Factor{RESET}")
    print(f"  {DIM}0.0 = no LoRA effect, 1.0 = full LoRA effect{RESET}")
    
    while True:
        try:
            user_input = prompt_with_default(f"\n{BOLD}LoRA scale", "0.3")
            val = float(user_input)
            if 0.0 <= val <= 1.0:
                return val
            print(f"{RED}Please enter a value between 0.0 and 1.0{RESET}")
        except ValueError:
            print(f"{RED}Please enter a valid number{RESET}")

def load_model(base_model, adapter_path, lora_scale):
    print(f"\n{DIM}Loading tokenizer...{RESET}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"{DIM}Loading base model: {base_model}...{RESET}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    if adapter_path:
        if adapter_path.startswith("https://huggingface.co/"):
            adapter_id = adapter_path.replace("https://huggingface.co/", "")
            print(f"{DIM}Loading adapter from HuggingFace: {adapter_id}...{RESET}")
        else:
            adapter_id = adapter_path
            print(f"{DIM}Loading local adapter...{RESET}")
        
        model = PeftModel.from_pretrained(base, adapter_id, scaling_factor=lora_scale)
        print(f"{GREEN}✓{RESET} Loaded adapter (scaling={lora_scale})")
    else:
        model = base
        print(f"{YELLOW}Using base model only{RESET}")
    
    model.eval()
    return model, tokenizer

def chat():
    base_model = get_model_choice()
    adapter_path = get_adapter_choice()
    lora_scale = get_lora_scale()
    
    model, tokenizer = load_model(base_model, adapter_path, lora_scale)
    
    print(f"\n{BOLD}{MAGENTA}{'─' * 56}{RESET}")
    print(f"{BOLD}{GREEN}Ready!{RESET} Type your questions below.")
    print(f"{DIM}Type 'quit' or 'exit' to stop.{RESET}\n")
    
    while True:
        try:
            q = input(f"\n{BOLD}{CYAN}You:{RESET} ").strip()
            if q.lower() in ("quit", "exit", "q"):
                print(f"{YELLOW}Goodbye!{RESET}")
                break
            if not q:
                continue
            
            prompt = f"""You are a knowledgeable assistant. Answer the following question accurately.

### Question: {q}

### Answer:"""
            
            print(f"\n{DIM}Thinking...{RESET}")
            print(f"\n{BOLD}{MAGENTA}Assistant:{RESET} ", end="", flush=True)
            
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            with torch.no_grad():
                gen_kwargs = dict(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=3,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer
                )
                thread = Thread(target=model.generate, kwargs=gen_kwargs)
                thread.start()
                
                for text in streamer:
                    print(text, end="", flush=True)
                thread.join()
            
            print("\n")
            
        except KeyboardInterrupt:
            print(f"\n\n{YELLOW}Goodbye!{RESET}")
            break
        except Exception as e:
            print(f"\n{RED}Error: {e}{RESET}")


if __name__ == "__main__":
    chat()
