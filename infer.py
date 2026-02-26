#!/usr/bin/env python3
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

MODEL = "teapotai/tinyteapot"
ADAPTER = "./epstein_lora_teapotai_tinyteapot"

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForSeq2SeqLM.from_pretrained(MODEL, device_map="cpu", torch_dtype=torch.float32, trust_remote_code=True)
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

print("Ready! Type 'quit' to exit.\n")

while True:
    try:
        q = input("Question: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue

        prompt = f"""You are a knowledgeable assistant with access to the Epstein files. Answer based ONLY on the provided documents. If you don't know, say so.

### Question: {q}

### Answer:"""

        with torch.no_grad():
            out = model.generate(**tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048), max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=tokenizer.pad_token_id)

        ans = tokenizer.decode(out[0], skip_special_tokens=True)
        if prompt in ans:
            ans = ans[len(prompt):].strip()
        print(f"Answer: {ans}\n")

    except KeyboardInterrupt:
        break
