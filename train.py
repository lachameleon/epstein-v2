#!/usr/bin/env python3
import os
import sys
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import logging
import re
import subprocess

subprocess.run("clear",shell=True,text=True)
print(r'''
                               _
                           ,--.\`-. __
                         _,.`. \:/,"  `-._
                     ,-*" _,.-;-*`-.+"*._ )
                    ( ,."* ,-" / `.  \.  `.
                   ,"   ,;"  ,"\../\  \:   \
                  (   ,"/   / \.,' :   ))  /
                   \  |/   / \.,'  /  // ,'
                    \_)\ ,' \.,'  (  / )/
                        `  \._,'   `"
                           \../
                           \../
                 ~        ~\../           ~~
          ~~          ~~   \../   ~~   ~      ~~
     ~~    ~   ~~  __...---\../-...__ ~~~     ~~
       ~~~~  ~_,--'        \../      `--.__ ~~    ~~
   ~~~  __,--'              `"             `--.__   ~~~
~~  ,--'                                         `--.
   '------......______             ______......------` ~~
 ~~~   ~    ~~      ~ `````---"""""  ~~   ~     ~~
        ~~~~    ~~  ~~~~       ~~~~~~  ~ ~~   ~~ ~~~  ~
     ~~   ~   ~~~     ~~~ ~         ~~       ~~   SSt
              ~        ~~       ~~~       ~

''')

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""

MODEL_QWEN = "Qwen/Qwen2.5-0.5B"
MODEL_STEP = "stepfun-ai/Step-3.5-Flash"
DATASET = "teyler/epstein-files-20k"
MAX_DATASET_SIZE = 50000
MAX_TEXT_LENGTH = 500


def get_lora_target_modules(model):
    model_type = model.config.model_type.lower() if hasattr(model.config, 'model_type') else ""
    
    if "t5" in model_type or "encoder-decoder" in model_type or "seq2seq" in model_type:
        return [
            "SelfAttention.q", "SelfAttention.k", "SelfAttention.v", "SelfAttention.o",
            "EncDecAttention.q", "EncDecAttention.k", "EncDecAttention.v", "EncDecAttention.o",
            "DenseReluDense.wi_0", "DenseReluDense.wi_1", "DenseReluDense.wo",
        ]
    elif "gpt" in model_type or "qwen" in model_type or "llama" in model_type or "mistral" in model_type:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]


def get_task_type(model):
    model_type = model.config.model_type.lower() if hasattr(model.config, 'model_type') else ""
    if "t5" in model_type or "encoder-decoder" in model_type or "seq2seq" in model_type:
        return TaskType.SEQ_CLS
    return TaskType.CAUSAL_LM


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 train.py <model_name>")
        print("Examples:")
        print("  python3 train.py Qwen/Qwen2.5-0.5B")
        print("  python3 train.py stepfun-ai/Step-3.5-Flash")
        print("  python3 train.py facebook/opt-350m")
        sys.exit(1)
    
    model_name = sys.argv[1]
    output_dir = f"./epstein_lora_{model_name.replace('/', '_')}"
    
    logger.info(f"MODEL_NAME: {model_name}")
    logger.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = None
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    
    logger.info(f"Loading dataset: {DATASET}")
    dataset = load_dataset(DATASET, split="train")
    if len(dataset) > MAX_DATASET_SIZE:
        dataset = dataset.select(range(MAX_DATASET_SIZE))
    logger.info(f"Dataset: {len(dataset)} samples")
    
    def format_instruction(example):
        text = example.get("text", "")[:MAX_TEXT_LENGTH]
        prompt = """
        You are a knowledgeable assistant with access to the Epstein files documents. Your task is to answer questions about this information accurately and factually based ONLY on the provided documents. Never fabricate information - only answer based on the provided context. If you don't know the answer based on the documents, say so.
### Document Content:
{text}

### Question: What information can you provide from these documents about the Epstein files?

### Answer:
"""
        return {"formatted_text": prompt}
    
    dataset = dataset.map(format_instruction, remove_columns=list(dataset.features.keys()), desc="Formatting")
    
    target_modules = get_lora_target_modules(model)
    task_type = get_task_type(model)
    logger.info(f"Model type: {model.config.model_type}, using target modules: {target_modules}")
    
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=target_modules,
        bias="none", task_type=task_type,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=5,
        optim="adamw_torch",
        warmup_steps=100,
        lr_scheduler_type="cosine",
        report_to="none",
        max_grad_norm=0.3,
        remove_unused_columns=False,
    )
    
    def formatting_func(example):
        return example["formatted_text"]
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )
    
    total_steps = trainer.get_train_dataloader().__len__() * 3 // training_args.gradient_accumulation_steps
    logger.info(f"TOTAL_STEPS: {total_steps}")
    
    logger.info("Starting training...")
    
    trainer.train()
    
    logger.info(f"Saving adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
