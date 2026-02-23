#!/usr/bin/env python3
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATASET_NAME = "teyler/epstein-files-20k"
OUTPUT_DIR = "./epstein_lora_adapter"

MODEL_NAME_STEP = "stepfun-ai/Step-3.5-Flash"

def load_model_and_tokenizer(model_choice="qwen"):
    if model_choice == "step":
        model_name = MODEL_NAME_STEP
    else:
        model_name = MODEL_NAME
    
    logger.info(f"Loading model {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model.config.use_cache = False
    
    return model, tokenizer

def prepare_dataset(tokenizer):
    logger.info(f"Loading dataset {DATASET_NAME}...")
    
    dataset = load_dataset(DATASET_NAME, split="train")
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    logger.info(f"Sample keys: {dataset[0].keys()}")
    
    def format_instruction(example):
        text = example.get("text", "")
        if not text:
            text = str(example)
        
        prompt = f"""You are a knowledgeable assistant with access to the Epstein files documents. Your task is to answer questions about this information accurately and factually based ONLY on the provided documents. Never fabricate information - only answer based on the provided context. If you don't know the answer based on the documents, say so.

### Document Content:
{text[:1500]}

### Question: What information can you provide from these documents about the Epstein files?

### Answer:"""
        
        return {"formatted_text": prompt}
    
    dataset = dataset.map(
        format_instruction,
        remove_columns=list(dataset.features.keys()),
        desc="Formatting dataset"
    )
    
    return dataset

def train_lora(model_choice="qwen"):
    model, tokenizer = load_model_and_tokenizer(model_choice)
    
    dataset = prepare_dataset(tokenizer)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        optim="paged_adamw_32bit",
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
        peft_config=lora_config,
        args=training_args,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info(f"Saving adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    logger.info("Training complete!")
    return model, tokenizer

if __name__ == "__main__":
    import sys
    model_choice = sys.argv[1] if len(sys.argv) > 1 else "qwen"
    train_lora(model_choice)
