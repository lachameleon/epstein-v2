#!/usr/bin/env python3
import os
import sys
import logging
import torch
from dataclasses import dataclass, field
from typing import Optional
import subprocess
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

subprocess.run("clear",text=True, shell=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_NAME_STEP = "stepfun-ai/Step-3.5-Flash"
ADAPTER_PATH = "./epstein_lora_adapter"
DATASET_NAME = "teyler/epstein-files-20k"
OUTPUT_DIR = "./epstein_lora_adapter"

model = None
tokenizer = None
model_loaded = False


def load_model_and_tokenizer(model_choice: str = "qwen"):
    global model, tokenizer, model_loaded
    
    if model_choice == "step":
        model_name = MODEL_NAME_STEP
    else:
        model_name = MODEL_NAME
    
    logger.info(f"Loading model {model_name}...")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    
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
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if os.path.exists(ADAPTER_PATH):
        logger.info(f"Loading adapter from {ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    else:
        logger.warning("No adapter found, using base model")
        model = base_model
    
    model.eval()
    model_loaded = True
    
    return model, tokenizer


def format_prompt(question: str) -> str:
    return f"""You are a knowledgeable assistant with access to the Epstein files documents. Your task is to answer questions about this information accurately and factually based ONLY on the provided documents. Never fabricate information - only answer based on the documents. If you don't know the answer, say so.

### Question: {question}

### Answer:"""


def generate_answer(question: str, max_new_tokens: int = 512) -> str:
    global model, tokenizer, model_loaded
    
    if not model_loaded:
        load_model_and_tokenizer()
    
    prompt = format_prompt(question)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    answer_start = response.find("### Answer:") + len("### Answer:")
    answer = response[answer_start:].strip()
    
    return answer


def train_model(model_choice: str = "qwen", progress=None):
    global model, tokenizer
    
    logger.info("Starting training...")
    
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer
    
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
    
    logger.info(f"Loading dataset {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split="train")
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
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
        args=training_args,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )
    
    trainer.train()
    
    logger.info(f"Saving adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    logger.info("Training complete!")
    
    global model_loaded
    model_loaded = False
    load_model_and_tokenizer(model_choice)
    
    return "Training complete! The model has been trained and saved."


def get_css():
    return """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .gradio-container {
        max-width: 900px !important;
        margin: auto !important;
        background: #0f0f1a !important;
        min-height: 100vh !important;
    }
    
    .main {
        background: #0f0f1a !important;
        padding: 20px !important;
    }
    
    /* Hide Gradio branding */
    .gradio-html span:not(.font-bold), 
    .gradio-html a, 
    footer, 
    .footer, 
    .with-loader,
    [class*="prose"],
    [class*="logo"],
    [class*="brand"] {
        display: none !important;
    }
    
    /* Remove bottom padding and footer */
    .gradio-container > .footer {
        display: none !important;
    }
    .gradio-container > .main > .contain > :last-child {
        display: none !important;
    }
    
    /* Dark theme overrides */
    .dark .prose {
        color: #e0e0e0 !important;
    }
    
    /* Tab styling */
    .tab-nav {
        background: #1a1a2e !important;
        border-radius: 12px !important;
    }
    
    .tab-item {
        background: transparent !important;
        color: #888 !important;
        border: none !important;
    }
    
    .tab-item.selected {
        background: #667eea !important;
        color: white !important;
    }
    
    /* Input styling */
    input, textarea {
        background: #1a1a2e !important;
        border: 1px solid #333 !important;
        color: #e0e0e0 !important;
        border-radius: 8px !important;
    }
    
    input:focus, textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Button styling */
    button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    
    button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
    
    button.secondary {
        background: #2a2a3e !important;
    }
    
    /* Card styling */
    .gradio-card {
        background: #1a1a2e !important;
        border: 1px solid #2a2a3e !important;
        border-radius: 12px !important;
    }
    
    /* Slider styling */
    input[type="range"] {
        background: #2a2a3e !important;
    }
    
    input[type="range"]::-webkit-slider-thumb {
        background: #667eea !important;
    }
    
    /* Radio button styling */
    .radio-group label {
        color: #e0e0e0 !important;
    }
    
    /* Examples styling */
    .examples {
        background: #1a1a2e !important;
        border-radius: 8px !important;
    }
    
    /* Header styling */
    .header-title {
        text-align: center;
        font-size: 2.5em;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5em;
    }
    .header-subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #888;
        margin-bottom: 1.5em;
    }
    
    /* Status box */
    .status-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    
    /* Label styling */
    .label {
        color: #888 !important;
        font-size: 0.9em !important;
    }
    
    /* Remove dark mode toggle */
    #theme-toggle, [data-testid="theme-toggle"] {
        display: none !important;
    }
    
    /* Compact spacing */
    .gap-2, .gap-4 {
        gap: 12px !important;
    }
    
    /* Loading animation */
    .loader {
        border-color: #667eea !important;
        border-top-color: transparent !important;
    }
    """


def get_js():
    return """
    function() {
        setInterval(function() {
            var footer = document.querySelector('footer');
            if (footer) footer.style.display = 'none';
            var gradioLogo = document.querySelector('[class*="gradio-logo"]');
            if (gradioLogo) gradioLogo.style.display = 'none';
            var links = document.querySelectorAll('a[href*="gradio"]');
            links.forEach(function(link) { link.style.display = 'none'; });
        }, 100);
        setTimeout(function() { clearInterval(); }, 3000);
    }
    """


def create_ui():
    import gradio as gr
    css = get_css()
    js = get_js()
    
    with gr.Blocks(title="Epstein Files V2") as demo:
        gr.Markdown("""
        <div class="header-title">🗂️ Epstein Files V2</div>
        <div class="header-subtitle">AI-Powered Document Q&A System</div>
        """)
        
        with gr.Tab("💬 Ask Questions"):
            with gr.Row():
                with gr.Column(scale=3):
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask about the Epstein files...",
                        lines=3
                    )
                with gr.Column(scale=1):
                    max_tokens = gr.Slider(
                        minimum=100,
                        maximum=2048,
                        value=512,
                        step=100,
                        label="Max Tokens"
                    )
            
            with gr.Row():
                model_select = gr.Radio(
                    ["qwen", "step"],
                    label="Model",
                    value="qwen",
                    interactive=True
                )
            
            submit_btn = gr.Button("🔍 Get Answer", variant="primary")
            
            answer_output = gr.Textbox(
                label="Answer",
                lines=8,
                interactive=False
            )
            
            with gr.Row():
                gr.Examples(
                    examples=[
                        ["What information is contained in the Epstein files?"],
                        ["Who are the key figures mentioned in the documents?"],
                        ["What locations are referenced in the Epstein files?"],
                    ],
                    inputs=question_input,
                )
            
            submit_btn.click(
                fn=lambda q, mt, m: generate_answer(q, int(mt)),
                inputs=[question_input, max_tokens, model_select],
                outputs=answer_output,
            )
            
            question_input.submit(
                fn=lambda q, mt, m: generate_answer(q, int(mt)),
                inputs=[question_input, max_tokens, model_select],
                outputs=answer_output,
            )
        
        with gr.Tab("⚙️ Train Model"):
            gr.Markdown("""
            ### Train a New Model
            Fine-tune the model on the Epstein files dataset.
            """)
            
            with gr.Row():
                train_model_select = gr.Radio(
                    ["qwen", "step"],
                    label="Base Model",
                    value="qwen"
                )
            
            train_btn = gr.Button("🚀 Start Training", variant="primary")
            
            train_status = gr.Textbox(
                label="Training Status",
                lines=10,
                interactive=False
            )
            
            train_btn.click(
                fn=lambda m: train_model(m),
                inputs=[train_model_select],
                outputs=[train_status],
            )
        
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ### About Epstein Files V2
            
            This system uses a fine-tuned Large Language Model to answer questions about the Epstein files.
            
            **Features:**
            - Ask questions about the Epstein documents
            - Fine-tune models on your own data
            - Support for multiple base models (Qwen, Step)
            
            **Models:**
            - Qwen2.5-0.5B (default)
            - Step-3.5-Flash
            
            The model is trained using LoRA (Low-Rank Adaptation) for efficient fine-tuning.
            """)
    
    return demo


def get_local_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


def main():
    import gradio as gr
    local_ip = get_local_ip()
    print(f"\n" + "="*50)
    print(f"🚀 Epstein Files V2")
    print(f"="*50)
    print(f"Local:   http://localhost:7860")
    print(f"Network: http://{local_ip}:7860")
    print(f"="*50 + "\n")
    
    demo = create_ui()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=get_css(),
        theme=gr.themes.Soft(),
        js=get_js(),
    )


if __name__ == "__main__":
    main()
