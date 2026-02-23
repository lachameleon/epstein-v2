#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_NAME_STEP = "stepfun-ai/Step-3.5-Flash"
ADAPTER_PATH = "./epstein_lora_adapter"

def load_model_for_inference(model_choice="qwen"):
    if model_choice == "step":
        model_name = MODEL_NAME_STEP
    else:
        model_name = MODEL_NAME
    
    logger.info(f"Loading base model {model_name}...")
    
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
    
    logger.info(f"Loading adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    
    return model, tokenizer

def format_prompt(question: str) -> str:
    return f"""You are a knowledgeable assistant with access to the Epstein files documents. Your task is to answer questions about this information accurately and factually based ONLY on the provided documents. Never fabricate information - only answer based on the documents. If you don't know the answer, say so.

### Question: {question}

### Answer:"""

def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 512) -> str:
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

def chat_loop(model, tokenizer):
    print("\n" + "="*50)
    print("Epstein Files Q&A System")
    print("="*50)
    print("Type your question about the Epstein files.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            question = input("You: ").strip()
            
            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            answer = generate_answer(model, tokenizer, question)
            print(f"\nAssistant: {answer}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")

def main():
    import sys
    model_choice = sys.argv[1] if len(sys.argv) > 1 else "qwen"
    
    model, tokenizer = load_model_for_inference(model_choice)
    
    test_question = "What information is contained in the Epstein files?"
    logger.info(f"Testing with question: {test_question}")
    answer = generate_answer(model, tokenizer, test_question)
    print(f"\nTest Answer: {answer}\n")
    
    chat_loop(model, tokenizer)

if __name__ == "__main__":
    main()
