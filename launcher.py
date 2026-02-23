#!/usr/bin/env python3
import os
import sys
import signal
import subprocess
import time
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

processes = []
discord_proc = None

def signal_handler(sig, frame):
    logger.info("Shutting down...")
    for proc in processes:
        proc.terminate()
    if discord_proc:
        discord_proc.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def start_discord_bot():
    global discord_proc
    logger.info("Starting Discord bot...")
    discord_proc = subprocess.Popen(
        [sys.executable, "discord_bot.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    processes.append(discord_proc)
    
    def log_discord():
        for line in discord_proc.stdout:
            print(f"[DISCORD] {line.rstrip()}")
    
    import threading
    threading.Thread(target=log_discord, daemon=True).start()
    time.sleep(2)

def start_training(model_choice="qwen"):
    logger.info(f"Starting training with model: {model_choice}")
    
    train_proc = subprocess.Popen(
        [sys.executable, "train_runner.py", model_choice],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    processes.append(train_proc)
    
    def log_training():
        for line in train_proc.stdout:
            print(f"[TRAIN] {line.rstrip()}")
            if "ERROR" in line:
                logger.error(f"Training error: {line}")
            elif "Training complete" in line:
                logger.info("Training completed!")
    
    import threading
    threading.Thread(target=log_training, daemon=True).start()

def check_dependencies():
    try:
        import discord
        import gradio
        import transformers
        import peft
        import trl
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False

def main():
    print(f"\n{'='*50}")
    print(f"🚀 Epstein Files V2 - Launcher")
    print(f"{'='*50}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}\n")
    
    if not check_dependencies():
        logger.error("Please install missing dependencies")
        return
    
    start_discord_bot()
    
    print(f"\n{'='*50}")
    print(f"Services started:")
    print(f"  - Discord Bot: Running")
    print(f"  - Gradio UI: python main.py")
    print(f"{'='*50}")
    print(f"\nTo start training, run:")
    print(f"  python train_runner.py qwen")
    print(f"\nOr use the /train command in Discord\n")
    
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
