#!/usr/bin/env python3
import subprocess
import sys
import signal
import os

def handle_interrupt(signum, frame):
    print("\n\nCtrl+C detected. Training will continue in the background.")
    print("You can check progress with: tail -f output.log")
    print("To stop training manually, run: pkill -f 'python3 train.py'")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_interrupt)

subprocess.run("clear", text=True, shell=True)

print("""
Welcome to



░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░      ░▒▓█▓▒░░▒▓███████▓▒░▒▓█▓▒░       ░▒▓██████▓▒░░▒▓███████▓▒░░▒▓███████▓▒░  
   ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░             ░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
   ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░             ░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
   ░▒▓█▓▒░   ░▒▓████████▓▒░▒▓██████▓▒░        ░▒▓█▓▒░░▒▓██████▓▒░░▒▓█▓▒░      ░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
   ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░             ░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
   ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░             ░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
   ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░      ░▒▓█▓▒░▒▓███████▓▒░░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░  

                                                                                                                                                                                                                     
""")

print("Enter the HuggingFace model path you want to fine-tune (or press Enter for default):")

model_input = input("Model: ").strip()

if not model_input:
    model_input = "Qwen/Qwen2.5-0.5B"
    print(f"Using default: {model_input}")

print(f"\nStarting training with model: {model_input}")
print("Output will be written to output.log")
print()
print("Training started in background. Press Ctrl+C to return to terminal (training will continue).")
print("To stop training: pkill -f 'python3 train.py'")
print()

proc = subprocess.Popen(
    f"python3 train.py {model_input} > output.log 2>&1",
    shell=True,
    preexec_fn=os.setsid
)

try:
    proc.wait()
except KeyboardInterrupt:
    print("\nReturning to terminal. Training continues in background.")
    print(f"Process PID: {proc.pid}")
    sys.exit(0)
