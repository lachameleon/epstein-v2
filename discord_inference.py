#!/usr/bin/env python3
import os
import re
import torch
from pathlib import Path
from threading import Thread

import discord
from discord import app_commands
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

load_dotenv()

BASE_MODEL = "teapotai/tinyteapot"
ADAPTER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "epstein_lora_teapotai_tinyteapot", "checkpoint-37500")

print(f"Loading model from: {ADAPTER_PATH}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    device_map="cpu",
    torch_dtype=torch.float32,
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, scaling_factor=0.3)
model.eval()

print("Model loaded successfully!")

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)


def generate_response(prompt: str) -> str:
    full_prompt = f"""You are a knowledgeable assistant. Answer the following question accurately.

### Question: {prompt}

### Answer:"""

    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = response.replace(full_prompt, "").strip()
    return response


@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    await tree.sync()


@tree.command(name="ping", description="Test if the bot is responding")
async def ping(interaction):
    await interaction.response.send_message("Pong! I'm online and ready to answer questions.")


@client.event
async def on_message(message):
    if message.author.bot:
        return

    content = message.content.strip()
    
    if message.reference and message.reference.message_id:
        try:
            referenced_msg = await message.channel.fetch_message(message.reference.message_id)
            if referenced_msg.author.id == client.user.id:
                user_question = referenced_msg.content
                async with message.channel.typing():
                    response = await generate_response_async(user_question)
                await message.reply(response[:2000] if len(response) > 2000 else response)
                return
        except Exception as e:
            await message.reply(f"Error: {e}")
            return

    if client.user in message.mentions:
        clean_content = re.sub(r'<@!?\d+>|<@&\d+>', '', content).strip()
        if not clean_content:
            await message.reply("Hello! What would you like to ask me?")
            return

        async with message.channel.typing():
            try:
                response = generate_response(clean_content)
                await message.reply(response[:2000] if len(response) > 2000 else response)
            except Exception as e:
                await message.reply(f"Error generating response: {e}")


async def setup_hook():
    await tree.sync()


def main():
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        print("Error: DISCORD_TOKEN not found in .env")
        return

    client.setup_hook = setup_hook
    client.run(token)


if __name__ == "__main__":
    main()
