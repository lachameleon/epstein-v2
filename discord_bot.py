#!/usr/bin/env python3
import os
import re
import asyncio
import logging
import threading
import discord
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from discord import app_commands, Embed, Colour
from discord.ext import commands

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv("discord_bot.env")

TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = int(os.getenv("GUILD_ID", "0"))
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "0"))
WATCH_LOG = os.getenv("WATCH_LOG", "train2.log")
NOTIFY_START = os.getenv("NOTIFY_START", "true").lower() == "true"
NOTIFY_COMPLETE = os.getenv("NOTIFY_COMPLETE", "true").lower() == "true"
NOTIFY_ERROR = os.getenv("NOTIFY_ERROR", "true").lower() == "true"
NOTIFY_STEP = os.getenv("NOTIFY_STEP", "true").lower() == "true"
STEP_INTERVAL = int(os.getenv("STEP_INTERVAL", "50"))

intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

log_watcher = None
last_position = 0
training_start_time = None
current_step = 0
total_steps = 0
loss_values = []
is_training = False


class LogWatcher:
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.last_position = 0
        self.running = False
        self.thread = None
        
        if self.log_file.exists():
            self.last_position = self.log_file.stat().st_size
            
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started watching {self.log_file}")
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Stopped log watcher")
        
    def _watch_loop(self):
        while self.running:
            try:
                if self.log_file.exists():
                    current_size = self.log_file.stat().st_size
                    
                    if current_size > self.last_position:
                        with open(self.log_file, 'r') as f:
                            f.seek(self.last_position)
                            new_lines = f.readlines()
                            self.last_position = current_size
                            
                        for line in new_lines:
                            self._parse_line(line.strip())
                            
            except Exception as e:
                logger.error(f"Error watching log: {e}")
                
            asyncio.run(asyncio.sleep(1))
    
    def _parse_line(self, line: str):
        global training_start_time, current_step, total_steps, loss_values, is_training
        
        if "Starting training" in line or "trainer.train()" in line:
            if NOTIFY_START and not is_training:
                is_training = True
                training_start_time = datetime.now()
                asyncio.create_task(self.send_embed(
                    title="🚀 Training Started",
                    description="LoRA training has begun on the Epstein files dataset",
                    color=Colour.green(),
                    fields=[
                        ("Model", "Qwen/Qwen2.5-0.5B", True),
                        ("Dataset", "teyler/epstein-files-20k", True),
                    ]
                ))
                
        elif "Training complete" in line or "trainer.train() completed" in line:
            if NOTIFY_COMPLETE and is_training:
                is_training = False
                elapsed = datetime.now() - training_start_time if training_start_time else None
                elapsed_str = str(elapsed).split('.')[0] if elapsed else "Unknown"
                
                avg_loss = sum(loss_values) / len(loss_values) if loss_values else 0
                
                asyncio.create_task(self.send_embed(
                    title="✅ Training Complete",
                    description="LoRA adapter has been saved successfully",
                    color=Colour.blue(),
                    fields=[
                        ("Duration", elapsed_str, True),
                        ("Steps", str(current_step), True),
                        ("Avg Loss", f"{avg_loss:.4f}", True),
                        ("Output", "./epstein_lora_adapter", False),
                    ]
                ))
                
                current_step = 0
                loss_values = []
                training_start_time = None
                
        elif "Step" in line and "/" in line:
            match = re.search(r"Step.*?(\d+)/(\d+)", line)
            if match:
                step = int(match.group(1))
                total = int(match.group(2))
                
                if NOTIFY_STEP and step % STEP_INTERVAL == 0 and step != current_step:
                    current_step = step
                    total_steps = total
                    
                    progress = (step / total) * 100
                    
                    asyncio.create_task(self.send_embed(
                        title=f"📊 Training Progress - Step {step}/{total}",
                        description=f"{'█' * int(progress/5)}{'░' * (20 - int(progress/5))} {progress:.1f}%",
                        color=Colour.purple(),
                        fields=[
                            ("Progress", f"{step}/{total} ({progress:.1f}%)", True),
                        ]
                    ))
                    
        elif "loss" in line.lower():
            match = re.search(r"loss[:\s=]+([0-9.]+)", line, re.IGNORECASE)
            if match:
                loss = float(match.group(1))
                loss_values.append(loss)
                
        elif "error" in line.lower() or "exception" in line.lower() or "traceback" in line:
            if NOTIFY_ERROR:
                asyncio.create_task(self.send_embed(
                    title="❌ Training Error",
                    description=f"```\n{line[:500]}\n```",
                    color=Colour.red(),
                    fields=[
                        ("Log", WATCH_LOG, True),
                    ]
                ))
                
        elif "Loading model" in line:
            asyncio.create_task(self.send_embed(
                title="📥 Loading Model",
                description=line,
                color=Colour.orange(),
            ))
            
        elif "Saving adapter" in line or "model.save_pretrained" in line:
            asyncio.create_task(self.send_embed(
                title="💾 Saving Model",
                description="Saving the trained adapter...",
                color=Colour.yellow(),
            ))
    
    async def send_embed(self, title: str, description: str, color: Colour, fields=None):
        try:
            channel = client.get_channel(CHANNEL_ID)
            if not channel:
                logger.error(f"Channel {CHANNEL_ID} not found")
                return
                
            embed = Embed(
                title=title,
                description=description,
                color=color,
                timestamp=datetime.now()
            )
            
            if fields:
                for name, value, inline in fields:
                    embed.add_field(name=name, value=value, inline=inline)
            
            embed.set_footer(text="Epstein Files V2 • Log Monitor")
            
            await channel.send(embed=embed)
            logger.info(f"Sent embed: {title}")
            
        except Exception as e:
            logger.error(f"Error sending embed: {e}")


@tree.command(name="status", guild=discord.Object(id=GUILD_ID))
async def status_command(interaction):
    """Check current training status"""
    global is_training, current_step, total_steps, loss_values, training_start_time
    
    if is_training:
        elapsed = datetime.now() - training_start_time if training_start_time else None
        elapsed_str = str(elapsed).split('.')[0] if elapsed else "Unknown"
        
        embed = Embed(
            title="📊 Training Status",
            description="Training is currently in progress",
            color=Colour.purple(),
            timestamp=datetime.now()
        )
        embed.add_field(name="Current Step", value=str(current_step), inline=True)
        embed.add_field(name="Total Steps", value=str(total_steps), inline=True)
        embed.add_field(name="Elapsed", value=elapsed_str, inline=True)
        
        if loss_values:
            embed.add_field(name="Latest Loss", value=f"{loss_values[-1]:.4f}", inline=True)
            embed.add_field(name="Avg Loss", value=f"{sum(loss_values)/len(loss_values):.4f}", inline=True)
    else:
        embed = Embed(
            title="✅ Idle",
            description="No training in progress",
            color=Colour.green(),
            timestamp=datetime.now()
        )
    
    await interaction.response.send_message(embed=embed)


@tree.command(name="logs", guild=discord.Object(id=GUILD_ID))
async def logs_command(interaction):
    """Get recent log entries"""
    try:
        if Path(WATCH_LOG).exists():
            with open(WATCH_LOG, 'r') as f:
                lines = f.readlines()[-20:]
            
            content = "".join(lines)
            if len(content) > 2000:
                content = content[:2000] + "..."
                
            embed = Embed(
                title="📋 Recent Logs",
                description=f"```\n{content}\n```",
                color=Colour.blue(),
                timestamp=datetime.now()
            )
        else:
            embed = Embed(
                title="❌ No Logs",
                description=f"Log file {WATCH_LOG} not found",
                color=Colour.red(),
            )
            
        await interaction.response.send_message(embed=embed)
        
    except Exception as e:
        await interaction.response.send_message(f"Error: {e}")


@tree.command(name="restart", guild=discord.Object(id=GUILD_ID))
async def restart_command(interaction):
    """Restart the log watcher"""
    global log_watcher
    
    if log_watcher:
        log_watcher.stop()
        log_watcher.start()
        
    embed = Embed(
        title="🔄 Log Watcher Restarted",
        description=f"Now watching: {WATCH_LOG}",
        color=Colour.green(),
        timestamp=datetime.now()
    )
    
    await interaction.response.send_message(embed=embed)


@tree.command(name="setlog", guild=discord.Object(id=GUILD_ID))
async def setlog_command(interaction, filename: str):
    """Change the log file being watched"""
    global log_watcher, last_position
    
    if not Path(filename).exists():
        await interaction.response.send_message(f"File {filename} not found!")
        return
    
    last_position = Path(filename).stat().st_size
    
    if log_watcher:
        log_watcher.stop()
        log_watcher = LogWatcher(filename)
        log_watcher.start()
    
    embed = Embed(
        title="📁 Log File Changed",
        description=f"Now watching: {filename}",
        color=Colour.blue(),
        timestamp=datetime.now()
    )
    
    await interaction.response.send_message(embed=embed)


@client.event
async def on_ready():
    global log_watcher
    
    logger.info(f"Logged in as {client.user}")
    
    try:
        tree.copy_global_to(guild=discord.Object(id=GUILD_ID))
        await tree.sync(guild=discord.Object(id=GUILD_ID))
        logger.info("Commands synced")
    except Exception as e:
        logger.error(f"Error syncing commands: {e}")
    
    log_watcher = LogWatcher(WATCH_LOG)
    log_watcher.start()
    
    embed = Embed(
        title="🤖 Log Monitor Online",
        description="Watching for training updates",
        color=Colour.green(),
        timestamp=datetime.now()
    )
    embed.add_field(name="Log File", value=WATCH_LOG, inline=True)
    embed.add_field(name="Channel", value=f"<#{CHANNEL_ID}>", inline=True)
    embed.set_footer(text="Epstein Files V2")
    
    channel = client.get_channel(CHANNEL_ID)
    if channel:
        try:
            await channel.send(embed=embed)
        except Exception as e:
            logger.warning(f"Could not send startup message: {e}")


def main():
    if not TOKEN or TOKEN == "your_bot_token_here":
        logger.error("Please set DISCORD_TOKEN in discord_bot.env")
        return
        
    if not CHANNEL_ID:
        logger.error("Please set CHANNEL_ID in discord_bot.env")
        return
        
    client.run(TOKEN)


if __name__ == "__main__":
    import discord
    main()
