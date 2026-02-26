import os
import re
import asyncio
import logging
from pathlib import Path

import discord
from discord import Embed
from discord import app_commands
from dotenv import load_dotenv

load_dotenv()

LOG_FILE = Path("output.log")
CHECK_INTERVAL = 1.0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

message = None
last_position = 0
training_process = None


def is_admin(interaction: discord.Interaction) -> bool:
    return interaction.user.guild_permissions.administrator


def parse_log_content(content: str) -> dict:
    lines = content.strip().split("\n")
    last_line = lines[-1] if lines else ""
    
    progress_pattern = r'(\d+)%\|([\|█▏▎▍▌▋▊▉])+\|(\d+)/(\d+)\s+\[(\d+:\d+:\d+)<(\d+:\d+:\d+),\s*([\d.]+)s/it\]'
    match = re.search(progress_pattern, last_line)
    
    if match:
        percentage = int(match.group(1))
        current = int(match.group(3))
        total = int(match.group(4))
        elapsed = match.group(5)
        remaining = match.group(6)
        speed = match.group(7)
        
        bar_length = 20
        filled = int(bar_length * percentage / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        return {
            "last_line": last_line,
            "has_progress": True,
            "percentage": percentage,
            "current": current,
            "total": total,
            "elapsed": elapsed,
            "remaining": remaining,
            "speed": speed,
            "bar": bar,
        }
    
    return {"last_line": last_line, "has_progress": False}


def create_embed(parsed: dict) -> Embed:
    last_line = parsed.get("last_line", "")
    
    if parsed.get("has_progress"):
        percentage = parsed["percentage"]
        current = parsed["current"]
        total = parsed["total"]
        remaining = parsed["remaining"]
        speed = parsed["speed"]
        bar = parsed["bar"]
        
        embed = Embed(
            title="🤖 Model Download",
            color=0x2ecc71 if percentage < 100 else 0x27ae60
        )
        
        embed.add_field(
            name="Progress",
            value=f"`{bar}`",
            inline=False
        )
        
        embed.add_field(
            name="Details",
            value=f"**{percentage}%** • {current:,} / {total:,}",
            inline=False
        )
        
        embed.add_field(
            name="Time Remaining",
            value=f"`{remaining}`",
            inline=True
        )
        
        embed.add_field(
            name="Speed",
            value=f"`{speed}s/it`",
            inline=True
        )
        
        embed.set_footer(text=f"Elapsed: {parsed.get('elapsed', 'N/A')}")
        return embed
    
    embed = Embed(
        title="📟 Training Log",
        description=f"```{last_line[:500]}```",
        color=0x3498db
    )
    embed.set_footer(text="Live training output...")
    return embed


def get_current_log_content() -> tuple[str, int]:
    global last_position
    if not LOG_FILE.exists():
        return "", 0

    with open(LOG_FILE, "r", errors="ignore") as f:
        f.seek(last_position)
        new_content = f.read()
        new_position = f.tell()

    if new_content:
        last_position = new_position

    full_content = ""
    if LOG_FILE.exists():
        with open(LOG_FILE, "r", errors="ignore") as f:
            full_content = f.read()

    return full_content, len(new_content) > 0


async def send_or_update_embed(channel):
    global message
    content, has_new = get_current_log_content()

    if not content:
        embed = Embed(title="📟 Training Log", description="Waiting for log output...", color=0x95a5a6)
        embed.set_footer(text="No output.log content yet")
        if message is None:
            message = await channel.send(embed=embed)
        return

    parsed = parse_log_content(content)
    embed = create_embed(parsed)

    if message is None:
        message = await channel.send(embed=embed)
    elif has_new:
        try:
            await message.delete()
            message = await channel.send(embed=embed)
        except discord.NotFound:
            message = await channel.send(embed=embed)


@client.event
async def on_ready():
    logger.info(f"Logged in as {client.user}")

    channel_id = int(os.getenv("CHANNEL_ID"))
    channel = client.get_channel(channel_id)

    if channel is None:
        logger.error(f"Could not find channel {channel_id}")
        await client.close()
        return

    logger.info(f"Monitoring {LOG_FILE} in channel {channel_id}")

    while True:
        try:
            await send_or_update_embed(channel)
        except Exception as e:
            logger.error(f"Error updating embed: {e}")
        await asyncio.sleep(CHECK_INTERVAL)


@tree.command(name="quit", description="Stop the training process (admin only)")
async def quit_command(interaction):
    if not interaction.guild:
        await interaction.response.send_message("❌ This command can only be used in a server.", ephemeral=True)
        return
    
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message("❌ Only administrators can stop training.", ephemeral=True)
        return
    
    await interaction.response.send_message("🛑 Stopping training... (This will stop at next checkpoint)")
    
    with open(LOG_FILE, "a") as f:
        f.write("\n--- TRAINING STOPPED BY ADMIN ---\n")
    
    import signal
    import os
    for pid in os.popen("pgrep -f 'python3 train.py'").read().strip().split('\n'):
        if pid:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except:
                pass


@tree.command(name="status", description="Check current training status")
async def status_command(interaction):
    content, _ = get_current_log_content()
    
    if not content:
        await interaction.response.send_message("📭 No training in progress or no log file found.", ephemeral=True)
        return
    
    parsed = parse_log_content(content)
    embed = create_embed(parsed)
    
    await interaction.response.send_message(embed=embed, ephemeral=True)


@client.event
async def on_message(message):
    if message.author.bot:
        return
    
    if not message.content.startswith("!stop"):
        return
    
    if not message.guild:
        return
    
    if not message.author.guild_permissions.administrator:
        await message.reply("❌ Only administrators can stop training.")
        return
    
    await message.reply("🛑 Stopping training... (This will stop at next checkpoint)")
    
    with open(LOG_FILE, "a") as f:
        f.write("\n--- TRAINING STOPPED BY ADMIN ---\n")
    
    import signal
    import os
    for pid in os.popen("pgrep -f 'python3 train.py'").read().strip().split('\n'):
        if pid:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except:
                pass


async def setup_hook():
    await tree.sync()


def main():
    token = os.getenv("DISCORD_TOKEN")
    if not token or token == "your_discord_token_here":
        logger.error("Please set DISCORD_TOKEN in .env")
        return

    client.setup_hook = setup_hook
    client.run(token)


if __name__ == "__main__":
    main()
