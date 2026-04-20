"""
Quick script to get your Telegram Chat ID.

Steps:
1. Message your bot on Telegram (any message)
2. Run this script
3. Copy the chat_id and add it to your .env file
"""

import asyncio
import aiohttp
import os
from dotenv import load_dotenv

load_dotenv()

async def get_chat_id():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN not found in .env file")
        return
    
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
    
    if not data.get("ok"):
        print(f"❌ Error: {data.get('description')}")
        return
    
    updates = data.get("result", [])
    if not updates:
        print("📭 No messages found. Send a message to your bot first!")
        return
    
    print(f"\n✅ Found {len(updates)} message(s):\n")
    for update in updates:
        msg = update.get("message", {})
        chat = msg.get("chat", {})
        from_user = msg.get("from", {})
        
        print(f"📨 From: {from_user.get('first_name', 'Unknown')} (@{from_user.get('username', 'N/A')})")
        print(f"   Chat ID: {chat.get('id')}")
        print(f"   Message: {msg.get('text', '(non-text)')}")
        print()
    
    # Show the first chat ID
    first_chat_id = updates[0].get("message", {}).get("chat", {}).get("id")
    if first_chat_id:
        print(f"\n💡 Add this to your .env file:")
        print(f"   TELEGRAM_DEFAULT_CHAT_ID={first_chat_id}\n")

if __name__ == "__main__":
    asyncio.run(get_chat_id())
