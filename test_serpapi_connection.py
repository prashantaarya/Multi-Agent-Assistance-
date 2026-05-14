"""
Test SerpAPI connection and API key
"""
import os
import asyncio
import aiohttp
from dotenv import load_dotenv

load_dotenv()

async def test_serpapi():
    api_key = os.getenv("SERPAPI_KEY")
    
    print("="*60)
    print("Testing SerpAPI Connection")
    print("="*60)
    print(f"\n1. API Key present: {'✅ YES' if api_key else '❌ NO'}")
    
    if api_key:
        print(f"   API Key (first 10 chars): {api_key[:10]}...")
    
    # Test DNS resolution
    print("\n2. Testing DNS resolution for serpapi.com...")
    try:
        import socket
        ip = socket.gethostbyname("serpapi.com")
        print(f"   ✅ Resolved to: {ip}")
    except Exception as e:
        print(f"   ❌ DNS failed: {e}")
        print("\n   DIAGNOSIS: Your network cannot resolve serpapi.com")
        print("   SOLUTIONS:")
        print("   - Check internet connection")
        print("   - Try different DNS (8.8.8.8 or 1.1.1.1)")
        print("   - Check firewall settings")
        return
    
    # Test HTTP connection
    print("\n3. Testing HTTPS connection to serpapi.com...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://serpapi.com/search",
                params={"engine": "google_jobs", "q": "test", "api_key": api_key},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                print(f"   Status: {resp.status}")
                
                if resp.status == 200:
                    data = await resp.json()
                    print(f"   ✅ Connection successful!")
                    print(f"   Search results: {len(data.get('jobs_results', []))} jobs found")
                elif resp.status == 401:
                    print(f"   ❌ Invalid API key")
                elif resp.status == 429:
                    print(f"   ⚠️  Rate limit exceeded (100 searches/month)")
                else:
                    text = await resp.text()
                    print(f"   ❌ Error: {text[:200]}")
    except asyncio.TimeoutError:
        print(f"   ❌ Timeout - serpapi.com not responding")
        print("\n   DIAGNOSIS: Connection timeout")
        print("   SOLUTIONS:")
        print("   - Check if firewall is blocking HTTPS")
        print("   - Try from a different network")
    except aiohttp.ClientConnectorError as e:
        print(f"   ❌ Connection failed: {e}")
        print("\n   DIAGNOSIS: Cannot establish connection to serpapi.com")
        print("   SOLUTIONS:")
        print("   - Check firewall/antivirus blocking outbound connections")
        print("   - Check proxy settings")
        print("   - Try: pip install --upgrade aiohttp certifi")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    
    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(test_serpapi())
