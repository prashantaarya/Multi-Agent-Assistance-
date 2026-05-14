# tests/test_auto_detect.py
"""
Quick test for auto-detect resume feature
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.resume_agent import ResumeAgent

async def test_auto_detect():
    print("\n🧪 Testing Auto-Detect Resume Feature\n")
    
    resume_agent = ResumeAgent(name="resume", model_client=None)
    
    # Test 1: Auto-detect without any parameters
    print("1. Testing auto-detect (should find Resume.pdf)...")
    result = await resume_agent.auto_detect_resume()
    
    print(f"\n{result['response']}")
    
    if result['data']:
        print(f"\n✅ Success! Found file: {result['data'].get('auto_detected_file')}")
        print(f"   Skills extracted: {result['data']['total_skills']}")
        print(f"   Experience: {result['data']['total_years']} years")
    else:
        print("\n❌ Failed to auto-detect resume")
    
    return result

if __name__ == "__main__":
    asyncio.run(test_auto_detect())
