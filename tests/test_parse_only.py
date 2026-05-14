# Test resume parsing in isolation

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def test_parse_resume():
    print("\n" + "="*80)
    print("TESTING: Can we parse Resume.pdf?")
    print("="*80 + "\n")
    
    from agents.resume_agent import ResumeAgent
    
    # Initialize agent (no LLM for mock mode)
    resume_agent = ResumeAgent(name="resume", model_client=None)
    
    # Test 1: Check if Resume.pdf exists
    resume_path = Path("data/resumes/Resume.pdf")
    if not resume_path.exists():
        print(f"❌ Resume.pdf not found at: {resume_path}")
        print("\nAvailable files in data/resumes/:")
        for f in Path("data/resumes").glob("*"):
            print(f"  - {f.name}")
        return
    
    print(f"✅ Found resume: {resume_path.name}\n")
    
    # Test 2: Try to parse it (mock mode - no LLM)
    print("Testing parse_resume (mock mode - no LLM)...")
    result = await resume_agent.parse_resume(
        user_id="test_user",
        file_path=str(resume_path),
        use_llm=False  # Use simple keyword extraction
    )
    
    print(f"\n{result['response']}\n")
    
    if result['data']:
        print("="*80)
        print("EXTRACTED DATA:")
        print("="*80)
        print(f"Name: {result['data']['personal']['name']}")
        print(f"Email: {result['data']['personal']['email']}")
        print(f"Skills: {', '.join(result['data']['skills'][:10])}")
        print(f"Total Skills: {result['data']['total_skills']}")
        print(f"Experience: {result['data']['total_years']} years")
        print("="*80)
        print("\n✅ Resume parsing works!")
    else:
        print("❌ Failed to extract data from resume")
    
    # Test 3: Auto-detect
    print("\n" + "="*80)
    print("Testing auto_detect_resume...")
    print("="*80 + "\n")
    
    result = await resume_agent.auto_detect_resume(user_id="test_user")
    print(f"{result['response']}\n")
    
    if result['data']:
        print(f"✅ Auto-detected: {result['data'].get('auto_detected_file')}")
    else:
        print("❌ Auto-detect failed")

if __name__ == "__main__":
    asyncio.run(test_parse_resume())
