# tests/test_resume_job_agents.py
"""
Test script for Resume and Job Agents

Tests:
1. Resume parsing (mock mode)
2. Profile retrieval
3. Job searching (mock mode)
4. Job ranking based on resume
5. Job details retrieval
6. Application tracking
7. Job fit analysis

Run: python tests/test_resume_job_agents.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.resume_agent import ResumeAgent
from agents.job_agent import JobAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv

load_dotenv()

async def test_resume_agent():
    """Test Resume Agent capabilities"""
    print("\n" + "="*80)
    print("TESTING RESUME AGENT")
    print("="*80)
    
    # Initialize agent (no model needed for mock mode)
    resume_agent = ResumeAgent(name="resume", model_client=None)
    
    # Test 1: Parse resume (mock mode - no LLM)
    print("\n1. Testing resume parsing (mock mode)...")
    result = await resume_agent.parse_resume(
        user_id="test_user",
        file_path="data/resumes/sample_resume.txt",
        use_llm=False  # Use mock extraction
    )
    print(f"\n{result['response']}")
    if result['data']:
        print(f"Extracted {result['data']['total_skills']} skills")
        print(f"Total experience: {result['data']['total_years']} years")
    
    # Test 2: Get profile
    print("\n2. Testing profile retrieval...")
    result = await resume_agent.get_profile(user_id="test_user")
    print(f"\n{result['response']}")
    
    # Test 3: Analyze fit
    print("\n3. Testing job fit analysis...")
    job_description = """
    Senior Python Developer - 3-5 years experience
    
    Requirements:
    - Strong Python programming (FastAPI, Django)
    - Experience with Docker and Kubernetes
    - AWS cloud experience required
    - PostgreSQL database skills
    - REST API design
    - CI/CD pipeline knowledge
    
    Nice to have:
    - LLM and AI experience
    - AutoGen framework knowledge
    """
    
    result = await resume_agent.analyze_fit(
        user_id="test_user",
        job_description=job_description
    )
    print(f"\n{result['response']}")
    
    # Test 4: Update preferences
    print("\n4. Testing preference update...")
    result = await resume_agent.update_preferences(
        user_id="test_user",
        preferences={
            "locations": ["Delhi", "Bangalore", "Remote"],
            "roles": ["Backend Developer", "Python Developer", "ML Engineer"],
            "min_salary": 1500000,  # 15 LPA in INR
            "job_types": ["Full-time", "Contract"]
        }
    )
    print(f"\n{result['response']}")
    
    return True

async def test_job_agent():
    """Test Job Agent capabilities"""
    print("\n" + "="*80)
    print("TESTING JOB AGENT")
    print("="*80)
    
    # Initialize agent
    job_agent = JobAgent(name="job", model_client=None)
    
    # Test 1: Search jobs (mock mode - no API key needed)
    print("\n1. Testing job search (mock mode)...")
    result = await job_agent.search_jobs(
        query="Python Developer",
        location="Delhi, India",
        num_results=5
    )
    print(f"\n{result['response']}")
    if result['data']:
        print(f"\nFound {result['data']['total']} jobs")
        print(f"First job: {result['data']['jobs'][0]['title']} at {result['data']['jobs'][0]['company']}")
    
    # Test 2: Rank jobs (requires resume profile)
    print("\n2. Testing job ranking...")
    result = await job_agent.rank_jobs(
        user_id="test_user",
        min_score=40  # Lower threshold for testing
    )
    print(f"\n{result['response']}")
    if result['data'] and result['data']['total'] > 0:
        print(f"\nTop match: {result['data']['ranked_jobs'][0]['title']} - {result['data']['ranked_jobs'][0]['match_score']}%")
    
    # Test 3: Get job details
    print("\n3. Testing job details retrieval...")
    result = await job_agent.get_job_details(job_id="mock_001")
    print(f"\n{result['response']}")
    
    # Test 4: Track application
    print("\n4. Testing application tracking...")
    result = await job_agent.track_application(
        user_id="test_user",
        job_id="mock_001",
        status="applied"
    )
    print(f"\n{result['response']}")
    
    return True

async def test_end_to_end_workflow():
    """Test complete job search workflow"""
    print("\n" + "="*80)
    print("END-TO-END WORKFLOW TEST")
    print("="*80)
    
    resume_agent = ResumeAgent(name="resume", model_client=None)
    job_agent = JobAgent(name="job", model_client=None)
    
    # Step 1: Parse resume
    print("\nStep 1: Parse resume...")
    result = await resume_agent.parse_resume(
        user_id="workflow_user",
        file_path="data/resumes/sample_resume.txt",
        use_llm=False
    )
    print(f"✅ Resume parsed: {result['data']['total_skills']} skills extracted")
    
    # Step 2: Search for jobs
    print("\nStep 2: Search for jobs...")
    result = await job_agent.search_jobs(
        query="Backend Developer Python",
        location="Delhi",
        num_results=10
    )
    print(f"✅ Found {result['data']['total']} jobs")
    
    # Step 3: Rank jobs by match
    print("\nStep 3: Rank jobs by resume match...")
    result = await job_agent.rank_jobs(
        user_id="workflow_user",
        min_score=50
    )
    if result['data'] and result['data']['total'] > 0:
        print(f"✅ Ranked {result['data']['total']} jobs (avg match: {result['data']['average_match']}%)")
        
        # Step 4: Get details for top match
        top_job_id = result['data']['ranked_jobs'][0]['job_id']
        top_job_title = result['data']['ranked_jobs'][0]['title']
        top_score = result['data']['ranked_jobs'][0]['match_score']
        
        print(f"\nStep 4: Get details for top match ({top_job_title} - {top_score}% match)...")
        result = await job_agent.get_job_details(job_id=top_job_id)
        print(f"✅ Retrieved job details")
        
        # Step 5: Analyze fit for top job
        print(f"\nStep 5: Analyze resume fit for {top_job_title}...")
        result = await resume_agent.analyze_fit(
            user_id="workflow_user",
            job_description=result['data']['description']
        )
        print(f"✅ Match analysis: {result['data']['match_score']}% match")
        
        # Step 6: Track application
        print(f"\nStep 6: Track application to {top_job_title}...")
        result = await job_agent.track_application(
            user_id="workflow_user",
            job_id=top_job_id,
            status="applied"
        )
        print(f"✅ Application tracked")
    else:
        print("⚠️ No jobs met minimum match score")
    
    print("\n" + "="*80)
    print("WORKFLOW COMPLETE!")
    print("="*80)
    
    return True

async def main():
    """Run all tests"""
    print("\n🧪 RESUME & JAR AGENT TEST SUITE")
    print("="*80)
    
    try:
        # Test Resume Agent
        await test_resume_agent()
        
        # Test Job Agent
        await test_job_agent()
        
        # Test end-to-end workflow
        await test_end_to_end_workflow()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        
        print("\n📊 Summary:")
        print("  - Resume Agent: 4 capabilities tested")
        print("  - Job Agent: 4 capabilities tested")
        print("  - End-to-end workflow: 6 steps completed")
        
        print("\n📁 Generated files:")
        print("  - data/user_profiles.json (user profiles)")
        print("  - data/job_listings.json (job search results)")
        print("  - data/applications.json (application tracking)")
        
        print("\n🚀 Next steps:")
        print("  1. Add SERPAPI_KEY to .env for real job search")
        print("  2. Install PyPDF2 for PDF resume parsing: pip install PyPDF2")
        print("  3. Install python-docx for DOCX parsing: pip install python-docx")
        print("  4. Test via API: POST /ask-direct with 'find Python jobs in Delhi'")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(main())
