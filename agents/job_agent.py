# agents/job_agent.py
"""
Job Agent - Search and match jobs based on user resume

Capabilities:
- search_jobs: Search for jobs using Google Jobs (SerpAPI)
- rank_jobs: Rank jobs by resume match score
- get_job_details: Get full details for a specific job
- track_application: Log job application

Integrates with Resume Agent for intelligent matching.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from autogen_agentchat.agents import AssistantAgent
from tools.job_search import JobSearcher
from tools.resume_parser import ResumeParser
from core.capabilities import register
from core.schemas import ToolParameter, ParameterType

logger = logging.getLogger("jarvis.job_agent")

class JobAgent(AssistantAgent):
    """
    Job Agent for searching and matching jobs.
    Works with Resume Agent for intelligent matching.
    """
    
    def __init__(self, name="job", model_client=None):
        super().__init__(
            name=name,
            model_client=model_client,
            system_message=(
                "You are the Job Agent. You help users search for jobs, "
                "rank them by match score, and track applications.\n\n"
                "⚠️ CRITICAL ANTI-HALLUCINATION RULES:\n"
                "1. NEVER EVER make up job listings, companies, or URLs\n"
                "2. ALWAYS call search_jobs tool when user asks to find jobs\n"
                "3. Do NOT invent job titles, salaries, or application links\n"
                "4. If you don't have job data, call search_jobs first\n\n"
                "WORKFLOW:\n"
                "When user asks 'find jobs':\n"
                "Step 1: Call search_jobs(query='...', location='...')\n"
                "Step 2: Wait for REAL data from API\n"
                "Step 3: Present the ACTUAL job listings (don't make them up!)\n\n"
                "ONLY present job data after calling the tool and getting results!"
            )
        )
        self.searcher = JobSearcher()
        self.parser = ResumeParser()  # For accessing user profiles
        self.applications_file = Path("data/applications.json")
        self.applications_file.parent.mkdir(parents=True, exist_ok=True)
        
        # ──────────────────────────────────────────────────────────────────────
        # Register 4 Job Capabilities
        # ──────────────────────────────────────────────────────────────────────
        
        # 1. Search Jobs
        register(
            capability="job.search",
            agent_name=self.name,
            handler=self.search_jobs,
            description="Search for jobs using Google Jobs based on keywords and location",
            parameters=[
                ToolParameter(
                    name="query",
                    type=ParameterType.STRING,
                    description="Job search query (e.g., 'Python Developer', 'ML Engineer')",
                    required=True
                ),
                ToolParameter(
                    name="location",
                    type=ParameterType.STRING,
                    description="Location filter (e.g., 'Delhi', 'Bangalore', 'Remote')",
                    required=False
                ),
                ToolParameter(
                    name="num_results",
                    type=ParameterType.INTEGER,
                    description="Number of results to return (default: 10)",
                    required=False
                )
            ]
        )
        
        # 2. Rank Jobs
        register(
            capability="job.rank",
            agent_name=self.name,
            handler=self.rank_jobs,
            description="Rank job listings by match score based on user's resume",
            parameters=[
                ToolParameter(
                    name="user_id",
                    type=ParameterType.STRING,
                    description="User identifier",
                    required=True
                ),
                ToolParameter(
                    name="job_ids",
                    type=ParameterType.ARRAY,
                    description="List of job IDs to rank (from search results)",
                    required=False
                ),
                ToolParameter(
                    name="min_score",
                    type=ParameterType.INTEGER,
                    description="Minimum match score to include (0-100, default: 50)",
                    required=False
                )
            ]
        )
        
        # 3. Get Job Details
        register(
            capability="job.get_details",
            agent_name=self.name,
            handler=self.get_job_details,
            description="Get full details for a specific job",
            parameters=[
                ToolParameter(
                    name="job_id",
                    type=ParameterType.STRING,
                    description="Job ID to retrieve",
                    required=True
                )
            ]
        )
        
        # 4. Track Application
        register(
            capability="job.track_application",
            agent_name=self.name,
            handler=self.track_application,
            description="Log that user applied to a job",
            parameters=[
                ToolParameter(
                    name="user_id",
                    type=ParameterType.STRING,
                    description="User identifier",
                    required=True
                ),
                ToolParameter(
                    name="job_id",
                    type=ParameterType.STRING,
                    description="Job ID user applied to",
                    required=True
                ),
                ToolParameter(
                    name="status",
                    type=ParameterType.STRING,
                    description="Application status (applied/submitted/pending)",
                    required=False
                )
            ]
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # CAPABILITY: search_jobs
    # ═══════════════════════════════════════════════════════════════════════
    
    async def search_jobs(
        self,
        query: str,
        location: str = "",
        num_results: int = 10
    ) -> Dict[str, Any]:
        """
        Search for jobs using Google Jobs via SerpAPI.
        
        SMART BEHAVIOR: Automatically parses user's resume if available,
        then searches for jobs matching their profile.
        
        Returns:
            {
                "response": "Found 10 jobs...",
                "data": {
                    "jobs": [...],
                    "total": 10,
                    "query": "...",
                    "location": "...",
                    "matched_with_resume": true/false,
                    "user_profile": {...} if resume found
                }
            }
        """
        try:
            # STEP 1: Try to auto-detect and parse user's resume
            user_profile = None
            resume_info = ""
            
            try:
                # Auto-detect resume in data/resumes/
                from pathlib import Path
                resumes_dir = Path("data/resumes")
                if resumes_dir.exists():
                    resume_files = list(resumes_dir.glob("*.pdf")) + list(resumes_dir.glob("*.docx")) + list(resumes_dir.glob("*.txt"))
                    
                    if resume_files:
                        # Get most recent resume
                        latest_resume = max(resume_files, key=lambda p: p.stat().st_mtime)
                        logger.info(f"🔍 Auto-detected resume: {latest_resume.name}")
                        
                        # Parse it silently
                        parse_result = await self.parser.parse_resume(
                            file_path=str(latest_resume),
                            user_id="default_user"
                        )
                        
                        if parse_result.get("data"):
                            user_profile = parse_result["data"]
                            user_name = user_profile.get("personal", {}).get("name", "Unknown")
                            user_skills = user_profile.get("skills", [])
                            resume_info = f"\n📄 Using resume: **{user_name}** ({len(user_skills)} skills detected)"
                            logger.info(f"✅ Parsed resume for: {user_name}")
            except Exception as e:
                logger.warning(f"Could not auto-parse resume: {e}")
            
            # STEP 2: Search for jobs
            jobs = await self.searcher.search_jobs(query, location, num_results)
            
            if not jobs:
                return {
                    "response": f"ℹ️ No jobs found for: **{query}** in {location or 'any location'}{resume_info}",
                    "data": {
                        "jobs": [],
                        "total": 0,
                        "query": query,
                        "location": location,
                        "matched_with_resume": user_profile is not None,
                        "user_profile": user_profile
                    }
                }
            
            # STEP 3: If we have user profile, calculate match scores
            if user_profile:
                for job in jobs:
                    match_score = self._calculate_match_score(job, user_profile)
                    job["match_score"] = match_score
                
                # Sort by match score (highest first)
                jobs.sort(key=lambda j: j.get("match_score", 0), reverse=True)
            
            # Format response
            # Check if using mock data
            is_mock = any(job.get("source", "").startswith("Mock") for job in jobs)
            
            job_list = []
            for idx, job in enumerate(jobs[:5], 1):  # Show first 5 in text
                match_indicator = ""
                if user_profile and "match_score" in job:
                    score = job["match_score"]
                    if score >= 80:
                        match_indicator = f" 🎯 **{score}% Match - Excellent!**"
                    elif score >= 60:
                        match_indicator = f" ✅ **{score}% Match - Good**"
                    else:
                        match_indicator = f" 💡 {score}% Match"
                
                job_list.append(
                    f"{idx}. **{job['title']}** at {job['company']}{match_indicator}\n"
                    f"   📍 {job['location']} | 💰 {job.get('salary', 'Not specified')}\n"
                    f"   🔗 {job['url']}"
                )
            
            # Add appropriate header based on data source
            user_context = resume_info if user_profile else ""
            
            if is_mock:
                header = f"⚠️ **Found {len(jobs)} sample jobs for: {query}** (Mock data - SerpAPI unavailable){user_context}\n\n"
                footer = "\n\n💡 **Note:** These are mock job listings for demonstration. Real job search requires network access to serpapi.com (currently blocked by firewall/DNS)."
            else:
                header = f"🔍 **Found {len(jobs)} jobs for: {query}** (from Google Jobs){user_context}\n\n"
                footer = ""
            
            if user_profile:
                header += f"✨ Jobs ranked by match to your skills: {', '.join(user_profile.get('skills', [])[:5])}...\n\n"
            
            response_text = header + "\n\n".join(job_list)
            
            if len(jobs) > 5:
                response_text += f"\n\n... and {len(jobs) - 5} more jobs"
            
            response_text += footer
            
            return {
                "response": response_text,
                "data": {
                    "jobs": jobs,
                    "total": len(jobs),
                    "query": query,
                    "location": location or "any",
                    "source": "mock" if is_mock else "serpapi",
                    "matched_with_resume": user_profile is not None,
                    "user_profile": user_profile
                }
            }
            
        except Exception as e:
            logger.error(f"Error searching jobs: {e}")
            return {
                "response": f"❌ Error searching jobs: {str(e)}",
                "data": None
            }
    
    # ═══════════════════════════════════════════════════════════════════════
    # CAPABILITY: rank_jobs
    # ═══════════════════════════════════════════════════════════════════════
    
    async def rank_jobs(
        self,
        user_id: str,
        job_ids: Optional[List[str]] = None,
        min_score: int = 50
    ) -> Dict[str, Any]:
        """
        Rank jobs by match score based on user's resume.
        
        Returns:
            {
                "response": "Ranked jobs...",
                "data": {
                    "ranked_jobs": [...],
                    "total": 5,
                    "average_match": 72
                }
            }
        """
        try:
            # Get user profile
            profile = self.parser.get_profile(user_id)
            
            if not profile:
                return {
                    "response": f"❌ No resume profile found for user: `{user_id}`. Parse resume first.",
                    "data": None
                }
            
            # Get jobs to rank
            all_jobs = self.searcher.get_saved_jobs()
            
            if job_ids:
                # Rank specific jobs
                jobs_to_rank = [j for j in all_jobs if j["job_id"] in job_ids]
            else:
                # Rank all saved jobs
                jobs_to_rank = all_jobs
            
            if not jobs_to_rank:
                return {
                    "response": "ℹ️ No jobs found to rank. Search for jobs first.",
                    "data": {
                        "ranked_jobs": [],
                        "total": 0
                    }
                }
            
            # Calculate match scores
            ranked_jobs = []
            user_skills = set(skill.lower() for skill in profile.get("skills", []))
            user_experience = profile.get("experience", [])
            total_years = sum(exp.get("years", 0) for exp in user_experience)
            
            for job in jobs_to_rank:
                score = self._calculate_match_score(
                    user_skills=user_skills,
                    total_years=total_years,
                    job=job
                )
                
                if score >= min_score:
                    ranked_jobs.append({
                        **job,
                        "match_score": score
                    })
            
            # Sort by match score
            ranked_jobs.sort(key=lambda x: x["match_score"], reverse=True)
            
            # Format response
            if not ranked_jobs:
                return {
                    "response": f"ℹ️ No jobs match your profile with score >= {min_score}%",
                    "data": {
                        "ranked_jobs": [],
                        "total": 0
                    }
                }
            
            job_list = []
            for idx, job in enumerate(ranked_jobs[:5], 1):
                job_list.append(
                    f"{idx}. **{job['title']}** at {job['company']} — {job['match_score']}% match\n"
                    f"   📍 {job['location']} | 💰 {job.get('salary', 'Not specified')}"
                )
            
            avg_score = sum(j["match_score"] for j in ranked_jobs) / len(ranked_jobs)
            
            response_text = (
                f"🎯 **Ranked {len(ranked_jobs)} Jobs by Match Score**\n"
                f"Average Match: {avg_score:.0f}%\n\n"
                + "\n\n".join(job_list)
            )
            
            if len(ranked_jobs) > 5:
                response_text += f"\n\n... and {len(ranked_jobs) - 5} more matches"
            
            return {
                "response": response_text,
                "data": {
                    "ranked_jobs": ranked_jobs,
                    "total": len(ranked_jobs),
                    "average_match": round(avg_score, 1)
                }
            }
            
        except Exception as e:
            logger.error(f"Error ranking jobs: {e}")
            return {
                "response": f"❌ Error ranking jobs: {str(e)}",
                "data": None
            }
    
    def _calculate_match_score(
        self,
        user_skills: set,
        total_years: float,
        job: Dict[str, Any]
    ) -> int:
        """
        Calculate match score (0-100) for a job.
        
        Scoring:
        - Skills overlap: 60%
        - Experience match: 30%
        - Location preference: 10%
        """
        job_skills = set(skill.lower() for skill in job.get("skills_required", []))
        
        # Skill match (60 points)
        if job_skills:
            matching_skills = user_skills & job_skills
            skill_score = (len(matching_skills) / len(job_skills)) * 60
        else:
            # If no skills listed, give partial credit
            skill_score = 30
        
        # Experience match (30 points)
        # Extract years from job description (simple heuristic)
        description = job.get("description", "").lower()
        
        if any(word in description for word in ["junior", "entry", "0-2", "fresher"]):
            required_years = 1
        elif any(word in description for word in ["senior", "lead", "5+", "5-7"]):
            required_years = 5
        else:
            required_years = 3  # Default mid-level
        
        # Score based on how close user experience is to requirement
        year_diff = abs(total_years - required_years)
        if year_diff <= 1:
            exp_score = 30
        elif year_diff <= 2:
            exp_score = 20
        elif year_diff <= 3:
            exp_score = 10
        else:
            exp_score = 5
        
        # Location match (10 points) - simplified
        # In production, check user preferences
        location_score = 10
        
        total_score = int(skill_score + exp_score + location_score)
        return min(total_score, 100)
    
    # ═══════════════════════════════════════════════════════════════════════
    # CAPABILITY: get_job_details
    # ═══════════════════════════════════════════════════════════════════════
    
    async def get_job_details(self, job_id: str) -> Dict[str, Any]:
        """
        Get full details for a specific job.
        
        Returns:
            {
                "response": "Job details...",
                "data": {job object}
            }
        """
        try:
            all_jobs = self.searcher.get_saved_jobs()
            job = next((j for j in all_jobs if j["job_id"] == job_id), None)
            
            if not job:
                return {
                    "response": f"❌ Job not found: `{job_id}`",
                    "data": None
                }
            
            response_text = (
                f"📋 **{job['title']}**\n"
                f"🏢 Company: {job['company']}\n"
                f"📍 Location: {job['location']}\n"
                f"💰 Salary: {job.get('salary', 'Not specified')}\n"
                f"📅 Posted: {job['posted_date']}\n"
                f"🔧 Job Type: {job['job_type']}\n\n"
                f"**Required Skills:** {', '.join(job.get('skills_required', ['Not specified']))}\n\n"
                f"**Description:**\n{job['description'][:500]}...\n\n"
                f"🔗 Apply: {job['url']}"
            )
            
            return {
                "response": response_text,
                "data": job
            }
            
        except Exception as e:
            logger.error(f"Error getting job details: {e}")
            return {
                "response": f"❌ Error retrieving job: {str(e)}",
                "data": None
            }
    
    # ═══════════════════════════════════════════════════════════════════════
    # CAPABILITY: track_application
    # ═══════════════════════════════════════════════════════════════════════
    
    async def track_application(
        self,
        user_id: str,
        job_id: str,
        status: str = "applied"
    ) -> Dict[str, Any]:
        """
        Track job application.
        
        Returns:
            {
                "response": "Application logged...",
                "data": {application object}
            }
        """
        try:
            # Load existing applications
            if self.applications_file.exists():
                with open(self.applications_file, 'r', encoding='utf-8') as f:
                    applications = json.load(f)
            else:
                applications = []
            
            # Get job details
            all_jobs = self.searcher.get_saved_jobs()
            job = next((j for j in all_jobs if j["job_id"] == job_id), None)
            
            # Create application record
            application = {
                "application_id": f"app_{len(applications) + 1:03d}",
                "user_id": user_id,
                "job_id": job_id,
                "job_title": job["title"] if job else "Unknown",
                "company": job["company"] if job else "Unknown",
                "applied_at": datetime.now().isoformat(),
                "status": status
            }
            
            applications.append(application)
            
            # Save
            with open(self.applications_file, 'w', encoding='utf-8') as f:
                json.dump(applications, f, indent=2, ensure_ascii=False)
            
            response_text = (
                f"✅ **Application Logged**\n\n"
                f"Job: **{application['job_title']}** at {application['company']}\n"
                f"Status: {status}\n"
                f"Applied: {application['applied_at']}\n\n"
                f"Total applications: {len(applications)}"
            )
            
            return {
                "response": response_text,
                "data": application
            }
            
        except Exception as e:
            logger.error(f"Error tracking application: {e}")
            return {
                "response": f"❌ Error tracking application: {str(e)}",
                "data": None
            }
