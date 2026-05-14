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
import asyncio
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
            description=(
                "Multi-source job search using Playwright. Auto-parses user's resume, "
                "scrapes LinkedIn + Naukri + Indeed in parallel, deduplicates, "
                "ranks by match score, and excludes already-applied jobs."
            ),
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
                ),
                ToolParameter(
                    name="sources",
                    type=ParameterType.ARRAY,
                    description=(
                        "Optional list of sources to search. Choices: 'linkedin', 'naukri', "
                        "'indeed'. Default = all."
                    ),
                    required=False,
                ),
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

        # 5. Auto Apply (NEW - Playwright powered)
        register(
            capability="job.auto_apply",
            agent_name=self.name,
            handler=self.auto_apply,
            description=(
                "Auto-apply to one or more jobs via browser automation. "
                "Supports Naukri, LinkedIn Easy Apply, and generic form filling. "
                "Modes: 'auto' (submit), 'semi-auto' (fill + pause), 'review' (fill only), 'dry-run' (simulate)."
            ),
            parameters=[
                ToolParameter(
                    name="job_ids",
                    type=ParameterType.ARRAY,
                    description="List of job IDs to apply to (from a prior search_jobs call)",
                    required=True,
                ),
                ToolParameter(
                    name="mode",
                    type=ParameterType.STRING,
                    description="Apply mode: 'auto', 'semi-auto' (default), 'review', or 'dry-run'",
                    required=False,
                ),
                ToolParameter(
                    name="min_match_score",
                    type=ParameterType.INTEGER,
                    description="Skip jobs below this match score (default: 0)",
                    required=False,
                ),
            ],
        )

        # 6. List Applications (NEW)
        register(
            capability="job.list_applications",
            agent_name=self.name,
            handler=self.list_applications,
            description="List the user's job applications with their statuses",
            parameters=[
                ToolParameter(
                    name="status",
                    type=ParameterType.STRING,
                    description="Filter by status (e.g. 'submitted', 'pending', 'failed')",
                    required=False,
                ),
                ToolParameter(
                    name="limit",
                    type=ParameterType.INTEGER,
                    description="Max applications to return (default: 20)",
                    required=False,
                ),
            ],
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # CAPABILITY: search_jobs
    # ═══════════════════════════════════════════════════════════════════════
    
    async def search_jobs(
        self,
        query: str,
        location: str = "",
        num_results: int = 10,
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        SMART multi-source job search powered by Playwright.

        Workflow:
          1. Auto-detect + parse user's resume from data/resumes/
          2. Scrape LinkedIn, Naukri, Indeed in parallel via Playwright
          3. Deduplicate across sources by (company, title)
          4. Filter out jobs already applied to
          5. Rank by skills/experience/location match score
          6. Fall back to SerpAPI / mock if all scrapers fail

        Returns:
            {
              "response": "human-readable summary",
              "data": {
                "jobs": [...],
                "total": int,
                "query": str,
                "location": str,
                "matched_with_resume": bool,
                "sources_searched": [...],
                "sources_failed": [...],
                "duration_ms": int
              }
            }
        """
        from core.job_models import JobSearchFilter, JobSource
        from tools.job_orchestrator import get_orchestrator

        try:
            # ── STEP 1: Auto-detect + parse resume ─────────────────────────
            user_profile = await self._load_user_profile()
            resume_info = ""
            inferred_from_resume = False
            if user_profile:
                name = user_profile.get("personal", {}).get("name", "Unknown")
                skills = user_profile.get("skills", []) or []
                resume_info = f"\n📄 Resume: **{name}** ({len(skills)} skills)"

            # ── STEP 1b: Infer query from resume if missing/vague ──────────
            # The user can say "find jobs related to my resume" without a role.
            # Derive query from the most recent job title or top skills.
            VAGUE_QUERIES = {
                "", "jobs", "job", "my resume", "resume", "for me",
                "related to my resume", "based on my resume", "matching jobs",
                "relevant jobs", "any", "anything", "find jobs", "find job",
            }
            q_clean = (query or "").strip().lower()
            if (not q_clean or q_clean in VAGUE_QUERIES) and user_profile:
                inferred = self._infer_query_from_resume(user_profile)
                if inferred:
                    logger.info("Inferred query from resume: %r", inferred)
                    query = inferred
                    inferred_from_resume = True
                    resume_info += f"\n🎯 Inferred role from resume: **{query}**"

            if not (query or "").strip():
                return {
                    "response": (
                        "❌ I couldn't find a resume to infer a role from, and no "
                        "role was provided. Please drop your resume in "
                        "`data/resumes/` or specify a role (e.g., 'Python Developer')."
                    ),
                    "data": None,
                }

            # ── STEP 2: Build filter ───────────────────────────────────────
            source_enums: List[JobSource] = []
            if sources:
                for s in sources:
                    try:
                        source_enums.append(JobSource(s.lower()))
                    except ValueError:
                        logger.warning("Unknown source: %s", s)

            filters = JobSearchFilter(
                query=query,
                location=location or "",
                num_results=num_results,
                sources=source_enums,
            )

            # ── STEP 3: Multi-source scrape ────────────────────────────────
            orch = get_orchestrator()
            result = await orch.search(filters, user_profile=user_profile, exclude_applied=True)

            # ── STEP 4: Fallback if zero results ───────────────────────────
            if result.total == 0:
                logger.info("No Playwright results — falling back to legacy searcher")
                legacy_jobs = await self.searcher.search_jobs(query, location, num_results)
                return self._format_legacy_response(legacy_jobs, query, location, user_profile, resume_info)

            # ── STEP 5: Format response ────────────────────────────────────
            return self._format_search_result(result, user_profile, resume_info)

        except Exception as e:
            logger.exception("search_jobs failed")
            return {
                "response": f"❌ Job search error: {str(e)}",
                "data": None,
            }

    # ── Helpers for search_jobs ────────────────────────────────────────────
    async def _load_user_profile(self) -> Optional[Dict[str, Any]]:
        """
        Load user profile for ranking.

        Priority:
          1. Saved profile in data/user_profiles.json (default_user)
          2. Auto-parse latest resume from data/resumes/ (sync API, run in thread)
        """
        try:
            # 1. Try saved profile first (fastest, no LLM calls)
            saved = self.parser.get_profile("default_user")
            if saved:
                logger.info("Loaded saved profile for default_user")
                return saved

            # 2. Auto-detect resume file and parse
            resumes_dir = Path("data/resumes")
            if not resumes_dir.exists():
                return None
            files = (
                list(resumes_dir.glob("*.pdf"))
                + list(resumes_dir.glob("*.docx"))
                + list(resumes_dir.glob("*.txt"))
            )
            if not files:
                return None
            latest = max(files, key=lambda p: p.stat().st_mtime)
            logger.info("Auto-detected resume: %s", latest.name)

            def _parse_sync() -> Dict[str, Any]:
                text = self.parser.extract_text(str(latest))
                llm = getattr(self, "model_client", None)
                data = self.parser.extract_structured_data(text, llm_client=llm)
                self.parser.save_profile("default_user", data)
                return data

            return await asyncio.to_thread(_parse_sync)
        except Exception as e:
            logger.warning("Resume auto-parse failed: %s", e)
            return None

    def _infer_query_from_resume(self, profile: Dict[str, Any]) -> str:
        """
        Derive a job-search query string from the user's resume.

        Priority:
          1. Most recent experience title (e.g., "Senior Python Developer")
          2. Preferred role from saved preferences
          3. Top 2 technical skills (e.g., "Python Machine Learning")
        """
        try:
            # 1. Most recent role from experience list
            experience = profile.get("experience") or []
            if isinstance(experience, list) and experience:
                # Heuristic: first item is typically the most recent
                latest = experience[0] if isinstance(experience[0], dict) else None
                if latest:
                    title = (
                        latest.get("role")
                        or latest.get("title")
                        or latest.get("position")
                        or ""
                    ).strip()
                    if title and len(title) <= 80:
                        return title

            # 2. Preferences
            prefs = profile.get("preferences") or {}
            roles = prefs.get("roles") if isinstance(prefs, dict) else None
            if isinstance(roles, list) and roles:
                first_role = str(roles[0]).strip()
                if first_role:
                    return first_role

            # 3. Top skills
            skills = profile.get("skills") or []
            if isinstance(skills, list) and skills:
                top = [str(s).strip() for s in skills[:2] if str(s).strip()]
                if top:
                    return " ".join(top) + " Developer"

            # 4. Summary first line
            summary = (profile.get("summary") or "").strip()
            if summary:
                first_line = summary.split("\n")[0].split(".")[0].strip()
                if 3 <= len(first_line) <= 80:
                    return first_line
        except Exception as e:
            logger.warning("Query inference from resume failed: %s", e)
        return ""

    def _format_search_result(self, result, user_profile, resume_info: str) -> Dict[str, Any]:
        """Format JobSearchResult into the agent's response/data dict."""
        jobs = result.jobs
        lines: List[str] = []
        for idx, j in enumerate(jobs[:10], 1):
            badge = ""
            if j.match_score is not None:
                if j.match_score >= 80:
                    badge = f" 🎯 **{j.match_score}% Match — Excellent**"
                elif j.match_score >= 60:
                    badge = f" ✅ **{j.match_score}% Match — Good**"
                else:
                    badge = f" 💡 {j.match_score}% Match"
            sal = f" | 💰 {j.salary}" if j.salary else ""
            lines.append(
                f"{idx}. **{j.title}** at **{j.company}** [{j.source.value}]{badge}\n"
                f"   📍 {j.location}{sal}\n"
                f"   🔗 {j.url}"
            )

        sources_str = ", ".join(s.value for s in result.sources_searched) or "none"
        header = (
            f"🔍 **Found {result.total} jobs for '{result.query}'**"
            f"{resume_info}\n"
            f"📡 Sources: {sources_str} | ⏱ {result.duration_ms}ms\n"
        )
        if result.sources_failed:
            failed = ", ".join(f["source"] for f in result.sources_failed)
            header += f"⚠️ Failed: {failed}\n"
        if user_profile:
            top_skills = ", ".join((user_profile.get("skills") or [])[:5])
            header += f"✨ Ranked by your skills: {top_skills}…\n"
        header += "\n"

        response_text = header + "\n\n".join(lines)
        if result.total > 10:
            response_text += f"\n\n…and {result.total - 10} more"

        return {
            "response": response_text,
            "data": {
                "jobs": [j.model_dump(mode="json") for j in jobs],
                "total": result.total,
                "query": result.query,
                "location": result.location,
                "matched_with_resume": result.matched_with_resume,
                "sources_searched": [s.value for s in result.sources_searched],
                "sources_failed": result.sources_failed,
                "duration_ms": result.duration_ms,
            },
        }

    def _format_legacy_response(self, jobs, query, location, user_profile, resume_info):
        """Fallback formatting for legacy searcher (SerpAPI / mock)."""
        if not jobs:
            return {
                "response": f"ℹ️ No jobs found for **{query}** in {location or 'any location'}{resume_info}",
                "data": {"jobs": [], "total": 0, "query": query, "location": location},
            }

        # Calculate match scores if profile available
        if user_profile:
            for job in jobs:
                job["match_score"] = self._calculate_match_score(job, user_profile)
            jobs.sort(key=lambda j: j.get("match_score", 0), reverse=True)

        is_mock = any(j.get("source", "").startswith("Mock") for j in jobs)
        lines = []
        for idx, j in enumerate(jobs[:5], 1):
            badge = ""
            if "match_score" in j:
                score = j["match_score"]
                if score >= 80:
                    badge = f" 🎯 {score}% Match"
                elif score >= 60:
                    badge = f" ✅ {score}% Match"
            lines.append(
                f"{idx}. **{j['title']}** at {j['company']}{badge}\n"
                f"   📍 {j['location']} | 💰 {j.get('salary', 'N/A')}\n"
                f"   🔗 {j['url']}"
            )

        prefix = "⚠️ Mock data (scrapers + SerpAPI both unavailable)" if is_mock else "🔍 Found jobs"
        response_text = f"{prefix} for **{query}**{resume_info}\n\n" + "\n\n".join(lines)
        return {
            "response": response_text,
            "data": {
                "jobs": jobs,
                "total": len(jobs),
                "query": query,
                "location": location or "any",
                "source": "mock" if is_mock else "legacy",
                "matched_with_resume": user_profile is not None,
            },
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

    # ═══════════════════════════════════════════════════════════════════════
    # CAPABILITY: auto_apply (NEW - Playwright powered)
    # ═══════════════════════════════════════════════════════════════════════

    async def auto_apply(
        self,
        job_ids: List[str],
        mode: str = "semi-auto",
        min_match_score: int = 0,
    ) -> Dict[str, Any]:
        """
        Auto-apply to jobs using Playwright browser automation.

        Args:
            job_ids: Job IDs from a previous search_jobs call
            mode: 'auto' | 'semi-auto' (default) | 'review' | 'dry-run'
            min_match_score: Skip jobs below this score (0–100)

        Returns:
            { "response": str, "data": { "results": [...], "summary": {...} } }
        """
        from core.job_models import ApplyMode, JobListing
        from tools.auto_applicator import AutoApplicator
        from tools.application_tracker import get_tracker

        try:
            # Validate mode
            try:
                apply_mode = ApplyMode(mode.lower())
            except ValueError:
                return {
                    "response": f"❌ Invalid mode '{mode}'. Use: auto, semi-auto, review, dry-run.",
                    "data": None,
                }

            # Load user profile
            user_profile = await self._load_user_profile()
            if not user_profile:
                return {
                    "response": "❌ No resume found in `data/resumes/`. Add your resume first.",
                    "data": None,
                }

            # Reconstruct JobListings from saved job_listings.json
            tracker = get_tracker()
            saved_jobs = self.searcher.get_saved_jobs() or []
            jobs_by_id = {j.get("job_id"): j for j in saved_jobs}

            selected: List[JobListing] = []
            missing: List[str] = []
            for jid in job_ids:
                raw = jobs_by_id.get(jid)
                if not raw:
                    missing.append(jid)
                    continue
                try:
                    # Best-effort conversion of legacy dicts into JobListing
                    selected.append(JobListing.model_validate(raw))
                except Exception:
                    # Legacy schema — try minimal mapping
                    try:
                        from core.job_models import JobSource
                        src = raw.get("source", "mock")
                        src_enum = JobSource(src) if src in JobSource._value2member_map_ else JobSource.MOCK
                        selected.append(
                            JobListing(
                                job_id=raw["job_id"],
                                source=src_enum,
                                title=raw.get("title", "Unknown"),
                                company=raw.get("company", "Unknown"),
                                location=raw.get("location", ""),
                                url=raw.get("url", ""),
                                description=raw.get("description"),
                                salary=raw.get("salary"),
                                skills_required=raw.get("skills_required", []),
                                match_score=raw.get("match_score"),
                            )
                        )
                    except Exception as e:
                        logger.warning("Skip job %s: %s", jid, e)
                        missing.append(jid)

            if not selected:
                return {
                    "response": (
                        f"❌ None of the {len(job_ids)} job IDs were found in saved listings. "
                        f"Run `search_jobs` first. Missing: {missing[:5]}"
                    ),
                    "data": {"missing": missing},
                }

            # Filter by match score
            if min_match_score > 0:
                before = len(selected)
                selected = [j for j in selected if (j.match_score or 0) >= min_match_score]
                logger.info(
                    "Filtered to %d/%d jobs above %d%% match",
                    len(selected), before, min_match_score,
                )

            if not selected:
                return {
                    "response": f"ℹ️ No jobs above {min_match_score}% match score.",
                    "data": None,
                }

            # Execute
            applicator = AutoApplicator()
            results = await applicator.apply_batch(
                selected,
                user_profile=user_profile,
                mode=apply_mode,
                user_id="default_user",
            )

            # Summarize
            summary = {
                "total": len(results),
                "submitted": sum(1 for r in results if r.submitted),
                "ready_to_submit": sum(1 for r in results if r.success and not r.submitted),
                "requires_manual": sum(1 for r in results if r.requires_manual_review),
                "failed": sum(1 for r in results if not r.success),
            }

            # Build response
            lines = [
                f"🤖 **Auto-Apply Complete** (mode: `{apply_mode.value}`)",
                f"📊 Total: {summary['total']} | "
                f"✅ Submitted: {summary['submitted']} | "
                f"⏸ Ready: {summary['ready_to_submit']} | "
                f"⚠️ Manual: {summary['requires_manual']} | "
                f"❌ Failed: {summary['failed']}",
                "",
            ]
            for r in results:
                icon = "✅" if r.submitted else ("⏸" if r.success else "❌")
                lines.append(
                    f"{icon} **{r.application.title}** @ **{r.application.company}** "
                    f"[{r.application.source.value}]\n   {r.reason or ''}"
                )

            return {
                "response": "\n".join(lines),
                "data": {
                    "summary": summary,
                    "results": [
                        {
                            "application_id": r.application.application_id,
                            "job_id": r.application.job_id,
                            "company": r.application.company,
                            "title": r.application.title,
                            "status": r.application.status.value,
                            "submitted": r.submitted,
                            "requires_manual": r.requires_manual_review,
                            "reason": r.reason,
                            "screenshots": r.screenshots,
                            "duration_ms": r.duration_ms,
                        }
                        for r in results
                    ],
                    "missing_job_ids": missing,
                },
            }

        except Exception as e:
            logger.exception("auto_apply failed")
            return {"response": f"❌ Auto-apply error: {e}", "data": None}

    # ═══════════════════════════════════════════════════════════════════════
    # CAPABILITY: list_applications (NEW)
    # ═══════════════════════════════════════════════════════════════════════

    async def list_applications(
        self,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """List the user's job applications with filters."""
        from core.job_models import ApplicationStatus
        from tools.application_tracker import get_tracker

        try:
            status_enum = None
            if status:
                try:
                    status_enum = ApplicationStatus(status.lower())
                except ValueError:
                    return {
                        "response": (
                            f"❌ Invalid status '{status}'. Choices: "
                            f"{[s.value for s in ApplicationStatus]}"
                        ),
                        "data": None,
                    }

            tracker = get_tracker()
            apps = tracker.list_applications(user_id="default_user", status=status_enum, limit=limit)
            stats = tracker.stats(user_id="default_user")

            if not apps:
                return {
                    "response": f"ℹ️ No applications found{f' with status `{status}`' if status else ''}.",
                    "data": {"applications": [], "stats": stats},
                }

            lines = [f"📋 **Your Applications** ({len(apps)} of {stats.get('total', 0)} total)\n"]
            for a in apps:
                ts = a.applied_at.strftime("%Y-%m-%d") if a.applied_at else "—"
                lines.append(
                    f"• **{a.title}** @ **{a.company}** [{a.source.value}]\n"
                    f"  Status: `{a.status.value}` | Applied: {ts} | Score: {a.match_score_at_apply or 'N/A'}\n"
                    f"  🔗 {a.job_url}"
                )

            return {
                "response": "\n".join(lines),
                "data": {
                    "applications": [a.model_dump(mode="json") for a in apps],
                    "stats": stats,
                },
            }
        except Exception as e:
            logger.exception("list_applications failed")
            return {"response": f"❌ Error: {e}", "data": None}
