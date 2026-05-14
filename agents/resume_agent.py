# agents/resume_agent.py
"""
Resume Agent - Parse and analyze resumes for job matching

Capabilities:
- parse_resume: Upload and parse resume (PDF/DOCX/TXT)
- get_profile: Retrieve user's resume profile
- analyze_fit: Match resume against job description
- update_preferences: Update job search preferences
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
from autogen_agentchat.agents import AssistantAgent
from tools.resume_parser import ResumeParser
from core.capabilities import register
from core.schemas import ToolParameter, ParameterType

logger = logging.getLogger("jarvis.resume_agent")

class ResumeAgent(AssistantAgent):
    """
    Resume Agent for parsing and analyzing resumes.
    Integrates with Job Agent for intelligent job matching.
    """
    
    def __init__(self, name="resume", model_client=None):
        super().__init__(
            name=name,
            model_client=model_client,
            system_message=(
                "You are the Resume Agent. You help users parse their resumes, "
                "extract structured information, and analyze job fit.\n\n"
                "⚠️ CRITICAL RULES:\n"
                "1. NEVER EVER hallucinate or make up resume data\n"
                "2. ALWAYS call tools to get actual data from files\n"
                "3. If you don't have data, call auto_detect_resume or get_profile first\n"
                "4. Do NOT invent skills, experience, or qualifications\n\n"
                "SMART DEFAULTS:\n"
                "- If user says 'my resume' but no file path given, call auto_detect_resume first\n"
                "- If no user_id given, use 'default_user'\n"
                "- After parsing, you can analyze fit or suggest job searches\n\n"
                "WORKFLOW:\n"
                "When user asks to 'review resume':\n"
                "Step 1: Call auto_detect_resume (to find and parse resume)\n"
                "Step 2: Present the ACTUAL extracted data (don't make it up!)\n"
                "Step 3: If they want jobs, tell them to ask for specific job searches"
            )
        )
        self.parser = ResumeParser()
        self.model_client = model_client
        
        # ──────────────────────────────────────────────────────────────────────
        # Register 4 Resume Capabilities
        # ──────────────────────────────────────────────────────────────────────
        
        # 1. Parse Resume
        register(
            capability="resume.parse",
            agent_name=self.name,
            handler=self.parse_resume,
            description="Parse resume file (PDF/DOCX/TXT) and extract structured data",
            parameters=[
                ToolParameter(
                    name="user_id",
                    type=ParameterType.STRING,
                    description="Unique user identifier",
                    required=True
                ),
                ToolParameter(
                    name="file_path",
                    type=ParameterType.STRING,
                    description="Path to resume file (PDF, DOCX, or TXT)",
                    required=True
                ),
                ToolParameter(
                    name="use_llm",
                    type=ParameterType.BOOLEAN,
                    description="Use LLM for extraction (True) or mock mode (False)",
                    required=False
                )
            ]
        )
        
        # 2. Get Profile
        register(
            capability="resume.get_profile",
            agent_name=self.name,
            handler=self.get_profile,
            description="Retrieve user's parsed resume profile from storage",
            parameters=[
                ToolParameter(
                    name="user_id",
                    type=ParameterType.STRING,
                    description="Unique user identifier",
                    required=True
                )
            ]
        )
        
        # 3. Analyze Fit
        register(
            capability="resume.analyze_fit",
            agent_name=self.name,
            handler=self.analyze_fit,
            description="Analyze how well user's resume matches a job description",
            parameters=[
                ToolParameter(
                    name="user_id",
                    type=ParameterType.STRING,
                    description="User identifier",
                    required=True
                ),
                ToolParameter(
                    name="job_description",
                    type=ParameterType.STRING,
                    description="Job description text to match against",
                    required=True
                )
            ]
        )
        
        # 4. Update Preferences
        register(
            capability="resume.update_preferences",
            agent_name=self.name,
            handler=self.update_preferences,
            description="Update user's job search preferences (locations, roles, salary, etc.)",
            parameters=[
                ToolParameter(
                    name="user_id",
                    type=ParameterType.STRING,
                    description="User identifier",
                    required=True
                ),
                ToolParameter(
                    name="preferences",
                    type=ParameterType.OBJECT,
                    description="Preferences dict with locations, roles, min_salary, job_types",
                    required=True
                )
            ]
        )
        
        # 5. Auto-detect Resume (NEW - Smart discovery)
        register(
            capability="resume.auto_detect",
            agent_name=self.name,
            handler=self.auto_detect_resume,
            description="Automatically find and parse the most recent resume in data/resumes folder",
            parameters=[
                ToolParameter(
                    name="user_id",
                    type=ParameterType.STRING,
                    description="User identifier (default: 'default_user')",
                    required=False
                )
            ]
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # CAPABILITY: auto_detect_resume (NEW - Smart Discovery)
    # ═══════════════════════════════════════════════════════════════════════
    
    async def auto_detect_resume(
        self,
        user_id: str = "default_user"
    ) -> Dict[str, Any]:
        """
        Automatically find and parse the most recent resume in data/resumes folder.
        
        Returns:
            {
                "response": "Auto-detected and parsed resume...",
                "data": {parsed profile data}
            }
        """
        try:
            resumes_dir = Path("data/resumes")
            
            if not resumes_dir.exists():
                return {
                    "response": "❌ No resumes folder found. Create `data/resumes/` and add your resume.",
                    "data": None
                }
            
            # Find all resume files
            resume_files = []
            for ext in ['.pdf', '.docx', '.txt']:
                resume_files.extend(resumes_dir.glob(f'*{ext}'))
            
            if not resume_files:
                return {
                    "response": "ℹ️ No resume files found in `data/resumes/`. Please add your resume (PDF, DOCX, or TXT).",
                    "data": None
                }
            
            # Use most recently modified file
            latest_resume = max(resume_files, key=lambda f: f.stat().st_mtime)
            
            logger.info(f"Auto-detected resume: {latest_resume.name}")
            
            # Parse it
            result = await self.parse_resume(
                user_id=user_id,
                file_path=str(latest_resume),
                use_llm=True
            )
            
            # Add auto-detection info to response
            if result.get("data"):
                result["response"] = (
                    f"🔍 **Auto-detected Resume:** `{latest_resume.name}`\n\n" +
                    result["response"]
                )
                result["data"]["auto_detected_file"] = latest_resume.name
            
            return result
            
        except Exception as e:
            logger.error(f"Error auto-detecting resume: {e}")
            return {
                "response": f"❌ Error finding resume: {str(e)}",
                "data": None
            }
    
    # ═══════════════════════════════════════════════════════════════════════
    # CAPABILITY: parse_resume
    # ═══════════════════════════════════════════════════════════════════════
    
    async def parse_resume(
        self,
        user_id: str,
        file_path: str,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Parse resume and extract structured data.
        
        Args:
            user_id: Unique identifier for user
            file_path: Path to resume file (PDF/DOCX/TXT)
            use_llm: Whether to use LLM for extraction (True) or mock mode (False)
        
        Returns:
            {
                "response": "Resume parsed successfully...",
                "data": {
                    "user_id": "john_doe",
                    "personal": {...},
                    "skills": [...],
                    "experience": [...],
                    "education": [...],
                    "profile_saved": true
                }
            }
        """
        try:
            # Extract text from resume
            resume_text = self.parser.extract_text(file_path)
            
            if not resume_text:
                return {
                    "response": "❌ Failed to extract text from resume. Please check the file format.",
                    "data": None
                }
            
            # Extract structured data
            llm_client = self.model_client if use_llm else None
            profile_data = self.parser.extract_structured_data(resume_text, llm_client)
            
            # Save profile
            self.parser.save_profile(user_id, profile_data)
            
            # Format response
            name = profile_data.get("personal", {}).get("name", "User")
            skills = profile_data.get("skills", [])
            experience = profile_data.get("experience", [])
            
            total_years = sum(exp.get("years", 0) for exp in experience)
            
            response_text = (
                f"✅ **Resume Parsed Successfully**\n\n"
                f"**Name:** {name}\n"
                f"**Skills:** {', '.join(skills[:8])}{'...' if len(skills) > 8 else ''}\n"
                f"**Experience:** {total_years:.1f} years across {len(experience)} role(s)\n"
                f"**Education:** {len(profile_data.get('education', []))} degree(s)\n\n"
                f"Profile saved for user: `{user_id}`"
            )
            
            return {
                "response": response_text,
                "data": {
                    "user_id": user_id,
                    "personal": profile_data.get("personal"),
                    "summary": profile_data.get("summary"),
                    "skills": skills,
                    "total_skills": len(skills),
                    "experience": experience,
                    "total_years": round(total_years, 1),
                    "education": profile_data.get("education"),
                    "projects": profile_data.get("projects", []),
                    "certifications": profile_data.get("certifications", []),
                    "profile_saved": True
                }
            }
            
        except FileNotFoundError:
            return {
                "response": f"❌ Resume file not found: `{file_path}`",
                "data": None
            }
        except Exception as e:
            logger.error(f"Error parsing resume: {e}")
            return {
                "response": f"❌ Error parsing resume: {str(e)}",
                "data": None
            }
    
    # ═══════════════════════════════════════════════════════════════════════
    # CAPABILITY: get_profile
    # ═══════════════════════════════════════════════════════════════════════
    
    async def get_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieve user's resume profile from storage.
        
        Args:
            user_id: Unique identifier for user
        
        Returns:
            {
                "response": "Profile for john_doe...",
                "data": {profile data}
            }
        """
        try:
            profile = self.parser.get_profile(user_id)
            
            if not profile:
                return {
                    "response": f"ℹ️ No profile found for user: `{user_id}`. Upload a resume first.",
                    "data": None
                }
            
            name = profile.get("personal", {}).get("name", "User")
            skills = profile.get("skills", [])
            experience = profile.get("experience", [])
            
            response_text = (
                f"📄 **Profile for {user_id}**\n\n"
                f"**Name:** {name}\n"
                f"**Skills:** {', '.join(skills[:10])}\n"
                f"**Experience:** {len(experience)} role(s)\n"
                f"**Last Updated:** {profile.get('updated_at', 'Unknown')}"
            )
            
            return {
                "response": response_text,
                "data": profile
            }
            
        except Exception as e:
            logger.error(f"Error retrieving profile: {e}")
            return {
                "response": f"❌ Error retrieving profile: {str(e)}",
                "data": None
            }
    
    # ═══════════════════════════════════════════════════════════════════════
    # CAPABILITY: analyze_fit
    # ═══════════════════════════════════════════════════════════════════════
    
    async def analyze_fit(
        self,
        user_id: str,
        job_description: str
    ) -> Dict[str, Any]:
        """
        Analyze how well user's resume matches a job description.
        
        Args:
            user_id: User identifier
            job_description: Job description text to match against
        
        Returns:
            {
                "response": "Match analysis...",
                "data": {
                    "match_score": 85,
                    "matching_skills": [...],
                    "missing_skills": [...],
                    "recommendations": [...]
                }
            }
        """
        try:
            # Get user profile
            profile = self.parser.get_profile(user_id)
            
            if not profile:
                return {
                    "response": f"❌ No profile found for user: `{user_id}`",
                    "data": None
                }
            
            user_skills = set(skill.lower() for skill in profile.get("skills", []))
            user_experience = profile.get("experience", [])
            total_years = sum(exp.get("years", 0) for exp in user_experience)
            
            # Extract skills from job description (simple keyword matching)
            # In production, use LLM for better extraction
            jd_lower = job_description.lower()
            
            # Common tech skills to check
            all_skills = [
                'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'Go', 'Rust',
                'React', 'Angular', 'Vue', 'Node.js', 'Django', 'Flask', 'FastAPI',
                'SQL', 'PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'Elasticsearch',
                'Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP', 'Terraform',
                'Git', 'CI/CD', 'Jenkins', 'GitLab', 'GitHub Actions',
                'Machine Learning', 'Deep Learning', 'NLP', 'Computer Vision',
                'TensorFlow', 'PyTorch', 'Scikit-learn', 'LLM', 'AutoGen',
                'REST API', 'GraphQL', 'Microservices', 'Agile', 'Scrum'
            ]
            
            required_skills = set(
                skill for skill in all_skills 
                if skill.lower() in jd_lower
            )
            
            # Calculate matching
            matching_skills = [
                skill for skill in required_skills 
                if skill.lower() in user_skills
            ]
            missing_skills = list(required_skills - set(skill.lower() for skill in matching_skills))
            
            # Calculate match score
            if required_skills:
                skill_match = (len(matching_skills) / len(required_skills)) * 60
            else:
                skill_match = 50
            
            # Experience match (simple heuristic)
            experience_match = min(total_years * 10, 30)
            
            match_score = int(skill_match + experience_match)
            
            # Recommendations
            recommendations = []
            if missing_skills:
                recommendations.append(f"Add these skills: {', '.join(missing_skills[:5])}")
            if total_years < 2:
                recommendations.append("Highlight projects and internships to compensate for experience")
            if match_score < 50:
                recommendations.append("This role may be challenging - consider upskilling first")
            elif match_score >= 80:
                recommendations.append("Excellent match! You should definitely apply")
            
            response_text = (
                f"🎯 **Job Match Analysis**\n\n"
                f"**Match Score:** {match_score}%\n"
                f"**Matching Skills:** {', '.join(matching_skills[:8]) if matching_skills else 'None'}\n"
                f"**Missing Skills:** {', '.join(missing_skills[:5]) if missing_skills else 'None'}\n\n"
                f"**Recommendations:**\n" + "\n".join(f"• {rec}" for rec in recommendations)
            )
            
            return {
                "response": response_text,
                "data": {
                    "match_score": match_score,
                    "matching_skills": matching_skills,
                    "missing_skills": missing_skills,
                    "total_experience_years": round(total_years, 1),
                    "recommendations": recommendations
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing fit: {e}")
            return {
                "response": f"❌ Error analyzing job fit: {str(e)}",
                "data": None
            }
    
    # ═══════════════════════════════════════════════════════════════════════
    # CAPABILITY: update_preferences
    # ═══════════════════════════════════════════════════════════════════════
    
    async def update_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update user's job search preferences.
        
        Args:
            user_id: User identifier
            preferences: Dict with keys like:
                - locations: ["Delhi", "Bangalore", "Remote"]
                - roles: ["Backend Developer", "ML Engineer"]
                - min_salary: 1000000 (in INR)
                - experience_level: "Mid-level"
                - job_types: ["Full-time", "Contract"]
        
        Returns:
            {
                "response": "Preferences updated...",
                "data": {updated preferences}
            }
        """
        try:
            # Get existing profile
            profile = self.parser.get_profile(user_id)
            
            if not profile:
                return {
                    "response": f"❌ No profile found for user: `{user_id}`",
                    "data": None
                }
            
            # Update preferences
            profile["preferences"] = preferences
            
            # Save profile
            self.parser.save_profile(user_id, profile)
            
            response_text = (
                f"✅ **Preferences Updated**\n\n"
                f"**Locations:** {', '.join(preferences.get('locations', ['Not set']))}\n"
                f"**Roles:** {', '.join(preferences.get('roles', ['Not set']))}\n"
                f"**Min Salary:** {preferences.get('min_salary', 'Not set')}\n"
                f"**Job Types:** {', '.join(preferences.get('job_types', ['Not set']))}"
            )
            
            return {
                "response": response_text,
                "data": preferences
            }
            
        except Exception as e:
            logger.error(f"Error updating preferences: {e}")
            return {
                "response": f"❌ Error updating preferences: {str(e)}",
                "data": None
            }
