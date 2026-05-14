# tools/resume_parser.py
"""
Resume Parser Tool

Extracts structured data from PDF/DOCX resumes using:
- PyPDF2 for PDF parsing
- python-docx for DOCX parsing
- LLM for intelligent extraction
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger("jarvis.resume_parser")

class ResumeParser:
    """
    Parse resumes and extract structured information.
    Supports PDF and DOCX formats.
    """
    
    def __init__(self, resumes_dir: str = "data/resumes"):
        self.resumes_dir = Path(resumes_dir)
        self.resumes_dir.mkdir(parents=True, exist_ok=True)
        
    def parse_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            import PyPDF2
            
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except ImportError:
            logger.warning("PyPDF2 not installed. Install with: pip install PyPDF2")
            return ""
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            return ""
    
    def parse_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            from docx import Document
            
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()
        except ImportError:
            logger.warning("python-docx not installed. Install with: pip install python-docx")
            return ""
        except Exception as e:
            logger.error(f"Error parsing DOCX: {e}")
            return ""
    
    def parse_text(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            logger.error(f"Error reading text file: {e}")
            return ""
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from resume file based on extension.
        Supports: .pdf, .docx, .txt
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Resume file not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self.parse_pdf(str(file_path))
        elif extension in ['.docx', '.doc']:
            return self.parse_docx(str(file_path))
        elif extension == '.txt':
            return self.parse_text(str(file_path))
        else:
            raise ValueError(f"Unsupported file format: {extension}. Use .pdf, .docx, or .txt")
    
    def extract_structured_data(self, resume_text: str, llm_client=None) -> Dict[str, Any]:
        """
        Use LLM to extract structured data from resume text.

        Returns:
            Dict with: personal, skills, experience, education, certifications, projects
        """
        # LLM prompt for structured extraction
        extraction_prompt = (
            "Extract structured information from this resume and return ONLY valid JSON "
            "(no markdown, no code blocks, no commentary).\n\n"
            f"RESUME:\n{resume_text}\n\n"
            "Return this EXACT JSON structure (use null for missing fields):\n"
            "{\n"
            '  "personal": {"name": "Full Name", "email": "...", "phone": "...", '
            '"location": "City, State", "linkedin": "...", "github": "..."},\n'
            '  "summary": "Brief professional summary in 1-2 sentences",\n'
            '  "skills": ["Skill1", "Skill2"],\n'
            '  "experience": [{"company": "...", "role": "...", "duration": "...", '
            '"years": 2.5, "domain": "...", "description": "..."}],\n'
            '  "education": [{"degree": "...", "field": "...", "institution": "...", '
            '"year": 2020, "grade": "..."}],\n'
            '  "projects": [{"name": "...", "description": "...", "technologies": ["..."]}],\n'
            '  "certifications": ["..."],\n'
            '  "languages": ["..."]\n'
            "}\n\n"
            "RULES:\n"
            "- Extract the candidate's REAL name from the very top of the resume.\n"
            "- Put work history in `experience` sorted MOST RECENT FIRST.\n"
            "- For `years`, estimate decimal years for each role.\n"
            "- Extract every skill mentioned (not just popular ones).\n"
            "- Return ONLY JSON. No prose, no markdown fences."
        )

        # ── Try direct Groq HTTP call first (most reliable) ────────────────
        result = self._call_groq_llm(extraction_prompt)
        if result is not None:
            return result

        # ── Try the passed-in autogen client as a fallback ─────────────────
        if llm_client is not None:
            try:
                response = llm_client.create(
                    messages=[{"role": "user", "content": extraction_prompt}],
                    temperature=0.2,
                    max_tokens=2000,
                )
                text = response.choices[0].message.content.strip()
                return self._parse_json_response(text)
            except Exception as e:
                logger.warning(f"autogen llm_client.create failed: {e}")

        # ── Final fallback: heuristic mock extraction ──────────────────────
        logger.warning("LLM extraction unavailable — falling back to heuristic parser")
        return self._mock_extraction(resume_text)

    def _call_groq_llm(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Direct Groq Chat Completions call. Returns parsed JSON or None."""
        import os
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        try:
            import httpx
            model = os.getenv("GROQ_RESUME_MODEL") or os.getenv(
                "GROQ_MODEL", "openai/gpt-oss-120b"
            )
            with httpx.Client(timeout=60) as client:
                resp = client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are a precise resume parser. Output ONLY a single valid JSON "
                                    "object. No prose, no markdown fences, no commentary."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 3000,
                    },
                )
            if resp.status_code != 200:
                logger.warning("Groq resume extraction HTTP %s: %s", resp.status_code, resp.text[:300])
                return None
            text = resp.json()["choices"][0]["message"]["content"]
            return self._parse_json_response(text)
        except Exception as e:
            logger.warning(f"Groq direct call failed: {e}")
            return None

    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Strip code fences and parse JSON. Returns None on failure."""
        if not text:
            return None
        t = text.strip()
        if t.startswith("```"):
            # Remove ```json ... ``` fences
            t = t.strip("`")
            if t.lower().startswith("json"):
                t = t[4:]
            t = t.strip()
        # Find outermost JSON object
        s, e = t.find("{"), t.rfind("}")
        if s == -1 or e == -1 or e <= s:
            return None
        try:
            return json.loads(t[s : e + 1])
        except Exception as ex:
            logger.warning(f"Failed to parse LLM JSON: {ex}")
            return None

    def _mock_extraction(self, resume_text: str) -> Dict[str, Any]:
        """
        Mock extraction for testing without LLM.
        Does simple text analysis.
        """
        lines = resume_text.split('\n')
        first_line = lines[0] if lines else "Unknown"
        
        # Simple email extraction
        email = None
        phone = None
        for line in lines:
            if '@' in line and not email:
                words = line.split()
                for word in words:
                    if '@' in word:
                        email = word.strip('.,;:()')
                        break
            if any(char.isdigit() for char in line) and len([c for c in line if c.isdigit()]) >= 10:
                if not phone:
                    phone = ''.join(filter(lambda x: x.isdigit() or x in ['+', '-'], line))[:15]
        
        # Extract common programming keywords as skills
        common_skills = [
            'Python', 'Java', 'JavaScript', 'C++', 'SQL', 'React', 'Node.js',
            'FastAPI', 'Django', 'Flask', 'PostgreSQL', 'MongoDB', 'Docker',
            'Kubernetes', 'AWS', 'Azure', 'GCP', 'Git', 'Linux', 'ML', 'AI',
            'TensorFlow', 'PyTorch', 'LLM', 'AutoGen', 'REST API'
        ]
        
        found_skills = [skill for skill in common_skills if skill.lower() in resume_text.lower()]
        
        return {
            "personal": {
                "name": first_line[:50],
                "email": email or "unknown@example.com",
                "phone": phone or "+91-0000000000",
                "location": "Not specified",
                "linkedin": None,
                "github": None
            },
            "summary": "Professional with experience in software development",
            "skills": found_skills[:15] if found_skills else ["Programming", "Problem Solving"],
            "experience": [
                {
                    "company": "Previous Company",
                    "role": "Software Developer",
                    "duration": "2020 - Present",
                    "years": 3,
                    "domain": "Software Development",
                    "description": "Worked on various projects"
                }
            ],
            "education": [
                {
                    "degree": "Bachelor's Degree",
                    "field": "Computer Science",
                    "institution": "University",
                    "year": 2020,
                    "grade": "N/A"
                }
            ],
            "projects": [],
            "certifications": [],
            "languages": ["English"]
        }
    
    def save_profile(self, user_id: str, profile_data: Dict[str, Any]) -> str:
        """
        Save parsed resume profile to JSON file.
        
        Returns:
            Path to saved profile
        """
        profiles_file = Path("data/user_profiles.json")
        profiles_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing profiles
        if profiles_file.exists():
            with open(profiles_file, 'r', encoding='utf-8') as f:
                profiles = json.load(f)
        else:
            profiles = {}
        
        # Add/update profile
        profiles[user_id] = {
            **profile_data,
            "user_id": user_id,
            "updated_at": "2026-04-20T00:00:00Z"  # Use datetime in production
        }
        
        # Save back
        with open(profiles_file, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved profile for user: {user_id}")
        return str(profiles_file)
    
    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load user profile from storage"""
        profiles_file = Path("data/user_profiles.json")
        
        if not profiles_file.exists():
            return None
        
        with open(profiles_file, 'r', encoding='utf-8') as f:
            profiles = json.load(f)
        
        return profiles.get(user_id)
