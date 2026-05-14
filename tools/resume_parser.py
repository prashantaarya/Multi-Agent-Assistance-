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
            Dict with: personal, skills, experience, education, preferences
        """
        if not llm_client:
            # Return mock data if no LLM available (for testing)
            return self._mock_extraction(resume_text)
        
        # LLM prompt for structured extraction
        extraction_prompt = f"""
Extract structured information from this resume and return ONLY valid JSON (no markdown, no code blocks):

{resume_text}

Return this exact JSON structure:
{{
  "personal": {{
    "name": "Full Name",
    "email": "email@example.com",
    "phone": "+91-XXXXXXXXXX",
    "location": "City, State",
    "linkedin": "linkedin.com/in/username",
    "github": "github.com/username"
  }},
  "summary": "Brief professional summary",
  "skills": ["Skill1", "Skill2", "Skill3"],
  "experience": [
    {{
      "company": "Company Name",
      "role": "Job Title",
      "duration": "Jan 2020 - Present",
      "years": 2.5,
      "domain": "Industry/Domain",
      "description": "Brief role description"
    }}
  ],
  "education": [
    {{
      "degree": "B.Tech/M.Tech/etc",
      "field": "Computer Science",
      "institution": "University Name",
      "year": 2020,
      "grade": "8.5 CGPA"
    }}
  ],
  "projects": [
    {{
      "name": "Project Name",
      "description": "What it does",
      "technologies": ["Tech1", "Tech2"]
    }}
  ],
  "certifications": ["Cert1", "Cert2"],
  "languages": ["English", "Hindi"]
}}

Extract all available information. Use null for missing fields. Return ONLY JSON.
"""
        
        try:
            # Call LLM (assuming Groq/OpenAI-compatible client)
            response = llm_client.create(
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            
            # Parse JSON from response
            result_text = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            
            return json.loads(result_text)
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return self._mock_extraction(resume_text)
    
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
