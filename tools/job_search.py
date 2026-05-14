# tools/job_search.py
"""
Job Search Tool using SerpAPI

Searches for jobs across Google Jobs using SerpAPI.
Free tier: 100 searches/month

Setup:
1. Get API key from serpapi.com
2. Add to .env: SERPAPI_KEY=your_key_here
"""

import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

logger = logging.getLogger("jarvis.job_search")

class JobSearcher:
    """
    Job search using SerpAPI for Google Jobs.
    """
    
    def __init__(self):
        self.api_key = os.getenv("SERPAPI_KEY")
        self.job_listings_file = Path("data/job_listings.json")
        self.job_listings_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.api_key:
            logger.warning("SERPAPI_KEY not set. Job search will use mock mode.")
    
    async def search_jobs(
        self,
        query: str,
        location: str = "",
        num_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for jobs using Google Jobs via SerpAPI.
        
        Args:
            query: Job search query (e.g., "Python Developer", "Backend Engineer")
            location: Location filter (e.g., "Delhi, India", "Remote")
            num_results: Number of results to return
        
        Returns:
            List of job dictionaries with:
                - job_id: Unique identifier
                - title: Job title
                - company: Company name
                - location: Job location
                - description: Job description
                - posted_date: When posted
                - url: Application URL
                - salary: Salary info (if available)
                - job_type: Full-time/Contract/etc
        """
        
        if not self.api_key:
            logger.info("Using mock job data (SERPAPI_KEY not set)")
            return self._mock_job_search(query, location, num_results)
        
        try:
            import aiohttp
            
            params = {
                "engine": "google_jobs",
                "q": query,
                "api_key": self.api_key,
                "num": min(num_results, 100)  # Google Jobs max is 100
            }
            
            if location:
                params["location"] = location
            
            url = "https://serpapi.com/search"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.error(f"SerpAPI error: HTTP {resp.status}")
                        return self._mock_job_search(query, location, num_results)
                    
                    data = await resp.json()
                    
                    jobs_raw = data.get("jobs_results", [])
                    
                    if not jobs_raw:
                        logger.info(f"No jobs found for: {query}")
                        return []
                    
                    # Parse and structure job data
                    jobs = []
                    for idx, job in enumerate(jobs_raw[:num_results]):
                        structured_job = {
                            "job_id": f"serpapi_{job.get('job_id', idx)}",
                            "title": job.get("title", "Unknown Title"),
                            "company": job.get("company_name", "Unknown Company"),
                            "location": job.get("location", location or "Not specified"),
                            "description": job.get("description", "No description available"),
                            "posted_date": job.get("detected_extensions", {}).get("posted_at", "Recently"),
                            "url": job.get("apply_options", [{}])[0].get("link") if job.get("apply_options") else job.get("share_url", ""),
                            "salary": self._extract_salary(job.get("detected_extensions", {})),
                            "job_type": self._extract_job_type(job.get("detected_extensions", {})),
                            "source": "Google Jobs via SerpAPI",
                            "skills_required": self._extract_skills(job.get("description", ""))
                        }
                        jobs.append(structured_job)
                    
                    # Save to file
                    self._save_jobs(jobs)
                    
                    logger.info(f"Found {len(jobs)} jobs for: {query} in {location}")
                    return jobs
                    
        except ImportError:
            logger.warning("aiohttp not installed. Install with: pip install aiohttp")
            return self._mock_job_search(query, location, num_results)
        except Exception as e:
            logger.error(f"Job search failed: {e}")
            return self._mock_job_search(query, location, num_results)
    
    def _extract_salary(self, extensions: Dict) -> Optional[str]:
        """Extract salary from job extensions"""
        # SerpAPI returns salary in various formats
        if "salary" in extensions:
            return extensions["salary"]
        
        # Check for salary in schedule_type
        schedule = extensions.get("schedule_type", "")
        if any(keyword in schedule.lower() for keyword in ["lakh", "k", "$", "₹", "salary"]):
            return schedule
        
        return None
    
    def _extract_job_type(self, extensions: Dict) -> str:
        """Extract job type from extensions"""
        job_type = extensions.get("schedule_type", "Full-time")
        
        # Normalize
        if "full" in job_type.lower():
            return "Full-time"
        elif "part" in job_type.lower():
            return "Part-time"
        elif "contract" in job_type.lower():
            return "Contract"
        elif "intern" in job_type.lower():
            return "Internship"
        
        return job_type
    
    def _extract_skills(self, description: str) -> List[str]:
        """Extract common skills from job description"""
        common_skills = [
            'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'Go', 'Rust',
            'React', 'Angular', 'Vue', 'Node.js', 'Django', 'Flask', 'FastAPI', 'Spring',
            'SQL', 'PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'Elasticsearch',
            'Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP', 'Terraform', 'Jenkins',
            'Git', 'CI/CD', 'Agile', 'Scrum', 'REST API', 'GraphQL', 'Microservices',
            'Machine Learning', 'Deep Learning', 'NLP', 'Computer Vision', 'AI',
            'TensorFlow', 'PyTorch', 'Scikit-learn', 'LLM', 'AutoGen'
        ]
        
        found_skills = []
        desc_lower = description.lower()
        
        for skill in common_skills:
            if skill.lower() in desc_lower:
                found_skills.append(skill)
        
        return found_skills[:15]  # Limit to top 15
    
    def _mock_job_search(self, query: str, location: str, num_results: int) -> List[Dict[str, Any]]:
        """Mock job search for testing without API key"""
        mock_jobs = [
            {
                "job_id": "mock_001",
                "title": "Senior Python Developer",
                "company": "TechCorp India",
                "location": location or "Delhi, India",
                "description": "Looking for an experienced Python developer with FastAPI, Docker, and cloud experience. 3-5 years required.",
                "posted_date": "2 days ago",
                "url": "https://careers.techcorp.in/jobs/senior-python-developer-delhi",
                "salary": "15-20 LPA",
                "job_type": "Full-time",
                "source": "Mock Data (SerpAPI unavailable)",
                "skills_required": ["Python", "FastAPI", "Docker", "AWS", "PostgreSQL"]
            },
            {
                "job_id": "mock_002",
                "title": "Backend Engineer",
                "company": "StartupXYZ",
                "location": location or "Bangalore, India",
                "description": "Build scalable backend services using Python, Node.js, and microservices. Experience with LLMs is a plus.",
                "posted_date": "1 week ago",
                "url": "https://startupxyz.com/careers/backend-engineer-bangalore",
                "salary": "12-18 LPA",
                "job_type": "Full-time",
                "source": "Mock Data (SerpAPI unavailable)",
                "skills_required": ["Python", "Node.js", "Microservices", "MongoDB", "LLM"]
            },
            {
                "job_id": "mock_003",
                "title": "ML Engineer",
                "company": "AI Innovations",
                "location": location or "Bangalore, India",
                "description": "Work on cutting-edge machine learning projects. PyTorch, TensorFlow, NLP experience required. 2+ years.",
                "posted_date": "3 days ago",
                "url": "https://aiinnovations.co.in/careers/ml-engineer",
                "salary": "18-25 LPA",
                "job_type": "Full-time",
                "source": "Mock Data (SerpAPI unavailable)",
                "skills_required": ["Python", "PyTorch", "TensorFlow", "NLP", "Machine Learning"]
            },
            {
                "job_id": "mock_004",
                "title": "Full Stack Developer",
                "company": "ProductCo",
                "location": location or "Delhi NCR, India",
                "description": "Join our team to build modern web applications. React, FastAPI, PostgreSQL. Fresh graduates welcome!",
                "posted_date": "5 days ago",
                "url": "https://productco.in/jobs/fullstack-developer-delhi",
                "salary": "8-12 LPA",
                "job_type": "Full-time",
                "source": "Mock Data (SerpAPI unavailable)",
                "skills_required": ["React", "FastAPI", "PostgreSQL", "JavaScript", "Python"]
            },
            {
                "job_id": "mock_005",
                "title": "DevOps Engineer",
                "company": "CloudTech Solutions",
                "location": location or "Mumbai, India",
                "description": "Manage cloud infrastructure using AWS, Kubernetes, Terraform. CI/CD pipeline expertise required. 3+ years.",
                "posted_date": "1 day ago",
                "url": "https://cloudtechsolutions.com/careers/devops-engineer-mumbai",
                "salary": "16-22 LPA",
                "job_type": "Full-time",
                "source": "Mock Data (SerpAPI unavailable)",
                "skills_required": ["AWS", "Kubernetes", "Docker", "Terraform", "CI/CD", "Python"]
            },
        ]
        
        # Filter by query keywords
        query_lower = query.lower()
        filtered = [
            job for job in mock_jobs
            if any(word in job["title"].lower() for word in query_lower.split())
            or any(word in job["description"].lower() for word in query_lower.split())
        ]
        
        results = filtered[:num_results] if filtered else mock_jobs[:num_results]
        
        # Save to file
        self._save_jobs(results)
        
        return results
    
    def _save_jobs(self, jobs: List[Dict[str, Any]]):
        """Save job listings to JSON file"""
        try:
            # Load existing
            if self.job_listings_file.exists():
                with open(self.job_listings_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            else:
                existing = []
            
            # Add new jobs (avoid duplicates by job_id)
            existing_ids = {job["job_id"] for job in existing}
            new_jobs = [job for job in jobs if job["job_id"] not in existing_ids]
            
            all_jobs = existing + new_jobs
            
            # Save
            with open(self.job_listings_file, 'w', encoding='utf-8') as f:
                json.dump(all_jobs, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(new_jobs)} new jobs to {self.job_listings_file}")
            
        except Exception as e:
            logger.error(f"Error saving jobs: {e}")
    
    def get_saved_jobs(self, filter_by: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve saved job listings from file.
        
        Args:
            filter_by: Optional filters like {"location": "Delhi", "min_salary": 15}
        
        Returns:
            List of job dictionaries
        """
        if not self.job_listings_file.exists():
            return []
        
        try:
            with open(self.job_listings_file, 'r', encoding='utf-8') as f:
                jobs = json.load(f)
            
            if not filter_by:
                return jobs
            
            # Apply filters
            filtered = jobs
            
            if "location" in filter_by:
                loc = filter_by["location"].lower()
                filtered = [j for j in filtered if loc in j["location"].lower()]
            
            if "company" in filter_by:
                comp = filter_by["company"].lower()
                filtered = [j for j in filtered if comp in j["company"].lower()]
            
            return filtered
            
        except Exception as e:
            logger.error(f"Error loading jobs: {e}")
            return []
