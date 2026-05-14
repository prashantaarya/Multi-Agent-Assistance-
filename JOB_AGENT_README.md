# Job Application Agent - Complete Implementation

## 🎯 Overview

Implemented a **2-agent system** for intelligent job search and application assistance:

1. **Resume Agent** - Parse resumes, analyze fit, manage preferences
2. **Job Agent** - Search jobs, rank by match score, track applications

## 🏗️ Architecture

```
User Request → Planner → Resume/Job Agent → Tools → Structured Response
```

### Resume Agent Capabilities
- `parse_resume` - Extract structured data from PDF/DOCX/TXT
- `get_profile` - Retrieve user's resume profile
- `analyze_fit` - Match resume against job description
- `update_preferences` - Update job search preferences

### Job Agent Capabilities
- `search_jobs` - Search via Google Jobs (SerpAPI or mock)
- `rank_jobs` - Rank jobs by resume match score (0-100%)
- `get_job_details` - Get full job description
- `track_application` - Log job applications

## 📂 Files Created

### Core Files
- `agents/resume_agent.py` - Resume parsing and analysis agent
- `agents/job_agent.py` - Job search and matching agent
- `tools/resume_parser.py` - PDF/DOCX/TXT parser with LLM extraction
- `tools/job_search.py` - SerpAPI integration for Google Jobs

### Data Files
- `data/resumes/` - Resume storage directory
- `data/user_profiles.json` - Parsed user profiles
- `data/job_listings.json` - Search results cache
- `data/applications.json` - Application tracking

### Test Files
- `tests/test_resume_job_agents.py` - Comprehensive test suite
- `data/resumes/sample_resume.txt` - Sample resume for testing

### Modified Files
- `agents/base_agents.py` - Added Resume + Job agents initialization and routing
- `agents/domain_agents.py` - Registered both agents with tool schemas

## 🚀 Setup

### 1. Install Dependencies

```bash
# Required
pip install aiohttp

# Optional (for PDF/DOCX parsing)
pip install PyPDF2 python-docx
```

### 2. Configure API Keys (Optional)

Add to `.env` for real API calls:

```env
# For real job search (100 free searches/month)
SERPAPI_KEY=your_serpapi_key_here
```

**Without API key:** System automatically uses **mock mode** with 5 sample jobs.

### 3. Test the Agents

```bash
python tests/test_resume_job_agents.py
```

Expected output:
- ✅ Resume parsed (skills extracted)
- ✅ Jobs searched (mock/real data)
- ✅ Jobs ranked by match score
- ✅ Application tracked

## 💡 Usage Examples

### Via API (Swagger UI: http://localhost:8000/docs)

**1. Parse Resume**
```
POST /ask-direct
{
  "query": "parse my resume from data/resumes/sample_resume.txt as user123"
}
```

Response:
```json
{
  "response": "✅ Resume Parsed Successfully\nName: Rahul Kumar\nSkills: Python, FastAPI, Docker...",
  "data": {
    "personal": {"name": "Rahul Kumar", "email": "..."},
    "skills": ["Python", "FastAPI", "Docker", ...],
    "total_years": 4.0,
    "experience": [...]
  }
}
```

**2. Search Jobs**
```
POST /ask-direct
{
  "query": "find Python developer jobs in Delhi"
}
```

Response:
```json
{
  "response": "🔍 Found 5 jobs for Python Developer...",
  "data": {
    "jobs": [
      {
        "job_id": "mock_001",
        "title": "Senior Python Developer",
        "company": "TechCorp India",
        "location": "Delhi, India",
        "salary": "15-20 LPA",
        "skills_required": ["Python", "FastAPI", "Docker", ...]
      }
    ],
    "total": 5
  }
}
```

**3. Rank Jobs by Match**
```
POST /ask-direct
{
  "query": "rank these jobs for user123"
}
```

Response:
```json
{
  "response": "🎯 Ranked 5 Jobs by Match Score\nAverage Match: 72%\n1. Senior Python Developer at TechCorp - 85% match...",
  "data": {
    "ranked_jobs": [
      {"job_id": "mock_001", "match_score": 85, ...}
    ],
    "average_match": 72.4
  }
}
```

**4. Track Application**
```
POST /ask-direct
{
  "query": "I applied to job mock_001 as user123"
}
```

**5. Analyze Job Fit**
```
POST /ask-direct
{
  "query": "how well do I match this job: Python developer with 3 years experience"
}
```

## 🧮 Match Score Algorithm

Jobs are ranked 0-100% based on:

| Factor | Weight | Description |
|--------|--------|-------------|
| **Skills Match** | 60% | Overlap between user skills and job requirements |
| **Experience Match** | 30% | Years of experience vs job requirement |
| **Location Preference** | 10% | Match with user's preferred locations |

### Example Calculation

User: 4 years experience, Skills: [Python, FastAPI, Docker, AWS, PostgreSQL]
Job: Requires [Python, FastAPI, Docker, Kubernetes, MongoDB], 3-5 years

- Skills Match: 3/5 matched = 36/60 points
- Experience Match: 4 years (within 3-5 range) = 30/30 points
- Location: Delhi (matches) = 10/10 points
- **Total Score: 76%** ✅ Good match!

## 📊 Data Flow

### 1. Resume Parsing Flow
```
User uploads resume → extract_text(file) → 
  → LLM extraction (or mock) → 
  → save_profile(user_id, data) → 
  → structured response {personal, skills, experience}
```

### 2. Job Search Flow
```
User query → parse keywords + location → 
  → SerpAPI call (or mock) → 
  → structure job data → 
  → save to job_listings.json → 
  → ranked list with metadata
```

### 3. Job Matching Flow
```
Job search results → load user profile → 
  → calculate_match_score(skills, experience, location) → 
  → rank by score → 
  → filter by min_score → 
  → return ranked list
```

## 🎨 Mock Mode Features

System works **without API keys** for testing:

### Resume Parser Mock Mode
- Extracts skills from resume text via keyword matching
- Finds common programming skills (Python, Java, React, etc.)
- Extracts email/phone with simple regex
- Good for testing without LLM API calls

### Job Search Mock Mode
- Returns 5 realistic mock jobs
- Filters by query keywords
- Includes tech companies with realistic JDs
- Matches user query (e.g., "ML Engineer" returns ML jobs)

## 🔧 Configuration

### Resume Agent Settings
```python
MOCK_MODE = False  # Set True to skip LLM extraction
SUPPORTED_FORMATS = ['.pdf', '.docx', '.txt']
```

### Job Agent Settings
```python
MAX_RESULTS = 100  # Google Jobs API limit
MIN_MATCH_SCORE = 50  # Default filter threshold
```

## 🧪 Testing Strategy

### Unit Tests (Included)
```bash
python tests/test_resume_job_agents.py
```

Tests cover:
- Resume parsing (mock + LLM modes)
- Profile retrieval
- Preference updates
- Job searching
- Job ranking
- Application tracking
- End-to-end workflow

### Manual Testing
```bash
# Start server
uvicorn backend.api:app --reload

# Open Swagger UI
http://localhost:8000/docs

# Try queries:
- "parse my resume"
- "find Python jobs in Delhi"
- "rank jobs for me"
- "show my profile"
```

## 🚧 Future Enhancements (Phase 4)

### Auto-Application Features
1. **Browser Automation** (Selenium/Playwright)
   - Auto-fill common fields (name, email, phone)
   - Upload resume to job portals
   - Handle multi-page application forms

2. **Smart Form Detection**
   - Identify input fields by label/placeholder
   - Map user profile to form fields
   - Handle dropdowns, checkboxes, file uploads

3. **CAPTCHA Handling**
   - Integrate 2Captcha API ($3/1000 solves)
   - Manual intervention fallback

4. **Application Templates**
   - Pre-defined mappings for popular job portals
   - LinkedIn Easy Apply automation
   - Indeed Quick Apply support

5. **Safety Features**
   - Preview before submit
   - Daily application limits (avoid spam)
   - User confirmation required
   - Track which jobs were auto-applied

### Enhanced Matching
1. **ML-Based Scoring**
   - Train model on successful applications
   - Semantic similarity (embeddings) for skill matching
   - Historical match accuracy analysis

2. **Multi-Source Job Search**
   - LinkedIn Jobs API
   - Indeed API
   - Naukri.com scraping (with rate limits)
   - Company career page monitoring

3. **Smart Recommendations**
   - "Add these skills to boost matches by 20%"
   - "Apply to jobs in these companies (high match rate)"
   - "Best time to apply: Monday 9-11 AM"

## 📝 Example Workflow

### Complete Job Hunt Session

```python
# 1. User uploads resume
"parse my resume from data/resumes/john_resume.pdf as john_doe"
→ ✅ Resume parsed: 15 skills, 3.5 years experience

# 2. Set job preferences
"update my job preferences: Delhi, Bangalore, Remote for Backend roles"
→ ✅ Preferences updated

# 3. Search for relevant jobs
"find Backend Python jobs in Delhi"
→ 🔍 Found 10 jobs

# 4. Rank jobs by match
"rank these jobs for me"
→ 🎯 Ranked 7 jobs (average match: 68%)
   1. Senior Python Developer - 85% match
   2. Backend Engineer - 78% match
   3. Full Stack Developer - 62% match

# 5. Get details for top match
"show me details for job_001"
→ 📋 Senior Python Developer at TechCorp
   Required: Python, FastAPI, Docker, AWS
   Salary: 15-20 LPA
   Posted: 2 days ago

# 6. Analyze fit
"how well do I match this job?"
→ 🎯 Match: 85%
   Matching skills: Python, FastAPI, Docker, AWS
   Missing skills: Kubernetes
   Recommendation: Excellent match! Apply now

# 7. Track application
"I applied to this job"
→ ✅ Application logged
   Total applications: 1
```

## 🐛 Troubleshooting

### Resume not parsing
- Check file path is correct
- Ensure file is UTF-8 encoded
- Install PyPDF2 for PDF: `pip install PyPDF2`
- Install python-docx for DOCX: `pip install python-docx`

### No jobs found
- Verify SERPAPI_KEY in .env (or use mock mode)
- Check query keywords are common job titles
- Try broader location (e.g., "India" instead of specific city)

### Low match scores
- Ensure resume is parsed first
- Check user profile has skills listed
- Verify job describes required skills
- Try min_score=40 for testing

### Agents not registered
- Check agent initialization in base_agents.py
- Verify capability registration in agent __init__
- Look for registration logs in console output

## 📈 Success Metrics

Track these in production:
- Resume parsing success rate
- Average job match score
- Applications per user per day
- Time saved vs manual job search
- Interview conversion rate

## 🔐 Privacy & Security

- User profiles stored locally in JSON (should migrate to encrypted DB)
- No resume data sent to third parties (except LLM for parsing)
- SerpAPI only receives search queries, not user data
- Application tracking stays local

## 📚 Resources

- SerpAPI Docs: https://serpapi.com/google-jobs-api
- AutoGen v0.4: https://microsoft.github.io/autogen/
- PyPDF2: https://pypdf2.readthedocs.io/
- python-docx: https://python-docx.readthedocs.io/

---

## ✅ Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Resume Parser | ✅ Complete | PDF/DOCX/TXT support, mock + LLM modes |
| Resume Agent | ✅ Complete | 4 capabilities registered |
| Job Search | ✅ Complete | SerpAPI + mock mode |
| Job Agent | ✅ Complete | 4 capabilities registered |
| Job Matching | ✅ Complete | Algorithm with 60/30/10 weighting |
| Application Tracking | ✅ Complete | JSON file storage |
| Agent Registration | ✅ Complete | Both agents in domain_agents.py |
| Planner Routing | ✅ Complete | Examples added to base_agents.py |
| Structured Data | ✅ Complete | {response, data} format |
| Tests | ✅ Complete | Comprehensive test suite |
| Auto-Application | ⏳ Future | Phase 4 enhancement |
| Multi-Source Search | ⏳ Future | LinkedIn, Indeed integration |

**Total LOC Added:** ~2,500 lines
**New Files:** 7
**Modified Files:** 2
**Test Coverage:** 8 capabilities tested

🚀 **System is ready for testing!**
