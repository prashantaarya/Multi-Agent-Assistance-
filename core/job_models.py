"""
Job search domain models — Pydantic schemas for type safety.

Production-grade models for jobs, applications, search filters, and tracking.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, HttpUrl, field_validator


# ════════════════════════════════════════════════════════════════════════════
# ENUMS
# ════════════════════════════════════════════════════════════════════════════

class JobSource(str, Enum):
    """Where the job listing came from."""
    LINKEDIN = "linkedin"
    NAUKRI = "naukri"
    INDEED = "indeed"
    GLASSDOOR = "glassdoor"
    WORKDAY = "workday"
    GOOGLE_JOBS = "google_jobs"
    COMPANY_CAREER = "company_career"
    SERPAPI = "serpapi"
    MOCK = "mock"


class JobType(str, Enum):
    """Employment type."""
    FULL_TIME = "Full-time"
    PART_TIME = "Part-time"
    CONTRACT = "Contract"
    INTERNSHIP = "Internship"
    FREELANCE = "Freelance"
    UNKNOWN = "Unknown"


class ApplicationStatus(str, Enum):
    """Application lifecycle states."""
    PENDING = "pending"
    READY_TO_SUBMIT = "ready_to_submit"
    SUBMITTED = "submitted"
    IN_REVIEW = "in_review"
    INTERVIEW_SCHEDULED = "interview_scheduled"
    REJECTED = "rejected"
    OFFER_RECEIVED = "offer_received"
    WITHDRAWN = "withdrawn"
    FAILED = "failed"
    REQUIRES_MANUAL = "requires_manual"


class ApplyMode(str, Enum):
    """How auto-apply should behave."""
    AUTO = "auto"            # Fill + submit automatically
    SEMI_AUTO = "semi-auto"  # Fill, pause for review before submit
    REVIEW = "review"        # Fill only, never submit
    DRY_RUN = "dry-run"      # Don't even open browser, just simulate


# ════════════════════════════════════════════════════════════════════════════
# JOB LISTING
# ════════════════════════════════════════════════════════════════════════════

class JobListing(BaseModel):
    """
    A single job posting from any source.
    Normalized format so downstream code is source-agnostic.
    """
    # Identity
    job_id: str = Field(..., description="Unique ID: <source>_<hash>")
    source: JobSource

    # Core data
    title: str
    company: str
    location: str
    url: str = Field(..., description="Apply / detail page URL")

    # Optional details
    description: Optional[str] = None
    salary: Optional[str] = None
    job_type: JobType = JobType.UNKNOWN
    posted_date: Optional[str] = None
    experience_required: Optional[str] = None
    skills_required: List[str] = Field(default_factory=list)

    # Scoring (populated by ranking engine)
    match_score: Optional[float] = Field(None, ge=0, le=100)
    skills_match: Optional[float] = Field(None, ge=0, le=100)
    experience_match: Optional[float] = Field(None, ge=0, le=100)
    location_match: Optional[float] = Field(None, ge=0, le=100)

    # Metadata
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
    is_easy_apply: bool = False
    raw_data: Optional[Dict[str, Any]] = Field(None, exclude=True)

    @field_validator("url")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        if not v or not v.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {v}")
        return v

    @classmethod
    def make_id(cls, source: JobSource, company: str, title: str, url: str = "") -> str:
        """Generate a deterministic ID for deduplication."""
        key = f"{source.value}|{company.lower().strip()}|{title.lower().strip()}|{url}"
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        return f"{source.value}_{h}"

    def dedup_key(self) -> str:
        """Cross-source dedup key (company + title)."""
        return hashlib.sha1(
            f"{self.company.lower().strip()}|{self.title.lower().strip()}".encode()
        ).hexdigest()[:16]


# ════════════════════════════════════════════════════════════════════════════
# SEARCH FILTERS
# ════════════════════════════════════════════════════════════════════════════

class JobSearchFilter(BaseModel):
    """Filters applied to job search across all sources."""
    query: str = Field(..., min_length=1, description="Job title / keywords")
    location: str = ""
    num_results: int = Field(10, ge=1, le=100)

    # Optional filters
    sources: List[JobSource] = Field(default_factory=list, description="Empty = all sources")
    job_types: List[JobType] = Field(default_factory=list)
    max_age_days: Optional[int] = Field(None, description="Posted within N days")
    min_salary_lpa: Optional[float] = None
    remote_only: bool = False
    easy_apply_only: bool = False


# ════════════════════════════════════════════════════════════════════════════
# APPLICATION TRACKING
# ════════════════════════════════════════════════════════════════════════════

class JobApplication(BaseModel):
    """Record of a job application (auto or manual)."""
    application_id: str
    job_id: str
    user_id: str = "default_user"

    # Snapshot of job at apply time
    company: str
    title: str
    job_url: str
    source: JobSource

    # Application state
    status: ApplicationStatus = ApplicationStatus.PENDING
    apply_mode: ApplyMode = ApplyMode.SEMI_AUTO
    applied_at: Optional[datetime] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    # Auto-apply audit trail
    fields_filled: Dict[str, str] = Field(default_factory=dict)
    resume_uploaded: bool = False
    cover_letter_generated: bool = False
    submission_proof_url: Optional[str] = None  # Screenshot path
    error_message: Optional[str] = None

    # Match snapshot at apply time
    match_score_at_apply: Optional[float] = None

    @classmethod
    def from_job(
        cls,
        job: JobListing,
        user_id: str = "default_user",
        apply_mode: ApplyMode = ApplyMode.SEMI_AUTO,
    ) -> "JobApplication":
        """Create application from a job listing."""
        return cls(
            application_id=f"app_{job.job_id}_{int(datetime.utcnow().timestamp())}",
            job_id=job.job_id,
            user_id=user_id,
            company=job.company,
            title=job.title,
            job_url=job.url,
            source=job.source,
            apply_mode=apply_mode,
            match_score_at_apply=job.match_score,
        )


# ════════════════════════════════════════════════════════════════════════════
# AUTO-APPLY RESULT
# ════════════════════════════════════════════════════════════════════════════

class ApplyResult(BaseModel):
    """Outcome of an auto-apply attempt."""
    success: bool
    application: JobApplication
    submitted: bool = False
    requires_manual_review: bool = False
    reason: Optional[str] = None
    screenshots: List[str] = Field(default_factory=list)
    duration_ms: int = 0


# ════════════════════════════════════════════════════════════════════════════
# SEARCH RESULT
# ════════════════════════════════════════════════════════════════════════════

class JobSearchResult(BaseModel):
    """Aggregated result from multi-source search."""
    query: str
    location: str
    jobs: List[JobListing] = Field(default_factory=list)
    total: int = 0
    sources_searched: List[JobSource] = Field(default_factory=list)
    sources_failed: List[Dict[str, str]] = Field(default_factory=list)
    duration_ms: int = 0
    matched_with_resume: bool = False

    @property
    def top_matches(self) -> List[JobListing]:
        """Top 10 by match score."""
        return sorted(
            [j for j in self.jobs if j.match_score is not None],
            key=lambda j: j.match_score or 0,
            reverse=True,
        )[:10]
