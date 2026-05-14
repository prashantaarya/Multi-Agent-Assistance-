"""
Multi-source job search orchestrator.

Runs scrapers in parallel, merges + deduplicates results, ranks against resume.
Falls back gracefully — partial results > no results.
"""

from __future__ import annotations

import asyncio
import logging
import os
from time import time
from typing import Dict, List, Optional, Set, Tuple

from core.job_models import (
    JobListing,
    JobSearchFilter,
    JobSearchResult,
    JobSource,
)
from tools.application_tracker import get_tracker
from tools.job_scrapers import SCRAPER_REGISTRY, get_scraper

logger = logging.getLogger("jarvis.job_orchestrator")

# Default sources if user doesn't specify
DEFAULT_SOURCES = [JobSource.NAUKRI, JobSource.LINKEDIN, JobSource.INDEED]


# ════════════════════════════════════════════════════════════════════════════
# RANKING
# ════════════════════════════════════════════════════════════════════════════

def calculate_match_score(job: JobListing, profile: dict) -> Tuple[float, float, float, float]:
    """
    Score 0–100 based on resume match.
    Returns (total, skills, experience, location).
    """
    if not profile:
        return 0.0, 0.0, 0.0, 0.0

    user_skills = {s.lower().strip() for s in (profile.get("skills") or [])}
    job_skills = {s.lower().strip() for s in (job.skills_required or [])}

    # Skills (60%)
    if user_skills and job_skills:
        overlap = len(user_skills & job_skills)
        union = len(job_skills) or 1
        skills_score = min(100.0, (overlap / union) * 100.0)
    elif user_skills and (job.description or job.title):
        # Fall back: count user skills mentioned in description
        text = f"{job.title} {job.description or ''}".lower()
        mentions = sum(1 for s in user_skills if s and s in text)
        skills_score = min(100.0, (mentions / max(5, len(user_skills) // 3)) * 100.0)
    else:
        skills_score = 0.0

    # Experience (30%)
    user_years = 0
    exp_list = profile.get("experience") or []
    if exp_list:
        user_years = max((e.get("years", 0) or 0) for e in exp_list)

    exp_score = 0.0
    if job.experience_required and user_years:
        import re
        nums = re.findall(r"\d+", job.experience_required)
        if nums:
            req_min = int(nums[0])
            if user_years >= req_min:
                exp_score = 100.0
            else:
                exp_score = max(0.0, (user_years / max(1, req_min)) * 100.0)
        else:
            exp_score = 70.0
    elif user_years:
        exp_score = 70.0  # No requirement listed → neutral positive

    # Location (10%)
    user_loc = (profile.get("personal", {}).get("location") or "").lower()
    job_loc = (job.location or "").lower()
    loc_score = 100.0
    if user_loc:
        if user_loc in job_loc or job_loc in user_loc:
            loc_score = 100.0
        elif "remote" in job_loc:
            loc_score = 90.0
        else:
            loc_score = 50.0

    total = (skills_score * 0.6) + (exp_score * 0.3) + (loc_score * 0.1)
    return round(total, 1), round(skills_score, 1), round(exp_score, 1), round(loc_score, 1)


# ════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════

class JobSearchOrchestrator:
    """Runs multi-source scrapers + ranks + deduplicates."""

    def __init__(self) -> None:
        self.tracker = get_tracker()
        self.scrape_timeout_s = int(os.getenv("SCRAPE_TIMEOUT_S", "60"))

    async def search(
        self,
        filters: JobSearchFilter,
        user_profile: Optional[dict] = None,
        exclude_applied: bool = True,
    ) -> JobSearchResult:
        """
        Execute multi-source search.

        Returns JobSearchResult with:
          - jobs sorted by match_score desc
          - sources_searched / sources_failed for transparency
        """
        t0 = time()
        sources = filters.sources or DEFAULT_SOURCES
        logger.info("Search '%s' in '%s' across %s", filters.query, filters.location, [s.value for s in sources])

        # Per-source target count — fetch a bit more than needed for dedup
        per_source = max(5, filters.num_results)

        tasks = []
        for src in sources:
            scraper = get_scraper(src)
            if scraper:
                tasks.append(self._scrape_one(scraper, filters, per_source))

        all_jobs: List[JobListing] = []
        sources_searched: List[JobSource] = []
        sources_failed: List[Dict[str, str]] = []

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for src, result in zip(sources, results):
                if isinstance(result, Exception):
                    logger.warning("Source %s failed: %s", src.value, result)
                    sources_failed.append({"source": src.value, "error": str(result)[:200]})
                else:
                    sources_searched.append(src)
                    all_jobs.extend(result)

        # Dedup by company+title
        all_jobs = self._dedup(all_jobs)

        # Exclude already-applied
        if exclude_applied:
            all_jobs = self.tracker.filter_new_jobs(all_jobs)

        # Rank
        if user_profile:
            for j in all_jobs:
                total, sk, ex, loc = calculate_match_score(j, user_profile)
                j.match_score = total
                j.skills_match = sk
                j.experience_match = ex
                j.location_match = loc
            all_jobs.sort(key=lambda j: j.match_score or 0, reverse=True)

        # Truncate to requested count
        all_jobs = all_jobs[: filters.num_results]

        return JobSearchResult(
            query=filters.query,
            location=filters.location,
            jobs=all_jobs,
            total=len(all_jobs),
            sources_searched=sources_searched,
            sources_failed=sources_failed,
            duration_ms=int((time() - t0) * 1000),
            matched_with_resume=user_profile is not None,
        )

    async def _scrape_one(self, scraper, filters: JobSearchFilter, n: int) -> List[JobListing]:
        """Run a single scraper with timeout."""
        try:
            return await asyncio.wait_for(
                scraper.scrape(filters.query, filters.location, n),
                timeout=self.scrape_timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("%s timed out after %ds", scraper.source.value, self.scrape_timeout_s)
            return []

    @staticmethod
    def _dedup(jobs: List[JobListing]) -> List[JobListing]:
        """Remove duplicates by (company, title). Keep best source."""
        # Priority: linkedin > naukri > indeed (richer data)
        priority = {
            JobSource.LINKEDIN: 1,
            JobSource.NAUKRI: 2,
            JobSource.INDEED: 3,
            JobSource.GLASSDOOR: 4,
            JobSource.WORKDAY: 5,
        }
        seen: Dict[str, JobListing] = {}
        for j in jobs:
            key = j.dedup_key()
            existing = seen.get(key)
            if not existing:
                seen[key] = j
            else:
                # Keep the higher-priority source
                if priority.get(j.source, 99) < priority.get(existing.source, 99):
                    seen[key] = j
        return list(seen.values())


# Singleton
_orchestrator: Optional[JobSearchOrchestrator] = None


def get_orchestrator() -> JobSearchOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = JobSearchOrchestrator()
    return _orchestrator
