"""
Job scrapers — Playwright-based scrapers for multiple job sites.

Architecture:
    BaseScraper (abstract)
      ├─ NaukriScraper      → naukri.com   (no login required for search)
      ├─ LinkedInScraper    → linkedin.com (login optional, more data with login)
      └─ IndeedScraper      → indeed.co.in (no login required)

All scrapers:
- Inherit retry/timeout/error handling from BaseScraper
- Return normalized JobListing objects
- Are async-safe and idempotent
"""

from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional
from urllib.parse import quote_plus, urljoin

from playwright.async_api import Page, TimeoutError as PWTimeout
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from core.job_models import JobListing, JobSource, JobType
from tools.browser_manager import BrowserManager

logger = logging.getLogger("jarvis.scrapers")


# Common skills lexicon (for extracting from descriptions)
COMMON_SKILLS = {
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi", "spring",
    "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra",
    "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "jenkins", "ansible",
    "git", "ci/cd", "agile", "scrum", "rest api", "graphql", "microservices",
    "machine learning", "deep learning", "nlp", "computer vision", "ai",
    "tensorflow", "pytorch", "scikit-learn", "llm", "langchain", "autogen",
    "kafka", "rabbitmq", "spark", "hadoop", "airflow", "snowflake", "databricks",
}


def extract_skills(text: str, limit: int = 15) -> List[str]:
    """Extract known skills from job description."""
    if not text:
        return []
    text_l = text.lower()
    found = []
    seen = set()
    for skill in COMMON_SKILLS:
        if skill in text_l and skill not in seen:
            found.append(skill.title() if skill.islower() else skill)
            seen.add(skill)
            if len(found) >= limit:
                break
    return found


# ════════════════════════════════════════════════════════════════════════════
# BASE SCRAPER
# ════════════════════════════════════════════════════════════════════════════

class BaseScraper(ABC):
    """Abstract scraper. Subclasses implement site-specific selectors."""

    source: JobSource
    site_key: str  # storage state key

    def __init__(self) -> None:
        self.bm = BrowserManager.instance()

    @abstractmethod
    async def scrape(self, query: str, location: str, num_results: int) -> List[JobListing]:
        """Site-specific scraping logic."""
        ...

    # ── Common utilities ───────────────────────────────────────────────────
    @staticmethod
    async def _safe_text(page: Page, selector: str, timeout: int = 2000) -> str:
        """Get text from selector, return '' on failure."""
        try:
            el = await page.wait_for_selector(selector, timeout=timeout, state="attached")
            if el:
                txt = await el.text_content()
                return (txt or "").strip()
        except (PWTimeout, Exception):
            pass
        return ""

    @staticmethod
    async def _safe_attr(page: Page, selector: str, attr: str, timeout: int = 2000) -> str:
        """Get attribute from selector, return '' on failure."""
        try:
            el = await page.wait_for_selector(selector, timeout=timeout, state="attached")
            if el:
                val = await el.get_attribute(attr)
                return (val or "").strip()
        except (PWTimeout, Exception):
            pass
        return ""

    @staticmethod
    def _normalize_url(href: str, base: str) -> str:
        """Resolve relative URLs."""
        if not href:
            return ""
        if href.startswith("http"):
            return href
        return urljoin(base, href)


# ════════════════════════════════════════════════════════════════════════════
# NAUKRI SCRAPER
# ════════════════════════════════════════════════════════════════════════════

class NaukriScraper(BaseScraper):
    """Naukri.com scraper — no login required for search."""

    source = JobSource.NAUKRI
    site_key = "naukri"

    BASE_URL = "https://www.naukri.com"

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=2, min=2, max=10),
        retry=retry_if_exception_type((PWTimeout, ConnectionError)),
        reraise=True,
    )
    async def scrape(self, query: str, location: str, num_results: int) -> List[JobListing]:
        # Build URL: naukri.com/python-developer-jobs-in-bangalore
        q_slug = re.sub(r"\s+", "-", query.strip().lower())
        loc_slug = re.sub(r"\s+", "-", (location or "").strip().lower())
        if loc_slug:
            url = f"{self.BASE_URL}/{q_slug}-jobs-in-{loc_slug}"
        else:
            url = f"{self.BASE_URL}/{q_slug}-jobs"

        logger.info("Naukri: scraping %s", url)
        jobs: List[JobListing] = []

        async with self.bm.new_page(site=self.site_key) as page:
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            except PWTimeout:
                logger.warning("Naukri: page load timeout")
                return []

            # Wait for job cards (Naukri uses several layouts)
            try:
                await page.wait_for_selector(
                    "div.srp-jobtuple-wrapper, article.jobTuple, div.cust-job-tuple",
                    timeout=10_000,
                )
            except PWTimeout:
                logger.warning("Naukri: no job cards found")
                return []

            cards = await page.query_selector_all(
                "div.srp-jobtuple-wrapper, article.jobTuple, div.cust-job-tuple"
            )
            logger.info("Naukri: found %d cards", len(cards))

            for card in cards[:num_results]:
                try:
                    job = await self._parse_card(card, location)
                    if job:
                        jobs.append(job)
                except Exception as e:
                    logger.debug("Naukri: skipping card: %s", e)
                    continue

        logger.info("Naukri: scraped %d jobs", len(jobs))
        return jobs

    async def _parse_card(self, card, default_location: str) -> Optional[JobListing]:
        """Parse a single Naukri job card."""
        # Title + URL
        title_el = await card.query_selector("a.title, a.cust-job-tuple__title")
        if not title_el:
            return None
        title = (await title_el.text_content() or "").strip()
        url = await title_el.get_attribute("href") or ""
        url = self._normalize_url(url, self.BASE_URL)
        if not title or not url:
            return None

        # Company
        company_el = await card.query_selector("a.comp-name, a.subTitle, a.cust-job-tuple__company-name")
        company = (await company_el.text_content()).strip() if company_el else "Unknown"

        # Location
        loc_el = await card.query_selector("span.locWdth, span.cust-job-tuple__location-data, li.location")
        loc = (await loc_el.text_content()).strip() if loc_el else default_location

        # Experience
        exp_el = await card.query_selector("span.expwdth, span.cust-job-tuple__experience-data, li.experience")
        exp = (await exp_el.text_content()).strip() if exp_el else None

        # Salary
        sal_el = await card.query_selector("span.sal, span.cust-job-tuple__salary-data, li.salary")
        salary = (await sal_el.text_content()).strip() if sal_el else None

        # Description preview (tags / skills)
        desc_el = await card.query_selector("span.job-desc, ul.tags-gt, div.cust-job-tuple__description")
        description = (await desc_el.text_content()).strip() if desc_el else ""

        # Posted date
        posted_el = await card.query_selector("span.job-post-day, span.cust-job-tuple__posted-date")
        posted = (await posted_el.text_content()).strip() if posted_el else None

        return JobListing(
            job_id=JobListing.make_id(self.source, company, title, url),
            source=self.source,
            title=title,
            company=company,
            location=loc,
            url=url,
            description=description,
            salary=salary,
            posted_date=posted,
            experience_required=exp,
            skills_required=extract_skills(f"{title} {description}"),
            job_type=JobType.FULL_TIME,
        )


# ════════════════════════════════════════════════════════════════════════════
# LINKEDIN SCRAPER
# ════════════════════════════════════════════════════════════════════════════

class LinkedInScraper(BaseScraper):
    """LinkedIn Jobs scraper (public search — no login needed for basic listings)."""

    source = JobSource.LINKEDIN
    site_key = "linkedin"

    BASE_URL = "https://www.linkedin.com"

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=2, min=2, max=10),
        retry=retry_if_exception_type((PWTimeout, ConnectionError)),
        reraise=True,
    )
    async def scrape(self, query: str, location: str, num_results: int) -> List[JobListing]:
        # Public job search (no login required)
        # https://www.linkedin.com/jobs/search/?keywords=...&location=...
        url = (
            f"{self.BASE_URL}/jobs/search/"
            f"?keywords={quote_plus(query)}"
            f"&location={quote_plus(location or '')}"
            f"&f_TPR=r604800"  # Posted in past week
        )

        logger.info("LinkedIn: scraping %s", url)
        jobs: List[JobListing] = []

        async with self.bm.new_page(site=self.site_key) as page:
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            except PWTimeout:
                logger.warning("LinkedIn: page load timeout")
                return []

            # LinkedIn shows a join modal — try to dismiss
            try:
                await page.locator("button[aria-label='Dismiss']").first.click(timeout=3000)
            except Exception:
                pass

            # Wait for job cards
            try:
                await page.wait_for_selector(
                    "div.base-card, ul.jobs-search__results-list li, div.job-search-card",
                    timeout=10_000,
                )
            except PWTimeout:
                logger.warning("LinkedIn: no job cards found (may need login)")
                return []

            # Scroll to load more results
            for _ in range(min(3, max(1, num_results // 10))):
                await page.mouse.wheel(0, 3000)
                await asyncio.sleep(1.0)

            cards = await page.query_selector_all(
                "div.base-card, ul.jobs-search__results-list > li, div.job-search-card"
            )
            logger.info("LinkedIn: found %d cards", len(cards))

            for card in cards[:num_results]:
                try:
                    job = await self._parse_card(card, location)
                    if job:
                        jobs.append(job)
                except Exception as e:
                    logger.debug("LinkedIn: skipping card: %s", e)
                    continue

        logger.info("LinkedIn: scraped %d jobs", len(jobs))
        return jobs

    async def _parse_card(self, card, default_location: str) -> Optional[JobListing]:
        # Title
        title_el = await card.query_selector("h3.base-search-card__title, h3.job-search-card__title")
        title = (await title_el.text_content()).strip() if title_el else ""

        # URL — LinkedIn wraps the whole card in an anchor
        link_el = await card.query_selector("a.base-card__full-link, a.job-search-card__link")
        url = await link_el.get_attribute("href") if link_el else ""
        url = (url or "").split("?")[0]  # strip tracking params

        if not title or not url:
            return None

        # Company
        comp_el = await card.query_selector("h4.base-search-card__subtitle, h4.job-search-card__subtitle")
        company = (await comp_el.text_content()).strip() if comp_el else "Unknown"

        # Location
        loc_el = await card.query_selector("span.job-search-card__location")
        loc = (await loc_el.text_content()).strip() if loc_el else default_location

        # Posted date
        time_el = await card.query_selector("time")
        posted = (await time_el.get_attribute("datetime")) if time_el else None

        return JobListing(
            job_id=JobListing.make_id(self.source, company, title, url),
            source=self.source,
            title=title,
            company=company,
            location=loc,
            url=url,
            posted_date=posted,
            skills_required=extract_skills(title),
            job_type=JobType.FULL_TIME,
            is_easy_apply=False,  # Easy Apply detection requires login
        )


# ════════════════════════════════════════════════════════════════════════════
# INDEED SCRAPER
# ════════════════════════════════════════════════════════════════════════════

class IndeedScraper(BaseScraper):
    """Indeed.co.in scraper — public search, no login."""

    source = JobSource.INDEED
    site_key = "indeed"

    BASE_URL = "https://in.indeed.com"

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=2, min=2, max=10),
        retry=retry_if_exception_type((PWTimeout, ConnectionError)),
        reraise=True,
    )
    async def scrape(self, query: str, location: str, num_results: int) -> List[JobListing]:
        url = (
            f"{self.BASE_URL}/jobs"
            f"?q={quote_plus(query)}"
            f"&l={quote_plus(location or '')}"
            f"&fromage=7"  # Past 7 days
        )

        logger.info("Indeed: scraping %s", url)
        jobs: List[JobListing] = []

        async with self.bm.new_page(site=self.site_key) as page:
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            except PWTimeout:
                logger.warning("Indeed: page load timeout")
                return []

            # Cloudflare check or no results
            try:
                await page.wait_for_selector(
                    "div.job_seen_beacon, a.tapItem, ul.jobsearch-ResultsList li",
                    timeout=10_000,
                )
            except PWTimeout:
                logger.warning("Indeed: no job cards (may be blocked)")
                return []

            cards = await page.query_selector_all(
                "div.job_seen_beacon, a.tapItem, ul.jobsearch-ResultsList > li"
            )
            logger.info("Indeed: found %d cards", len(cards))

            for card in cards[:num_results]:
                try:
                    job = await self._parse_card(card, location)
                    if job:
                        jobs.append(job)
                except Exception as e:
                    logger.debug("Indeed: skipping card: %s", e)
                    continue

        logger.info("Indeed: scraped %d jobs", len(jobs))
        return jobs

    async def _parse_card(self, card, default_location: str) -> Optional[JobListing]:
        # Title
        title_el = await card.query_selector("h2.jobTitle span, a.jcs-JobTitle span, h2.jobTitle a")
        title = (await title_el.text_content()).strip() if title_el else ""

        # URL
        link_el = await card.query_selector("a.jcs-JobTitle, h2.jobTitle a")
        href = await link_el.get_attribute("href") if link_el else ""
        url = self._normalize_url(href, self.BASE_URL)

        if not title or not url:
            return None

        # Company
        comp_el = await card.query_selector("span.companyName, [data-testid='company-name']")
        company = (await comp_el.text_content()).strip() if comp_el else "Unknown"

        # Location
        loc_el = await card.query_selector("div.companyLocation, [data-testid='text-location']")
        loc = (await loc_el.text_content()).strip() if loc_el else default_location

        # Salary
        sal_el = await card.query_selector("div.salary-snippet-container, [data-testid='attribute_snippet_testid']")
        salary = (await sal_el.text_content()).strip() if sal_el else None

        # Snippet
        snippet_el = await card.query_selector("div.job-snippet, [data-testid='job-snippet']")
        description = (await snippet_el.text_content()).strip() if snippet_el else ""

        return JobListing(
            job_id=JobListing.make_id(self.source, company, title, url),
            source=self.source,
            title=title,
            company=company,
            location=loc,
            url=url,
            description=description,
            salary=salary,
            skills_required=extract_skills(f"{title} {description}"),
            job_type=JobType.FULL_TIME,
        )


# ════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ════════════════════════════════════════════════════════════════════════════

SCRAPER_REGISTRY = {
    JobSource.NAUKRI: NaukriScraper,
    JobSource.LINKEDIN: LinkedInScraper,
    JobSource.INDEED: IndeedScraper,
}


def get_scraper(source: JobSource) -> Optional[BaseScraper]:
    cls = SCRAPER_REGISTRY.get(source)
    return cls() if cls else None
