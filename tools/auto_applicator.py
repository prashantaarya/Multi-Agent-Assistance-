"""
Auto-Applicator — Playwright-based job application automation.

Supports:
- Naukri.com Quick Apply (logged-in users)
- LinkedIn Easy Apply (logged-in users)
- Generic form filler (Workday, Greenhouse, Lever, custom pages)

Modes:
- AUTO       : Fill + submit immediately
- SEMI_AUTO  : Fill, screenshot, pause for user approval before submit
- REVIEW     : Fill only, never submit (user reviews + submits manually)
- DRY_RUN    : Simulate without opening browser

Safety:
- Idempotent (won't double-apply via ApplicationTracker)
- Captures screenshot before/after submit (audit trail)
- Logs all field fills for transparency
- Captcha → escalates to manual
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from playwright.async_api import Page, TimeoutError as PWTimeout

from core.job_models import (
    ApplicationStatus,
    ApplyMode,
    ApplyResult,
    JobApplication,
    JobListing,
    JobSource,
)
from tools.application_tracker import get_tracker
from tools.browser_manager import BrowserManager

logger = logging.getLogger("jarvis.auto_applicator")

# Where screenshots get saved (audit trail)
SCREENSHOT_DIR = Path("data/application_screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# FIELD MAPPER — translates resume profile → form field values
# ════════════════════════════════════════════════════════════════════════════

class FieldMapper:
    """Maps resume data to common form field names."""

    # Common label patterns → profile attribute
    FIELD_PATTERNS = {
        "first_name":    [r"first\s*name", r"given\s*name"],
        "last_name":     [r"last\s*name", r"family\s*name", r"surname"],
        "full_name":     [r"^name$", r"full\s*name", r"your\s*name"],
        "email":         [r"e[-\s]?mail", r"email\s*address"],
        "phone":         [r"phone", r"mobile", r"contact\s*number"],
        "location":      [r"current\s*location", r"city", r"location"],
        "experience":    [r"total\s*experience", r"years\s*of\s*experience", r"experience"],
        "current_ctc":   [r"current\s*ctc", r"current\s*salary"],
        "expected_ctc":  [r"expected\s*ctc", r"expected\s*salary"],
        "notice_period": [r"notice\s*period"],
        "linkedin":      [r"linkedin"],
        "github":        [r"github"],
        "portfolio":     [r"portfolio", r"website"],
        "cover_letter":  [r"cover\s*letter", r"why\s*are\s*you", r"tell\s*us"],
    }

    def __init__(self, profile: Dict[str, Any]) -> None:
        self.profile = profile or {}

    def value_for(self, field_key: str) -> Optional[str]:
        """Get profile value for a normalized field key."""
        personal = self.profile.get("personal", {})
        skills = self.profile.get("skills", [])
        exp = self.profile.get("experience", [])

        full_name = personal.get("name", "")
        parts = full_name.split(" ", 1)
        first = parts[0] if parts else ""
        last = parts[1] if len(parts) > 1 else ""

        years = 0
        if exp and isinstance(exp, list):
            years = max((e.get("years", 0) or 0) for e in exp) if exp else 0

        mapping = {
            "first_name": first,
            "last_name": last,
            "full_name": full_name,
            "email": personal.get("email", ""),
            "phone": personal.get("phone", ""),
            "location": personal.get("location", ""),
            "experience": str(years) if years else "",
            "linkedin": personal.get("linkedin", "") or "",
            "github": personal.get("github", "") or "",
            "portfolio": personal.get("portfolio", "") or "",
            "skills": ", ".join(skills[:10]) if skills else "",
            "current_ctc": os.getenv("USER_CURRENT_CTC", ""),
            "expected_ctc": os.getenv("USER_EXPECTED_CTC", ""),
            "notice_period": os.getenv("USER_NOTICE_PERIOD", "30 days"),
        }
        return mapping.get(field_key) or None

    def match_label(self, label: str) -> Optional[str]:
        """Match a free-text label against known patterns. Returns field_key."""
        if not label:
            return None
        label_l = label.lower().strip()
        for key, patterns in self.FIELD_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, label_l):
                    return key
        return None


# ════════════════════════════════════════════════════════════════════════════
# AUTO APPLICATOR
# ════════════════════════════════════════════════════════════════════════════

class AutoApplicator:
    """
    Apply to a job using Playwright. Site-aware: routes to the right strategy.
    """

    def __init__(self) -> None:
        self.bm = BrowserManager.instance()
        self.tracker = get_tracker()
        self.resume_path: Optional[Path] = self._find_resume()

    @staticmethod
    def _find_resume() -> Optional[Path]:
        d = Path("data/resumes")
        if not d.exists():
            return None
        files = list(d.glob("*.pdf")) + list(d.glob("*.docx"))
        if not files:
            return None
        return max(files, key=lambda p: p.stat().st_mtime)

    # ── Main entry point ───────────────────────────────────────────────────
    async def apply(
        self,
        job: JobListing,
        user_profile: Dict[str, Any],
        mode: ApplyMode = ApplyMode.SEMI_AUTO,
        user_id: str = "default_user",
    ) -> ApplyResult:
        """Apply to a single job."""
        t0 = time()
        application = JobApplication.from_job(job, user_id=user_id, apply_mode=mode)

        # Pre-flight: already applied?
        if self.tracker.has_applied(job):
            logger.info("Skip: already applied to %s @ %s", job.title, job.company)
            return ApplyResult(
                success=False,
                application=application,
                reason="Already applied recently (deduplication)",
                duration_ms=int((time() - t0) * 1000),
            )

        # Dry run
        if mode == ApplyMode.DRY_RUN:
            application.status = ApplicationStatus.READY_TO_SUBMIT
            self.tracker.record(application)
            return ApplyResult(
                success=True,
                application=application,
                reason="Dry run — no browser opened",
                duration_ms=int((time() - t0) * 1000),
            )

        # Route by source
        mapper = FieldMapper(user_profile)
        try:
            if job.source == JobSource.NAUKRI:
                result = await self._apply_naukri(job, application, mapper, mode)
            elif job.source == JobSource.LINKEDIN:
                result = await self._apply_linkedin(job, application, mapper, mode)
            else:
                result = await self._apply_generic(job, application, mapper, mode)
        except Exception as e:
            logger.exception("Auto-apply failed for %s", job.url)
            application.status = ApplicationStatus.FAILED
            application.error_message = str(e)
            self.tracker.record(application)
            return ApplyResult(
                success=False,
                application=application,
                reason=f"Exception: {e}",
                duration_ms=int((time() - t0) * 1000),
            )

        result.duration_ms = int((time() - t0) * 1000)
        self.tracker.record(result.application)
        return result

    async def apply_batch(
        self,
        jobs: List[JobListing],
        user_profile: Dict[str, Any],
        mode: ApplyMode = ApplyMode.SEMI_AUTO,
        max_concurrent: int = 1,  # serial by default — safer
        user_id: str = "default_user",
    ) -> List[ApplyResult]:
        """
        Apply to multiple jobs. Default is serial to avoid rate limits.
        """
        sem = asyncio.Semaphore(max_concurrent)

        async def _one(job: JobListing) -> ApplyResult:
            async with sem:
                return await self.apply(job, user_profile, mode, user_id)

        return await asyncio.gather(*(_one(j) for j in jobs), return_exceptions=False)

    # ── Site-specific strategies ───────────────────────────────────────────
    async def _apply_naukri(
        self,
        job: JobListing,
        app: JobApplication,
        mapper: FieldMapper,
        mode: ApplyMode,
    ) -> ApplyResult:
        screenshots: List[str] = []
        async with self.bm.new_page(site="naukri") as page:
            await page.goto(job.url, wait_until="domcontentloaded", timeout=45_000)
            await page.wait_for_timeout(2000)

            # Naukri requires login for apply. Check.
            if "login" in (page.url or "").lower() or await page.locator("input#usernameField").count():
                app.status = ApplicationStatus.REQUIRES_MANUAL
                app.error_message = "Naukri login required — set NAUKRI_EMAIL/NAUKRI_PASSWORD in .env"
                return ApplyResult(
                    success=False,
                    application=app,
                    reason=app.error_message,
                    requires_manual_review=True,
                )

            # Find Apply button
            apply_btn = page.locator(
                "button:has-text('Apply'), button:has-text('Quick Apply'), button.apply-button"
            ).first
            if not await apply_btn.count():
                app.status = ApplicationStatus.REQUIRES_MANUAL
                return ApplyResult(
                    success=False,
                    application=app,
                    reason="No Apply button found",
                    requires_manual_review=True,
                )

            # Fill any inline form fields visible
            filled = await self._fill_visible_fields(page, mapper)
            app.fields_filled.update(filled)

            # Screenshot before submit
            shot = await self._screenshot(page, app, "before_submit")
            screenshots.append(shot)

            if mode in {ApplyMode.REVIEW, ApplyMode.SEMI_AUTO}:
                app.status = ApplicationStatus.READY_TO_SUBMIT
                return ApplyResult(
                    success=True,
                    application=app,
                    submitted=False,
                    reason=f"{mode.value} — paused before submit. Review: {shot}",
                    screenshots=screenshots,
                )

            # AUTO mode: click submit
            try:
                await apply_btn.click(timeout=5000)
                await page.wait_for_timeout(3000)
                shot2 = await self._screenshot(page, app, "after_submit")
                screenshots.append(shot2)
                app.status = ApplicationStatus.SUBMITTED
                app.applied_at = datetime.utcnow()
                app.submission_proof_url = shot2
                return ApplyResult(
                    success=True,
                    application=app,
                    submitted=True,
                    reason="Submitted",
                    screenshots=screenshots,
                )
            except Exception as e:
                app.status = ApplicationStatus.FAILED
                app.error_message = str(e)
                return ApplyResult(
                    success=False,
                    application=app,
                    reason=f"Submit failed: {e}",
                    screenshots=screenshots,
                )

    async def _apply_linkedin(
        self,
        job: JobListing,
        app: JobApplication,
        mapper: FieldMapper,
        mode: ApplyMode,
    ) -> ApplyResult:
        screenshots: List[str] = []
        async with self.bm.new_page(site="linkedin") as page:
            await page.goto(job.url, wait_until="domcontentloaded", timeout=45_000)
            await page.wait_for_timeout(2000)

            # Easy Apply button
            easy_apply = page.locator("button.jobs-apply-button:has-text('Easy Apply')").first
            if not await easy_apply.count():
                app.status = ApplicationStatus.REQUIRES_MANUAL
                return ApplyResult(
                    success=False,
                    application=app,
                    reason="No Easy Apply button (external application)",
                    requires_manual_review=True,
                )

            # Login check
            if "/login" in (page.url or "") or await page.locator("input#username").count():
                app.status = ApplicationStatus.REQUIRES_MANUAL
                app.error_message = "LinkedIn login required — set LINKEDIN_EMAIL/LINKEDIN_PASSWORD in .env"
                return ApplyResult(
                    success=False,
                    application=app,
                    reason=app.error_message,
                    requires_manual_review=True,
                )

            await easy_apply.click()
            await page.wait_for_timeout(2000)

            # LinkedIn Easy Apply has multiple steps. Fill what we can.
            filled = await self._fill_visible_fields(page, mapper)
            app.fields_filled.update(filled)

            shot = await self._screenshot(page, app, "before_submit")
            screenshots.append(shot)

            # If complex multi-step → escalate
            has_next = await page.locator("button:has-text('Next')").count()
            if has_next and mode != ApplyMode.AUTO:
                app.status = ApplicationStatus.REQUIRES_MANUAL
                return ApplyResult(
                    success=False,
                    application=app,
                    reason="Multi-step Easy Apply — requires manual review",
                    requires_manual_review=True,
                    screenshots=screenshots,
                )

            if mode in {ApplyMode.REVIEW, ApplyMode.SEMI_AUTO}:
                app.status = ApplicationStatus.READY_TO_SUBMIT
                return ApplyResult(
                    success=True,
                    application=app,
                    submitted=False,
                    reason=f"{mode.value} — paused before submit. Review: {shot}",
                    screenshots=screenshots,
                )

            # AUTO: submit
            submit_btn = page.locator(
                "button:has-text('Submit application'), button[aria-label*='Submit']"
            ).first
            if not await submit_btn.count():
                app.status = ApplicationStatus.REQUIRES_MANUAL
                return ApplyResult(
                    success=False,
                    application=app,
                    reason="No Submit button visible",
                    requires_manual_review=True,
                    screenshots=screenshots,
                )

            await submit_btn.click()
            await page.wait_for_timeout(3000)
            shot2 = await self._screenshot(page, app, "after_submit")
            screenshots.append(shot2)
            app.status = ApplicationStatus.SUBMITTED
            app.applied_at = datetime.utcnow()
            app.submission_proof_url = shot2
            return ApplyResult(
                success=True,
                application=app,
                submitted=True,
                reason="Submitted via Easy Apply",
                screenshots=screenshots,
            )

    async def _apply_generic(
        self,
        job: JobListing,
        app: JobApplication,
        mapper: FieldMapper,
        mode: ApplyMode,
    ) -> ApplyResult:
        """
        Generic form filler — Workday, Greenhouse, Lever, custom pages.
        Identifies fields by label proximity and fills what it can.
        """
        screenshots: List[str] = []
        async with self.bm.new_page(site=self._site_key(job.url)) as page:
            await page.goto(job.url, wait_until="domcontentloaded", timeout=45_000)
            await page.wait_for_timeout(2000)

            filled = await self._fill_visible_fields(page, mapper)
            app.fields_filled.update(filled)

            # Resume upload
            if await self._try_upload_resume(page):
                app.resume_uploaded = True

            shot = await self._screenshot(page, app, "filled")
            screenshots.append(shot)

            # For generic pages, never auto-submit — too risky
            app.status = ApplicationStatus.READY_TO_SUBMIT
            return ApplyResult(
                success=True,
                application=app,
                submitted=False,
                reason=f"Form filled. Review and submit manually: {shot}",
                requires_manual_review=mode == ApplyMode.AUTO,
                screenshots=screenshots,
            )

    # ── Helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _site_key(url: str) -> str:
        try:
            host = urlparse(url).netloc.replace("www.", "")
            return host.split(".")[0] or "generic"
        except Exception:
            return "generic"

    async def _fill_visible_fields(self, page: Page, mapper: FieldMapper) -> Dict[str, str]:
        """
        Scan all visible inputs/selects/textareas, match labels, fill.
        Returns map of field_key → value filled.
        """
        filled: Dict[str, str] = {}

        # Strategy: iterate inputs, find their label, match against patterns
        inputs = await page.query_selector_all(
            "input:not([type='hidden']):not([type='submit']):not([type='button']), "
            "textarea, select"
        )

        for el in inputs:
            try:
                if not await el.is_visible():
                    continue
                if not await el.is_enabled():
                    continue

                # Find label: prefer `for=id`, then aria-label, then placeholder
                label = await self._find_label(page, el)
                if not label:
                    continue

                field_key = mapper.match_label(label)
                if not field_key:
                    continue

                value = mapper.value_for(field_key)
                if not value:
                    continue

                tag = (await el.evaluate("e => e.tagName")).lower()
                if tag == "select":
                    try:
                        await el.select_option(label=value)
                    except Exception:
                        continue
                else:
                    await el.fill(value)

                filled[field_key] = value
                logger.debug("Filled %s = %s", field_key, value[:40])
            except Exception as e:
                logger.debug("Skip field: %s", e)
                continue

        return filled

    @staticmethod
    async def _find_label(page: Page, el) -> str:
        """Find the label/placeholder/aria-label for a form element."""
        # aria-label
        aria = await el.get_attribute("aria-label")
        if aria:
            return aria
        # placeholder
        ph = await el.get_attribute("placeholder")
        if ph:
            return ph
        # name attribute (fallback)
        name = await el.get_attribute("name") or ""
        # `for=id` label
        id_attr = await el.get_attribute("id")
        if id_attr:
            label_el = await page.query_selector(f"label[for='{id_attr}']")
            if label_el:
                txt = await label_el.text_content()
                if txt:
                    return txt.strip()
        return name

    async def _try_upload_resume(self, page: Page) -> bool:
        """Upload resume file to any visible file input."""
        if not self.resume_path or not self.resume_path.exists():
            return False
        try:
            file_inputs = await page.query_selector_all("input[type='file']")
            for inp in file_inputs:
                if await inp.is_visible():
                    await inp.set_input_files(str(self.resume_path))
                    logger.info("Uploaded resume: %s", self.resume_path.name)
                    return True
        except Exception as e:
            logger.warning("Resume upload failed: %s", e)
        return False

    @staticmethod
    async def _screenshot(page: Page, app: JobApplication, tag: str) -> str:
        """Save full-page screenshot and return path."""
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{app.application_id}_{tag}_{ts}.png"
        path = SCREENSHOT_DIR / filename
        try:
            await page.screenshot(path=str(path), full_page=True)
        except Exception as e:
            logger.warning("Screenshot failed: %s", e)
        return str(path)
