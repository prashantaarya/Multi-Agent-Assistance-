"""
Application tracker — persistent storage + deduplication.

Stores: data/applications.json
Index : data/applications_index.json (fast dedup lookup)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Set

from core.job_models import (
    ApplicationStatus,
    JobApplication,
    JobListing,
)

logger = logging.getLogger("jarvis.application_tracker")


class ApplicationTracker:
    """
    Tracks all job applications with O(1) dedup lookup.

    Public API:
      - has_applied(job)         → bool
      - record(application)      → None
      - get_application(job_id)  → Optional[JobApplication]
      - list_applications(...)   → List[JobApplication]
      - update_status(...)       → None
      - stats()                  → Dict
    """

    def __init__(self, data_dir: Path = Path("data")) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.apps_file = self.data_dir / "applications.json"
        self.index_file = self.data_dir / "applications_index.json"
        self._cache: Optional[List[JobApplication]] = None
        self._index: Optional[Dict[str, str]] = None  # dedup_key → application_id

    # ── Persistence ────────────────────────────────────────────────────────
    def _load(self) -> List[JobApplication]:
        if self._cache is not None:
            return self._cache
        if not self.apps_file.exists():
            self._cache = []
            self._index = {}
            return self._cache
        try:
            raw = json.loads(self.apps_file.read_text(encoding="utf-8"))
            self._cache = [JobApplication.model_validate(a) for a in raw]
        except Exception as e:
            logger.error("Failed to load applications: %s", e)
            self._cache = []
        self._rebuild_index()
        return self._cache

    def _rebuild_index(self) -> None:
        self._index = {}
        for app in self._cache or []:
            key = self._dedup_key(app.company, app.title)
            self._index[key] = app.application_id

    def _persist(self) -> None:
        apps = self._cache or []
        try:
            self.apps_file.write_text(
                json.dumps(
                    [a.model_dump(mode="json") for a in apps],
                    indent=2,
                    default=str,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            self.index_file.write_text(
                json.dumps(self._index or {}, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.error("Failed to persist applications: %s", e)

    @staticmethod
    def _dedup_key(company: str, title: str) -> str:
        return f"{company.lower().strip()}|{title.lower().strip()}"

    # ── Public API ─────────────────────────────────────────────────────────
    def has_applied(self, job: JobListing, within_days: int = 90) -> bool:
        """
        True if user already applied to this company+title within `within_days`.
        Prevents spamming the same company.
        """
        apps = self._load()
        key = self._dedup_key(job.company, job.title)
        if key not in (self._index or {}):
            return False

        app_id = self._index[key]
        app = next((a for a in apps if a.application_id == app_id), None)
        if not app or not app.applied_at:
            return False

        if app.status in {ApplicationStatus.FAILED, ApplicationStatus.WITHDRAWN}:
            return False

        delta = datetime.utcnow() - app.applied_at.replace(tzinfo=None)
        return delta < timedelta(days=within_days)

    def record(self, application: JobApplication) -> JobApplication:
        """Insert or update an application."""
        apps = self._load()

        # Update existing
        for i, existing in enumerate(apps):
            if existing.application_id == application.application_id:
                application.last_updated = datetime.utcnow()
                apps[i] = application
                self._persist()
                logger.info("Updated application %s", application.application_id)
                return application

        # Insert new
        application.last_updated = datetime.utcnow()
        apps.append(application)
        key = self._dedup_key(application.company, application.title)
        if self._index is None:
            self._index = {}
        self._index[key] = application.application_id
        self._persist()
        logger.info(
            "Recorded application %s (%s @ %s)",
            application.application_id,
            application.title,
            application.company,
        )
        return application

    def get_application(self, application_id: str) -> Optional[JobApplication]:
        apps = self._load()
        return next((a for a in apps if a.application_id == application_id), None)

    def list_applications(
        self,
        user_id: Optional[str] = None,
        status: Optional[ApplicationStatus] = None,
        limit: int = 100,
    ) -> List[JobApplication]:
        apps = self._load()
        if user_id:
            apps = [a for a in apps if a.user_id == user_id]
        if status:
            apps = [a for a in apps if a.status == status]
        return sorted(apps, key=lambda a: a.last_updated, reverse=True)[:limit]

    def update_status(
        self,
        application_id: str,
        status: ApplicationStatus,
        **fields,
    ) -> Optional[JobApplication]:
        apps = self._load()
        for i, app in enumerate(apps):
            if app.application_id == application_id:
                app.status = status
                app.last_updated = datetime.utcnow()
                for k, v in fields.items():
                    if hasattr(app, k):
                        setattr(app, k, v)
                apps[i] = app
                self._persist()
                return app
        return None

    def stats(self, user_id: str = "default_user") -> Dict[str, int]:
        apps = [a for a in self._load() if a.user_id == user_id]
        counts: Dict[str, int] = {}
        for a in apps:
            counts[a.status.value] = counts.get(a.status.value, 0) + 1
        counts["total"] = len(apps)
        return counts

    def filter_new_jobs(self, jobs: List[JobListing], within_days: int = 90) -> List[JobListing]:
        """Return only jobs the user has NOT applied to recently."""
        return [j for j in jobs if not self.has_applied(j, within_days=within_days)]


# Singleton
_tracker: Optional[ApplicationTracker] = None


def get_tracker() -> ApplicationTracker:
    global _tracker
    if _tracker is None:
        _tracker = ApplicationTracker()
    return _tracker
