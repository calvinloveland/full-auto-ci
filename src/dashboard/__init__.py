"""Flask dashboard application for Full Auto CI."""

from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime
from typing import Any, Dict

from flask import (
    Blueprint,
    Flask,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)

from ..service import CIService

logger = logging.getLogger(__name__)


def _ensure_secret_key(service: CIService) -> str:
    explicit = service.config.get("dashboard", "secret_key")
    env_key = os.getenv("FULL_AUTO_CI_DASHBOARD_SECRET")

    if explicit:
        return str(explicit)
    if env_key:
        return env_key

    generated = secrets.token_hex(16)
    logger.warning("Dashboard secret key not configured; generated ephemeral key")
    return generated


def _timeago(value: int | None) -> str:
    if value is None:
        return "—"

    try:
        dt = datetime.fromtimestamp(int(value))
    except (TypeError, ValueError):
        return "—"

    seconds = int((datetime.now() - dt).total_seconds())
    if seconds < 60:
        result = "just now"
    else:
        periods = [
            ("minute", 60),
            ("hour", 3600),
            ("day", 86400),
            ("week", 604800),
            ("month", 2592000),
            ("year", 31536000),
        ]

        result = "—"
        for index, (name, duration) in enumerate(periods):
            next_boundary = periods[index + 1][1] if index + 1 < len(periods) else None
            if next_boundary is not None and seconds >= next_boundary:
                continue

            count = max(1, seconds // duration)
            plural = "s" if count != 1 else ""
            result = f"{count} {name}{plural} ago"
            break

    return result


def _status_class(status: str | None) -> str:
    mapping = {
        "pending": "status-pending",
        "queued": "status-queued",
        "running": "status-running",
        "completed": "status-success",
        "success": "status-success",
        "error": "status-error",
        "failed": "status-error",
    }
    return mapping.get((status or "").lower(), "status-unknown")


def _summarize_repo(repo: Dict[str, Any], data_access) -> Dict[str, Any]:
    repo_copy = dict(repo)
    summary = data_access.summarize_test_runs(repo["id"])
    recent = data_access.fetch_recent_test_runs(repo["id"], limit=1)
    latest = recent[0] if recent else None

    repo_copy.update(
        {
            "summary": summary,
            "latest_run": latest,
            "latest_status": (latest or {}).get("status"),
            "latest_started_at": (latest or {}).get("started_at"),
        }
    )
    return repo_copy


def _compute_overview_metrics(repositories: list[Dict[str, Any]]) -> Dict[str, Any]:
    passing_statuses = {"completed", "success"}
    failing_statuses = {"error", "failed"}
    queued_statuses = {"pending", "queued"}
    running_statuses = {"running"}

    metrics = {
        "total": len(repositories),
        "passing": 0,
        "failing": 0,
        "running": 0,
        "queued": 0,
        "never": 0,
    }

    for repo in repositories:
        status = (repo.get("latest_status") or "").lower()
        if not status:
            metrics["never"] += 1
            continue
        if status in passing_statuses:
            metrics["passing"] += 1
        elif status in failing_statuses:
            metrics["failing"] += 1
        elif status in running_statuses:
            metrics["running"] += 1
        elif status in queued_statuses:
            metrics["queued"] += 1
        else:
            metrics["never"] += 1

    return metrics


def _hydrate_test_runs(data_access, runs):
    hydrated = []
    for run in runs:
        commit = data_access.fetch_commit_for_test_run(run["id"])
        results = data_access.fetch_results_for_test_run(run["id"])
        hydrated.append(
            {
                **run,
                "commit": commit,
                "results": results,
            }
        )
    return hydrated


def _build_repository_insights(service: CIService, data_access, repo_id: int):
    repository = service.get_repository(repo_id)
    if not repository:
        return None

    runs = data_access.fetch_recent_test_runs(repo_id, limit=20)
    hydrated = _hydrate_test_runs(data_access, runs)
    summary = data_access.summarize_test_runs(repo_id)

    return {
        "repository": repository,
        "test_runs": hydrated,
        "summary": summary,
        "last_run": hydrated[0] if hydrated else None,
    }


def create_app(config_path: str | None = None, db_path: str | None = None) -> Flask:
    """Create and configure the dashboard Flask application."""

    service = CIService(config_path=config_path, db_path=db_path)
    app = Flask(__name__, template_folder="templates", static_folder="static")

    secret_key = _ensure_secret_key(service)
    app.config["SECRET_KEY"] = secret_key
    app.config["CI_SERVICE"] = service
    app.config["DATA_ACCESS"] = service.data

    dashboard_bp = Blueprint(
        "dashboard",
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    @dashboard_bp.route("/")
    def home():
        service = current_app.config["CI_SERVICE"]
        data_access = current_app.config["DATA_ACCESS"]
        repositories = service.list_repositories()
        enriched = [_summarize_repo(repo, data_access) for repo in repositories]
        overview = _compute_overview_metrics(enriched)
        return render_template("index.html", repositories=enriched, overview=overview)

    @dashboard_bp.route("/partials/repositories")
    def repositories_partial():
        service = current_app.config["CI_SERVICE"]
        data_access = current_app.config["DATA_ACCESS"]
        repositories = service.list_repositories()
        enriched = [_summarize_repo(repo, data_access) for repo in repositories]
        overview = _compute_overview_metrics(enriched)
        return render_template(
            "partials/repositories_overview.html",
            repositories=enriched,
            overview=overview,
        )

    @dashboard_bp.route("/repo/<int:repo_id>")
    def repository_detail(repo_id: int):
        service = current_app.config["CI_SERVICE"]
        data_access = current_app.config["DATA_ACCESS"]
        insights = _build_repository_insights(service, data_access, repo_id)
        if not insights:
            abort(404)

        return render_template(
            "repository_detail.html",
            **insights,
        )

    @dashboard_bp.route("/repo/<int:repo_id>/insights")
    def repository_insights(repo_id: int):
        service = current_app.config["CI_SERVICE"]
        data_access = current_app.config["DATA_ACCESS"]
        insights = _build_repository_insights(service, data_access, repo_id)
        if not insights:
            abort(404)

        return render_template("partials/repository_insights.html", **insights)

    @dashboard_bp.post("/repo/<int:repo_id>/rerun")
    def rerun_test(repo_id: int):
        service = current_app.config["CI_SERVICE"]
        commit_hash = request.form.get("commit_hash")
        if not commit_hash:
            flash("Commit hash is required", "error")
            return redirect(url_for("dashboard.repository_detail", repo_id=repo_id))

        if service.add_test_task(repo_id, commit_hash):
            flash(f"Queued tests for commit {commit_hash[:7]}", "success")
        else:
            flash("Failed to queue test run. Check logs for details.", "error")
        return redirect(url_for("dashboard.repository_detail", repo_id=repo_id))

    app.register_blueprint(dashboard_bp)
    app.add_template_filter(_timeago, name="timeago")
    app.add_template_filter(_status_class, name="status_class")
    return app


__all__ = ["create_app"]
