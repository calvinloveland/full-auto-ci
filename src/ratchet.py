"""Ratchet evaluation for tool outputs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .config import Config
from .db import DataAccess

logger = logging.getLogger(__name__)


@dataclass
class RatchetRule:
    """Describes how to evaluate a tool's ratchet."""

    metric: str
    direction: str
    target: float
    tolerance: float = 0.0


@dataclass
class RatchetEvaluation:
    """Result of evaluating a ratchet rule."""

    success: bool
    status: str
    previous_best: Optional[float]
    message: Optional[str] = None


class RatchetManager:
    """Coordinates ratchet enforcement for tool results."""

    DEFAULT_RULES: Dict[str, RatchetRule] = {
        "coverage": RatchetRule("percentage", "higher", 90.0, 0.0),
        "pylint": RatchetRule("score", "higher", 10.0, 0.0),
        "lizard": RatchetRule("summary.above_threshold", "lower", 0.0, 0.0),
    }

    def __init__(self, data: DataAccess, config: Config):
        self._data = data
        self._config = config

    def apply(self, repo_id: int, results: Dict[str, Dict[str, Any]]) -> None:
        """Mutate ``results`` with ratchet enforcement for each tool."""

        for tool_name, payload in list(results.items()):
            if not isinstance(payload, dict):
                continue
            if payload.get("status") != "success":
                continue

            rule = self._build_rule(tool_name)
            if rule is None:
                continue

            metric_value = self._extract_metric(payload, rule.metric)
            numeric_value = self._coerce_float(metric_value)
            if numeric_value is None:
                message = f"Ratchet metric '{rule.metric}' missing or non-numeric in {tool_name} result"
                payload["status"] = "error"
                payload["error"] = message
                logger.warning(message)
                continue

            evaluation = self._evaluate(repo_id, tool_name, numeric_value, rule)

            payload.setdefault("ratchet", {})
            payload["ratchet"].update(
                {
                    "metric": rule.metric,
                    "value": numeric_value,
                    "target": rule.target,
                    "direction": rule.direction,
                    "status": evaluation.status,
                    "previous_best": evaluation.previous_best,
                }
            )
            if evaluation.message:
                payload["ratchet"]["message"] = evaluation.message

            if not evaluation.success:
                payload["status"] = "error"
                payload["error"] = evaluation.message or (
                    f"Ratchet requirements for {tool_name} were not met"
                )

    # ------------------------------------------------------------------
    # Rule resolution
    # ------------------------------------------------------------------
    def _build_rule(self, tool_name: str) -> Optional[RatchetRule]:
        ratchet_config = self._ratchet_config(tool_name)
        if not self._coerce_bool(ratchet_config.get("enabled")):
            return None

        defaults = self.DEFAULT_RULES.get(
            tool_name, RatchetRule("value", "higher", 0.0, 0.0)
        )

        metric = ratchet_config.get("metric") or defaults.metric
        direction = self._ratchet_direction(tool_name, ratchet_config, defaults)

        target_value = self._ratchet_target(tool_name, ratchet_config, defaults)
        if target_value is None:
            return None

        tolerance_value = self._ratchet_tolerance(ratchet_config, defaults)

        return RatchetRule(
            metric=str(metric),
            direction=direction,
            target=target_value,
            tolerance=tolerance_value,
        )

    def _ratchet_config(self, tool_name: str) -> Dict[str, Any]:
        tools_config = self._config.get("tools") or {}
        tool_config = (
            tools_config.get(tool_name, {}) if isinstance(tools_config, dict) else {}
        )
        ratchet_config = (
            tool_config.get("ratchet") if isinstance(tool_config, dict) else None
        )
        return ratchet_config if isinstance(ratchet_config, dict) else {}

    def _ratchet_direction(
        self,
        tool_name: str,
        ratchet_config: Dict[str, Any],
        defaults: RatchetRule,
    ) -> str:
        direction_raw = (
            ratchet_config.get("direction") or defaults.direction or "higher"
        ).lower()
        if direction_raw in {"higher", "lower"}:
            return direction_raw

        logger.warning(
            "Invalid ratchet direction '%s' for %s; defaulting to %s",
            direction_raw,
            tool_name,
            defaults.direction,
        )
        return defaults.direction

    def _ratchet_target(
        self,
        tool_name: str,
        ratchet_config: Dict[str, Any],
        defaults: RatchetRule,
    ) -> Optional[float]:
        target = ratchet_config.get("target", defaults.target)
        target_value = self._coerce_float(target)
        if target_value is None:
            logger.warning(
                "Ratchet target missing or invalid for %s; skipping ratchet enforcement",
                tool_name,
            )
            return None
        return target_value

    def _ratchet_tolerance(
        self,
        ratchet_config: Dict[str, Any],
        defaults: RatchetRule,
    ) -> float:
        tolerance_value = self._coerce_float(
            ratchet_config.get("tolerance", defaults.tolerance)
        )
        return tolerance_value if tolerance_value is not None else defaults.tolerance

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def _evaluate(
        self,
        repo_id: int,
        tool_name: str,
        current_value: float,
        rule: RatchetRule,
    ) -> RatchetEvaluation:
        previous_best = self._best_historical_value(repo_id, tool_name, rule)

        if self._meets_target(current_value, rule):
            message = f"{tool_name} ratchet target achieved: {rule.metric} {self._format_direction(rule)} {self._format_value(rule.target)}"
            return RatchetEvaluation(True, "target-met", previous_best, message)

        if previous_best is None:
            message = f"{tool_name} ratchet baseline set at {self._format_value(current_value)}; target {self._format_value(rule.target)}"
            return RatchetEvaluation(True, "baseline", previous_best, message)

        if self._regressed(current_value, previous_best, rule):
            if self._meets_target(previous_best, rule):
                message = (
                    f"{tool_name} ratchet requires {rule.metric} {self._format_direction(rule)} {self._format_value(rule.target)} "
                    f"(previous best {self._format_value(previous_best)}); observed {self._format_value(current_value)}"
                )
            else:
                comparison = "≥" if rule.direction == "higher" else "≤"
                message = (
                    f"{tool_name} ratchet requires {rule.metric} {comparison} {self._format_value(previous_best)} "
                    f"(best so far); observed {self._format_value(current_value)}"
                )
            return RatchetEvaluation(False, "regression", previous_best, message)

        if self._is_better(current_value, previous_best, rule):
            message = (
                f"{tool_name} ratchet improved {rule.metric} from {self._format_value(previous_best)} "
                f"to {self._format_value(current_value)}; target {self._format_value(rule.target)}"
            )
            return RatchetEvaluation(True, "progress", previous_best, message)

        message = f"{tool_name} ratchet holding at {self._format_value(current_value)}; target {self._format_value(rule.target)}"
        return RatchetEvaluation(True, "holding", previous_best, message)

    def _best_historical_value(
        self, repo_id: int, tool_name: str, rule: RatchetRule
    ) -> Optional[float]:
        history = self._data.fetch_tool_history(repo_id, tool_name)
        best: Optional[float] = None
        for entry in history:
            if entry.get("status") != "success":
                continue
            data = entry.get("data")
            if not isinstance(data, dict):
                continue
            metric_value = self._extract_metric(data, rule.metric)
            numeric_value = self._coerce_float(metric_value)
            if numeric_value is None:
                continue
            if best is None:
                best = numeric_value
                continue
            if rule.direction == "higher" and numeric_value > best:
                best = numeric_value
            elif rule.direction == "lower" and numeric_value < best:
                best = numeric_value
        return best

    @staticmethod
    def _extract_metric(payload: Any, path: str) -> Optional[Any]:
        if not path:
            return None
        parts = path.split(".")
        current: Any = payload
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip():
            try:
                return float(value.strip())
            except ValueError:
                return None
        return None

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return False

    @staticmethod
    def _format_value(value: float) -> str:
        return f"{value:.2f}"

    @staticmethod
    def _format_direction(rule: RatchetRule) -> str:
        return "≥" if rule.direction == "higher" else "≤"

    def _meets_target(self, value: float, rule: RatchetRule) -> bool:
        tolerance = max(rule.tolerance, 0.0)
        if rule.direction == "higher":
            return value >= rule.target - tolerance
        return value <= rule.target + tolerance

    def _regressed(self, value: float, previous: float, rule: RatchetRule) -> bool:
        tolerance = max(rule.tolerance, 0.0)
        if rule.direction == "higher":
            return value < previous - tolerance
        return value > previous + tolerance

    def _is_better(self, value: float, previous: float, rule: RatchetRule) -> bool:
        tolerance = max(rule.tolerance, 0.0)
        if rule.direction == "higher":
            return value > previous + tolerance
        return value < previous - tolerance
