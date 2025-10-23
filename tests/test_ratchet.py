"""Tests for the ratchet manager."""

import copy
import unittest

from src.config import Config
from src.ratchet import RatchetManager


class DummyDataAccess:
    def __init__(self, history=None):
        self._history = history or {}

    def fetch_tool_history(self, repo_id, tool, limit=None):
        _ = limit
        return copy.deepcopy(self._history.get((repo_id, tool), []))


class TestRatchetManager(unittest.TestCase):
    def setUp(self):
        self.config = Config(config_path="/nonexistent")
        # prevent attempts to load real config
        self.config.config = copy.deepcopy(Config.DEFAULT_CONFIG)
        for tool in self.config.config.get("tools", {}).values():
            if isinstance(tool, dict) and "ratchet" in tool:
                tool["ratchet"]["enabled"] = True

    def test_coverage_progress_allows_success(self):
        history = {
            (1, "coverage"): [
                {
                    "status": "success",
                    "data": {"percentage": 78.0},
                    "output": "",
                    "duration": 1.0,
                    "created_at": None,
                }
            ]
        }
        data = DummyDataAccess(history)
        manager = RatchetManager(data, self.config)

        results = {"coverage": {"status": "success", "percentage": 80.0}}
        manager.apply(1, results)

        coverage = results["coverage"]
        self.assertEqual(coverage["status"], "success")
        self.assertEqual(coverage["ratchet"]["status"], "progress")
        self.assertGreater(
            coverage["ratchet"]["value"], coverage["ratchet"]["previous_best"]
        )

    def test_coverage_regression_fails(self):
        history = {
            (1, "coverage"): [
                {
                    "status": "success",
                    "data": {"percentage": 82.0},
                    "output": "",
                    "duration": 1.0,
                    "created_at": None,
                }
            ]
        }
        data = DummyDataAccess(history)
        manager = RatchetManager(data, self.config)

        results = {"coverage": {"status": "success", "percentage": 80.0}}
        manager.apply(1, results)

        coverage = results["coverage"]
        self.assertEqual(coverage["status"], "error")
        self.assertEqual(coverage["ratchet"]["status"], "regression")
        self.assertIn("requires", coverage["error"].lower())

    def test_coverage_target_enforced_after_reaching(self):
        history = {
            (1, "coverage"): [
                {
                    "status": "success",
                    "data": {"percentage": 92.0},
                    "output": "",
                    "duration": 1.0,
                    "created_at": None,
                }
            ]
        }
        data = DummyDataAccess(history)
        manager = RatchetManager(data, self.config)

        results = {"coverage": {"status": "success", "percentage": 85.0}}
        manager.apply(1, results)

        coverage = results["coverage"]
        self.assertEqual(coverage["status"], "error")
        self.assertEqual(coverage["ratchet"]["status"], "regression")
        self.assertIn("requires", coverage["error"].lower())

    def test_lizard_counts_must_not_increase(self):
        history = {
            (1, "lizard"): [
                {
                    "status": "success",
                    "data": {"summary": {"above_threshold": 5}},
                    "output": "",
                    "duration": 1.0,
                    "created_at": None,
                }
            ]
        }
        data = DummyDataAccess(history)
        manager = RatchetManager(data, self.config)

        results = {
            "lizard": {
                "status": "success",
                "summary": {"above_threshold": 4},
            }
        }
        manager.apply(1, results)

        lizard = results["lizard"]
        self.assertEqual(lizard["status"], "success")
        self.assertEqual(lizard["ratchet"]["status"], "progress")

    def test_baseline_allows_initial_failure(self):
        data = DummyDataAccess()
        manager = RatchetManager(data, self.config)

        results = {"coverage": {"status": "success", "percentage": 50.0}}
        manager.apply(1, results)

        coverage = results["coverage"]
        self.assertEqual(coverage["status"], "success")
        self.assertEqual(coverage["ratchet"]["status"], "baseline")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
