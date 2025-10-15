"""Tools for code analysis and testing."""

import json
import logging
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Tool:  # pylint: disable=too-few-public-methods
    """Base class for all tools."""

    def __init__(self, name: str):
        """Initialize a tool.

        Args:
            name: Tool name
        """
        self.name = name

    def run(self, repo_path: str) -> Dict[str, Any]:
        """Run the tool.

        Args:
            repo_path: Path to the repository

        Returns:
            Tool results
        """
        raise NotImplementedError("Subclasses must implement this method")


class Pylint(Tool):  # pylint: disable=too-few-public-methods
    """Pylint code analysis tool."""

    def __init__(self):
        """Initialize Pylint."""
        super().__init__("pylint")

    def run(self, repo_path: str) -> Dict[str, Any]:
        """Run Pylint.

        Args:
            repo_path: Path to the repository

        Returns:
            Pylint results
        """
        try:
            targets = self._discover_targets(repo_path)
            logger.info(
                "Running Pylint on %s (targets: %s)", repo_path, ", ".join(targets)
            )

            cmd = ["pylint", "--output-format=json", *targets]
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                cwd=repo_path,
            )

            # Parse Pylint output
            if process.returncode >= 0 and process.stdout:
                try:
                    results = json.loads(process.stdout)

                    # Count issues by type
                    issues_by_type = {}
                    for item in results:
                        issue_type = item.get("type", "unknown")
                        issues_by_type[issue_type] = (
                            issues_by_type.get(issue_type, 0) + 1
                        )

                    # Calculate score
                    # Pylint score is from 0 to 10, with 10 being perfect
                    # A rough estimation if we don't have the exact score
                    score = 10.0
                    for issue_type, count in issues_by_type.items():
                        if issue_type == "error":
                            score -= 0.5 * count
                        elif issue_type == "warning":
                            score -= 0.2 * count
                        elif issue_type == "convention":
                            score -= 0.1 * count

                    score = max(0.0, score)

                    return {
                        "status": "success",
                        "score": score,
                        "issues": issues_by_type,
                        "details": results,
                    }
                except json.JSONDecodeError:
                    logger.error("Failed to parse Pylint JSON output")
                    return {"status": "error", "error": "Failed to parse Pylint output"}
            else:
                logger.error("Pylint failed with return code %s", process.returncode)
                return {
                    "status": "error",
                    "error": f"Pylint failed with return code {process.returncode}",
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                }
        except Exception as error:  # pylint: disable=broad-except
            logger.exception("Error running Pylint")
            return {"status": "error", "error": str(error)}

    def _discover_targets(self, repo_path: str) -> List[str]:
        """Determine which paths Pylint should analyze for the given repository."""

        if self._has_explicit_config(repo_path):
            return ["."]

        targets: List[str] = []

        src_dir = os.path.join(repo_path, "src")
        if os.path.isdir(src_dir):
            packages = self._package_directories(src_dir)
            if packages:
                targets.extend([os.path.join("src", pkg) for pkg in packages])
            else:
                targets.append("src")

        if not targets:
            root_packages = self._package_directories(repo_path)
            if root_packages:
                targets.extend(root_packages)

        if not targets:
            return ["."]

        seen = set()
        unique_targets: List[str] = []
        for target in targets:
            if target not in seen:
                seen.add(target)
                unique_targets.append(target)

        return unique_targets

    def _package_directories(self, base_path: str) -> List[str]:
        """Return Python package directories directly under ``base_path``."""

        try:
            entries = sorted(os.listdir(base_path))
        except OSError:
            return []

        packages: List[str] = []
        for entry in entries:
            if entry.startswith("."):
                continue
            full_path = os.path.join(base_path, entry)
            if not os.path.isdir(full_path):
                continue
            init_file = os.path.join(full_path, "__init__.py")
            if os.path.isfile(init_file):
                packages.append(entry)
        return packages

    def _has_explicit_config(self, repo_path: str) -> bool:
        """Return True if the repository supplies its own Pylint configuration."""

        config_files = [".pylintrc", "pylintrc"]
        for config in config_files:
            if os.path.isfile(os.path.join(repo_path, config)):
                return True

        setup_cfg = os.path.join(repo_path, "setup.cfg")
        if self._file_contains(setup_cfg, "[pylint]"):
            return True

        pyproject = os.path.join(repo_path, "pyproject.toml")
        if self._file_contains(pyproject, "[tool.pylint"):
            return True

        return False

    def _file_contains(self, path: str, needle: str) -> bool:
        if not os.path.isfile(path):
            return False

        try:
            with open(path, "r", encoding="utf-8") as handle:
                return needle in handle.read()
        except OSError:
            return False


class Coverage(Tool):  # pylint: disable=too-few-public-methods
    """Coverage measurement tool."""

    def __init__(self, run_tests_cmd: Optional[List[str]] = None):
        """Initialize Coverage.

        Args:
            run_tests_cmd: Command to run tests (defaults to pytest)
        """
        super().__init__("coverage")
        self.run_tests_cmd = run_tests_cmd or ["pytest"]

    ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

    @dataclass
    class _RunContext:
        returncode: int
        duration: float
        stdout: Optional[str]
        stderr: Optional[str]
        pytest_details: Optional[Dict[str, Any]]
        pytest_summary: Optional[Dict[str, Any]]
        embedded_results: List[Dict[str, Any]]

    @dataclass
    class _XmlContext:
        returncode: int
        duration: float
        stdout: Optional[str]
        stderr: Optional[str]

    @dataclass
    class _PytestSummary:
        status: str
        counts: List[Dict[str, Any]]
        duration: Optional[float]

    def run(self, repo_path: str) -> Dict[str, Any]:
        """Run coverage.

        Args:
            repo_path: Path to the repository

        Returns:
            Coverage results
        """
        original_dir = os.getcwd()
        try:
            os.chdir(repo_path)
            return self._run_inside_repository(repo_path)
        except Exception as error:  # pylint: disable=broad-except
            logger.exception("Error running Coverage")
            return {"status": "error", "error": str(error)}
        finally:
            os.chdir(original_dir)

    def _run_inside_repository(self, repo_path: str) -> Dict[str, Any]:
        run_ctx = self._execute_coverage_run(repo_path)
        if run_ctx.returncode != 0:
            return self._build_test_failure_result(run_ctx)

        xml_ctx = self._generate_coverage_xml()
        if xml_ctx.returncode != 0:
            return self._build_xml_failure_result(run_ctx, xml_ctx)

        try:
            coverage_pct, files_coverage = self._load_coverage_report(repo_path)
        except FileNotFoundError:
            logger.error("Coverage XML file not found")
            return {"status": "error", "error": "Coverage XML file not found"}

        total_duration = run_ctx.duration + xml_ctx.duration
        return self._build_success_result(
            coverage_pct,
            files_coverage,
            total_duration,
            run_ctx,
        )

    @classmethod
    def _parse_pytest_output(cls, stdout: Optional[str]) -> Optional[Dict[str, Any]]:
        prepared = cls._prepare_pytest_lines(stdout)
        if not prepared:
            return None

        text, lines = prepared
        summary_line = cls._find_summary_line(lines)
        summary = cls._parse_summary_details(summary_line)
        collected = cls._extract_collected_count(lines)
        raw_output = cls._truncate_output(text)

        return {
            "status": summary.status,
            "summary": summary_line,
            "counts": summary.counts,
            "collected": collected,
            "duration": summary.duration,
            "raw_output": raw_output,
        }

    @classmethod
    def _prepare_pytest_lines(
        cls, stdout: Optional[str]
    ) -> Optional[Tuple[str, List[str]]]:
        if stdout is None:
            return None

        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        elif not isinstance(stdout, str):
            return None

        if not stdout.strip():
            return None

        text = cls.ANSI_ESCAPE_RE.sub("", stdout).replace("\r\n", "\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None

        return text, lines

    @staticmethod
    def _find_summary_line(lines: List[str]) -> Optional[str]:
        for line in reversed(lines):
            if line.startswith("=") and line.endswith("="):
                candidate = line.strip("= ")
                if candidate:
                    return candidate
        return None

    @classmethod
    def _parse_summary_details(
        cls, summary_line: Optional[str]
    ) -> "Coverage._PytestSummary":
        if not summary_line:
            return cls._PytestSummary(status="success", counts=[], duration=None)

        counts = cls._extract_summary_counts(summary_line)
        status = cls._derive_summary_status(counts)
        duration = cls._extract_duration(summary_line)
        return cls._PytestSummary(status=status, counts=counts, duration=duration)

    @staticmethod
    def _extract_summary_counts(summary_line: str) -> List[Dict[str, Any]]:
        counts: List[Dict[str, Any]] = []
        for count_str, label in re.findall(r"(\d+)\s+([A-Za-z_]+)", summary_line):
            count = int(count_str)
            if count <= 0:
                continue
            label_lower = label.lower()
            counts.append({"label": label_lower, "count": count})
        return counts

    @staticmethod
    def _derive_summary_status(counts: List[Dict[str, Any]]) -> str:
        for entry in counts:
            if entry["label"] in {"failed", "error", "errors"}:
                return "error"
        return "success"

    @staticmethod
    def _extract_duration(summary_line: str) -> Optional[float]:
        match = re.search(r"in\s+([0-9]+(?:\.[0-9]+)?)s", summary_line)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _extract_collected_count(lines: List[str]) -> Optional[int]:
        for line in lines:
            match = re.match(r"collected\s+(\d+)\s+items?", line, re.IGNORECASE)
            if not match:
                continue
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    @staticmethod
    def _truncate_output(text: str) -> str:
        trimmed = text.strip()
        if len(trimmed) <= 20000:
            return trimmed
        return trimmed[-20000:]

    def _execute_coverage_run(self, repo_path: str) -> "Coverage._RunContext":
        logger.info("Running coverage on %s", repo_path)
        cmd = ["coverage", "run", "-m", *self.run_tests_cmd]
        start_time = time.perf_counter()
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        duration = time.perf_counter() - start_time

        pytest_details = self._parse_pytest_output(process.stdout)
        pytest_summary: Optional[Dict[str, Any]] = None
        embedded_results: List[Dict[str, Any]] = []
        if pytest_details:
            pytest_summary = dict(pytest_details)
            pytest_summary.pop("raw_output", None)
            embedded_results.append(
                {
                    "tool": "pytest",
                    "status": pytest_details.get("status", "unknown"),
                    "duration": pytest_details.get("duration") or duration,
                    "output": pytest_details,
                }
            )

        return self._RunContext(
            returncode=process.returncode,
            duration=duration,
            stdout=process.stdout,
            stderr=process.stderr,
            pytest_details=pytest_details,
            pytest_summary=pytest_summary,
            embedded_results=embedded_results,
        )

    def _generate_coverage_xml(self) -> "Coverage._XmlContext":
        cmd = ["coverage", "xml"]
        start_time = time.perf_counter()
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        duration = time.perf_counter() - start_time

        return self._XmlContext(
            returncode=process.returncode,
            duration=duration,
            stdout=process.stdout,
            stderr=process.stderr,
        )

    def _load_coverage_report(
        self, repo_path: str
    ) -> Tuple[float, List[Dict[str, Any]]]:
        coverage_xml_path = os.path.join(repo_path, "coverage.xml")
        if not os.path.exists(coverage_xml_path):
            raise FileNotFoundError("Coverage XML file not found")

        tree = ET.parse(coverage_xml_path)
        root = tree.getroot()

        coverage_pct = float(root.get("line-rate", "0")) * 100
        files_coverage = []
        for class_elem in root.findall(".//class"):
            filename = class_elem.get("filename", "unknown")
            line_rate = float(class_elem.get("line-rate", "0")) * 100
            files_coverage.append({"filename": filename, "coverage": line_rate})

        return coverage_pct, files_coverage

    def _build_test_failure_result(
        self, run_ctx: "Coverage._RunContext"
    ) -> Dict[str, Any]:
        logger.error("Test run failed with return code %s", run_ctx.returncode)
        error_result: Dict[str, Any] = {
            "status": "error",
            "error": f"Test run failed with return code {run_ctx.returncode}",
            "stdout": run_ctx.stdout,
            "stderr": run_ctx.stderr,
            "duration": run_ctx.duration,
        }
        self._apply_pytest_metadata(error_result, run_ctx)
        return error_result

    def _build_xml_failure_result(
        self,
        run_ctx: "Coverage._RunContext",
        xml_ctx: "Coverage._XmlContext",
    ) -> Dict[str, Any]:
        logger.error(
            "Coverage XML generation failed with return code %s",
            xml_ctx.returncode,
        )
        error_result: Dict[str, Any] = {
            "status": "error",
            "error": (
                "Coverage XML generation failed with return code "
                f"{xml_ctx.returncode}"
            ),
            "stdout": xml_ctx.stdout,
            "stderr": xml_ctx.stderr,
            "duration": run_ctx.duration + xml_ctx.duration,
        }
        self._apply_pytest_metadata(error_result, run_ctx)
        return error_result

    def _build_success_result(
        self,
        coverage_pct: float,
        files_coverage: List[Dict[str, Any]],
        duration: float,
        run_ctx: "Coverage._RunContext",
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "status": "success",
            "percentage": coverage_pct,
            "files": files_coverage,
            "duration": duration,
        }
        self._apply_pytest_metadata(result, run_ctx)
        return result

    @staticmethod
    def _apply_pytest_metadata(
        payload: Dict[str, Any], run_ctx: "Coverage._RunContext"
    ) -> None:
        if run_ctx.pytest_summary:
            payload["pytest_summary"] = run_ctx.pytest_summary
        if run_ctx.embedded_results:
            payload["embedded_results"] = run_ctx.embedded_results


class ToolRunner:
    """Runner for multiple tools."""

    def __init__(self, tools: Optional[List[Tool]] = None):
        """Initialize the tool runner.

        Args:
            tools: List of tools to run
        """
        self.tools = tools or []

    def add_tool(self, tool: Tool):
        """Add a tool to the runner.

        Args:
            tool: Tool to add
        """
        self.tools.append(tool)

    def run_all(self, repo_path: str) -> Dict[str, Dict[str, Any]]:
        """Run all tools.

        Args:
            repo_path: Path to the repository

        Returns:
            Dictionary with results from all tools
        """
        results = {}
        for tool in self.tools:
            results[tool.name] = tool.run(repo_path)
        return results
