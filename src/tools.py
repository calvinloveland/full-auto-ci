"""Tools for code analysis and testing."""

import json
import logging
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

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
        except Exception as e:  # pylint: disable=broad-except
            logger.exception("Error running Pylint")
            return {"status": "error", "error": str(e)}

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

    def run(self, repo_path: str) -> Dict[str, Any]:
        """Run coverage.

        Args:
            repo_path: Path to the repository

        Returns:
            Coverage results
        """
        try:
            # Change to repository directory
            original_dir = os.getcwd()
            os.chdir(repo_path)

            try:
                # Run tests with coverage
                logger.info("Running coverage on %s", repo_path)
                cmd = ["coverage", "run", "-m"] + self.run_tests_cmd
                start_time = time.perf_counter()
                process = subprocess.run(
                    cmd, capture_output=True, text=True, check=False
                )
                test_duration = time.perf_counter() - start_time

                pytest_details = self._parse_pytest_output(process.stdout)
                embedded_results: List[Dict[str, Any]] = []
                pytest_summary = None
                if pytest_details:
                    summary_payload = dict(pytest_details)
                    summary_payload.pop("raw_output", None)
                    pytest_summary = summary_payload
                    embedded_results.append(
                        {
                            "tool": "pytest",
                            "status": pytest_details.get("status", "unknown"),
                            "duration": pytest_details.get("duration") or test_duration,
                            "output": pytest_details,
                        }
                    )

                if process.returncode != 0:
                    logger.error(
                        "Test run failed with return code %s", process.returncode
                    )
                    error_result: Dict[str, Any] = {
                        "status": "error",
                        "error": f"Test run failed with return code {process.returncode}",
                        "stdout": process.stdout,
                        "stderr": process.stderr,
                        "duration": test_duration,
                    }
                    if pytest_summary:
                        error_result["pytest_summary"] = pytest_summary
                    if embedded_results:
                        error_result["embedded_results"] = embedded_results
                    return error_result

                # Generate XML report
                cmd = ["coverage", "xml"]
                start_xml = time.perf_counter()
                process = subprocess.run(
                    cmd, capture_output=True, text=True, check=False
                )
                xml_duration = time.perf_counter() - start_xml

                if process.returncode != 0:
                    logger.error(
                        "Coverage XML generation failed with return code %s",
                        process.returncode,
                    )
                    error_message = (
                        "Coverage XML generation failed with return code "
                        f"{process.returncode}"
                    )
                    error_result = {
                        "status": "error",
                        "error": error_message,
                        "stdout": process.stdout,
                        "stderr": process.stderr,
                        "duration": test_duration + xml_duration,
                    }
                    if pytest_summary:
                        error_result["pytest_summary"] = pytest_summary
                    if embedded_results:
                        error_result["embedded_results"] = embedded_results
                    return error_result

                # Parse coverage XML
                coverage_xml_path = os.path.join(repo_path, "coverage.xml")
                if not os.path.exists(coverage_xml_path):
                    logger.error("Coverage XML file not found")
                    return {"status": "error", "error": "Coverage XML file not found"}

                tree = ET.parse(coverage_xml_path)
                root = tree.getroot()

                # Extract coverage percentage
                coverage_pct = float(root.get("line-rate", "0")) * 100

                # Extract detailed coverage by file
                files_coverage = []
                for class_elem in root.findall(".//class"):
                    filename = class_elem.get("filename", "unknown")
                    line_rate = float(class_elem.get("line-rate", "0")) * 100
                    files_coverage.append({"filename": filename, "coverage": line_rate})

                result: Dict[str, Any] = {
                    "status": "success",
                    "percentage": coverage_pct,
                    "files": files_coverage,
                    "duration": test_duration + xml_duration,
                }
                if pytest_summary:
                    result["pytest_summary"] = pytest_summary
                if embedded_results:
                    result["embedded_results"] = embedded_results
                return result
            finally:
                # Return to original directory
                os.chdir(original_dir)
        except Exception as e:  # pylint: disable=broad-except
            logger.exception("Error running Coverage")
            return {"status": "error", "error": str(e)}

    @classmethod
    def _parse_pytest_output(cls, stdout: Optional[str]) -> Optional[Dict[str, Any]]:
        if not stdout:
            return None

        text = cls.ANSI_ESCAPE_RE.sub("", stdout).replace("\r\n", "\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None

        summary_line = None
        for line in reversed(lines):
            if line.startswith("=") and line.endswith("="):
                candidate = line.strip("= ")
                if candidate:
                    summary_line = candidate
                    break

        counts: List[Dict[str, Any]] = []
        status = "success"
        duration = None

        if summary_line:
            for count_str, label in re.findall(r"(\d+)\s+([A-Za-z_]+)", summary_line):
                count = int(count_str)
                label_lower = label.lower()
                if count > 0:
                    counts.append({"label": label_lower, "count": count})
                if label_lower in {"failed", "error", "errors"} and count > 0:
                    status = "error"

            duration_match = re.search(r"in\s+([0-9]+(?:\.[0-9]+)?)s", summary_line)
            if duration_match:
                try:
                    duration = float(duration_match.group(1))
                except ValueError:
                    duration = None

        collected = None
        for line in lines:
            match = re.match(r"collected\s+(\d+)\s+items?", line, re.IGNORECASE)
            if match:
                try:
                    collected = int(match.group(1))
                except ValueError:
                    collected = None
                break

        raw_output = text.strip()
        if len(raw_output) > 20000:
            raw_output = raw_output[-20000:]

        return {
            "status": status,
            "summary": summary_line,
            "counts": counts,
            "collected": collected,
            "duration": duration,
            "raw_output": raw_output,
        }


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
