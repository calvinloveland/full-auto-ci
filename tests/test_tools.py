"""Tests for the tools module."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.tools import Coverage, Pylint, Tool, ToolRunner


class TestTool(unittest.TestCase):
    """Test cases for the base Tool class."""

    def test_init(self):
        """Test initialization."""
        tool = Tool("test")
        self.assertEqual(tool.name, "test")

    def test_run_not_implemented(self):
        """Test that run() raises NotImplementedError."""
        tool = Tool("test")
        with self.assertRaises(NotImplementedError):
            tool.run("/path/to/repo")


class TestPylint(unittest.TestCase):
    """Test cases for the Pylint tool."""

    def setUp(self):
        """Set up test fixtures."""
        self.pylint = Pylint()

    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.pylint.name, "pylint")

    @patch("subprocess.run")
    def test_run_success(self, mock_run):
        """Test running Pylint with success."""
        # Mock the subprocess.run result
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = json.dumps(
            [
                {"type": "convention", "message": "Missing docstring"},
                {"type": "warning", "message": "Unused import"},
                {"type": "error", "message": "Undefined name"},
            ]
        )
        mock_run.return_value = mock_process

        # Run Pylint
        with patch.object(self.pylint, "_discover_targets", return_value=["src"]):
            result = self.pylint.run("/path/to/repo")

        # Verify the result
        self.assertEqual(result["status"], "success")
        self.assertLess(result["score"], 10.0)  # Score should be reduced for issues
        self.assertEqual(result["issues"]["convention"], 1)
        self.assertEqual(result["issues"]["warning"], 1)
        self.assertEqual(result["issues"]["error"], 1)
        mock_run.assert_called_with(
            ["pylint", "--output-format=json", "src"],
            capture_output=True,
            text=True,
            check=False,
            cwd="/path/to/repo",
        )

    @patch("subprocess.run")
    def test_run_fail(self, mock_run):
        """Test running Pylint with failure."""
        # Mock the subprocess.run result
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stdout = ""
        mock_process.stderr = "Error running pylint"
        mock_run.return_value = mock_process

        # Run Pylint
        with patch.object(self.pylint, "_discover_targets", return_value=["src"]):
            result = self.pylint.run("/path/to/repo")

        # Verify the result
        self.assertEqual(result["status"], "error")
        self.assertIn("error", result)

    @patch("subprocess.run")
    def test_run_invalid_json(self, mock_run):
        """Test running Pylint with invalid JSON output."""
        # Mock the subprocess.run result
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Not JSON"
        mock_run.return_value = mock_process

        # Run Pylint
        with patch.object(self.pylint, "_discover_targets", return_value=["src"]):
            result = self.pylint.run("/path/to/repo")

        # Verify the result
        self.assertEqual(result["status"], "error")
        self.assertIn("error", result)

    def test_discover_targets_prefers_src_packages(self):
        """Pylint should target packages inside the src directory when present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            package = Path(tmpdir) / "src" / "full_auto_ci"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")

            targets = self.pylint._discover_targets(
                tmpdir
            )  # pylint: disable=protected-access
            self.assertEqual(targets, ["src/full_auto_ci"])

    def test_discover_targets_respects_config(self):
        """When explicit config exists, run Pylint from repository root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / ".pylintrc").write_text("", encoding="utf-8")

            targets = self.pylint._discover_targets(
                tmpdir
            )  # pylint: disable=protected-access
            self.assertEqual(targets, ["."])

    def test_discover_targets_falls_back_to_packages(self):
        """If no src directory exists, fall back to top-level Python packages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "myapp"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("", encoding="utf-8")

            targets = self.pylint._discover_targets(
                tmpdir
            )  # pylint: disable=protected-access
            self.assertEqual(targets, ["myapp"])

    def test_discover_targets_defaults_to_repo(self):
        """If no obvious directories are found, lint the whole repo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            targets = self.pylint._discover_targets(
                tmpdir
            )  # pylint: disable=protected-access
            self.assertEqual(targets, ["."])


class TestCoverage(unittest.TestCase):
    """Test cases for the Coverage tool."""

    def setUp(self):
        """Set up test fixtures."""
        self.coverage = Coverage()

    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.coverage.name, "coverage")
        self.assertEqual(self.coverage.run_tests_cmd, ["pytest"])

        # Test with custom command
        custom_coverage = Coverage(run_tests_cmd=["python", "-m", "unittest"])
        self.assertEqual(custom_coverage.run_tests_cmd, ["python", "-m", "unittest"])

    @patch("os.chdir")
    @patch("subprocess.run")
    @patch("os.path.exists")
    @patch("xml.etree.ElementTree.parse")
    def test_run_success(self, mock_parse, mock_exists, mock_run, mock_chdir):
        """Test running Coverage with success."""
        # Mock the subprocess.run results
        mock_process1 = MagicMock()
        mock_process1.returncode = 0
        mock_process2 = MagicMock()
        mock_process2.returncode = 0
        mock_run.side_effect = [mock_process1, mock_process2]

        # Mock os.path.exists to return True for coverage.xml
        mock_exists.return_value = True

        # Mock XML parsing
        mock_root = MagicMock()
        mock_root.get.side_effect = lambda key, default: (
            "0.85" if key == "line-rate" else default
        )

        mock_file1 = MagicMock()
        mock_file1.get.side_effect = lambda key, default: (
            "file1.py"
            if key == "filename"
            else "0.9" if key == "line-rate" else default
        )

        mock_file2 = MagicMock()
        mock_file2.get.side_effect = lambda key, default: (
            "file2.py"
            if key == "filename"
            else "0.8" if key == "line-rate" else default
        )

        mock_root.findall.return_value = [mock_file1, mock_file2]

        mock_tree = MagicMock()
        mock_tree.getroot.return_value = mock_root
        mock_parse.return_value = mock_tree

        # Run Coverage
        result = self.coverage.run("/path/to/repo")

        # Verify the result
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["percentage"], 85.0)
        self.assertEqual(len(result["files"]), 2)
        self.assertEqual(result["files"][0]["filename"], "file1.py")
        self.assertEqual(result["files"][0]["coverage"], 90.0)
        self.assertEqual(result["files"][1]["filename"], "file2.py")
        self.assertEqual(result["files"][1]["coverage"], 80.0)
        self.assertEqual(mock_chdir.call_args_list[0][0][0], "/path/to/repo")

    @patch("os.chdir")
    @patch("subprocess.run")
    def test_run_test_fail(self, mock_run, mock_chdir):
        """Test running Coverage with test failure."""
        # Mock the subprocess.run result for test run
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stdout = "Test failed"
        mock_process.stderr = "Error in tests"
        mock_run.return_value = mock_process

        # Run Coverage
        result = self.coverage.run("/path/to/repo")

        # Verify the result
        self.assertEqual(result["status"], "error")
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Test run failed with return code 1")
        self.assertEqual(mock_chdir.call_args_list[0][0][0], "/path/to/repo")


class TestToolRunner(unittest.TestCase):
    """Test cases for the ToolRunner."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = ToolRunner()

    def test_init(self):
        """Test initialization."""
        self.assertEqual(len(self.runner.tools), 0)

        # Test with tools
        tool1 = Tool("tool1")
        tool2 = Tool("tool2")
        runner = ToolRunner(tools=[tool1, tool2])
        self.assertEqual(len(runner.tools), 2)
        self.assertEqual(runner.tools[0].name, "tool1")
        self.assertEqual(runner.tools[1].name, "tool2")

    def test_add_tool(self):
        """Test adding a tool."""
        tool = Tool("test")
        self.runner.add_tool(tool)
        self.assertEqual(len(self.runner.tools), 1)
        self.assertEqual(self.runner.tools[0].name, "test")

    def test_run_all(self):
        """Test running all tools."""
        # Create mock tools
        tool1 = MagicMock()
        tool1.name = "tool1"
        tool1.run.return_value = {"status": "success", "score": 9.5}

        tool2 = MagicMock()
        tool2.name = "tool2"
        tool2.run.return_value = {"status": "error", "error": "Something went wrong"}

        # Add tools to runner
        self.runner.add_tool(tool1)
        self.runner.add_tool(tool2)

        # Run all tools
        results = self.runner.run_all("/path/to/repo")

        # Verify the results
        self.assertEqual(len(results), 2)
        self.assertEqual(results["tool1"]["status"], "success")
        self.assertEqual(results["tool1"]["score"], 9.5)
        self.assertEqual(results["tool2"]["status"], "error")
        self.assertEqual(results["tool2"]["error"], "Something went wrong")

        # Verify that each tool's run method was called
        tool1.run.assert_called_once_with("/path/to/repo")
        tool2.run.assert_called_once_with("/path/to/repo")


if __name__ == "__main__":
    unittest.main()
