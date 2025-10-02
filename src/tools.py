"""Tools for code analysis and testing."""
import os
import logging
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Tool:
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


class Pylint(Tool):
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
            # Run Pylint and capture output
            logger.info(f"Running Pylint on {repo_path}")
            cmd = ["pylint", "--output-format=json", repo_path]
            process = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            # Parse Pylint output
            if process.returncode >= 0 and process.stdout:
                try:
                    results = json.loads(process.stdout)
                    
                    # Count issues by type
                    issues_by_type = {}
                    for item in results:
                        issue_type = item.get("type", "unknown")
                        issues_by_type[issue_type] = issues_by_type.get(issue_type, 0) + 1
                    
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
                        "details": results
                    }
                except json.JSONDecodeError:
                    logger.error("Failed to parse Pylint JSON output")
                    return {
                        "status": "error",
                        "error": "Failed to parse Pylint output"
                    }
            else:
                logger.error(f"Pylint failed with return code {process.returncode}")
                return {
                    "status": "error",
                    "error": f"Pylint failed with return code {process.returncode}",
                    "stdout": process.stdout,
                    "stderr": process.stderr
                }
        except Exception as e:
            logger.exception("Error running Pylint")
            return {
                "status": "error",
                "error": str(e)
            }


class Coverage(Tool):
    """Coverage measurement tool."""
    
    def __init__(self, run_tests_cmd: Optional[List[str]] = None):
        """Initialize Coverage.
        
        Args:
            run_tests_cmd: Command to run tests (defaults to pytest)
        """
        super().__init__("coverage")
        self.run_tests_cmd = run_tests_cmd or ["pytest"]
    
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
                logger.info(f"Running coverage on {repo_path}")
                cmd = ["coverage", "run", "-m"] + self.run_tests_cmd
                process = subprocess.run(cmd, capture_output=True, text=True, check=False)
                
                if process.returncode != 0:
                    logger.error(f"Test run failed with return code {process.returncode}")
                    return {
                        "status": "error",
                        "error": f"Test run failed with return code {process.returncode}",
                        "stdout": process.stdout,
                        "stderr": process.stderr
                    }
                
                # Generate XML report
                cmd = ["coverage", "xml"]
                process = subprocess.run(cmd, capture_output=True, text=True, check=False)
                
                if process.returncode != 0:
                    logger.error(f"Coverage XML generation failed with return code {process.returncode}")
                    return {
                        "status": "error",
                        "error": f"Coverage XML generation failed with return code {process.returncode}",
                        "stdout": process.stdout,
                        "stderr": process.stderr
                    }
                
                # Parse coverage XML
                coverage_xml_path = os.path.join(repo_path, "coverage.xml")
                if not os.path.exists(coverage_xml_path):
                    logger.error("Coverage XML file not found")
                    return {
                        "status": "error",
                        "error": "Coverage XML file not found"
                    }
                
                tree = ET.parse(coverage_xml_path)
                root = tree.getroot()
                
                # Extract coverage percentage
                coverage_pct = float(root.get("line-rate", "0")) * 100
                
                # Extract detailed coverage by file
                files_coverage = []
                for class_elem in root.findall(".//class"):
                    filename = class_elem.get("filename", "unknown")
                    line_rate = float(class_elem.get("line-rate", "0")) * 100
                    files_coverage.append({
                        "filename": filename,
                        "coverage": line_rate
                    })
                
                return {
                    "status": "success",
                    "percentage": coverage_pct,
                    "files": files_coverage
                }
            finally:
                # Return to original directory
                os.chdir(original_dir)
        except Exception as e:
            logger.exception("Error running Coverage")
            return {
                "status": "error",
                "error": str(e)
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
