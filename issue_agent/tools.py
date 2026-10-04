import hashlib
import os
import subprocess
import sys
from pathlib import Path

from langchain_core.tools import tool

from .config import logger
from .repository import (
    _checkout_issue_branch,
    _fetch_issue_details,
    _omitted_python_definitions,
    _resolve_repo_path,
)
from .state import AgentFinishedWithoutChanges, STATE, GitHubIssueNotFoundError


@tool
def fetch_github_issue(issue_number: int) -> str:
    """Fetch the title and body of a GitHub Issue by its issue number. Call this FIRST."""
    logger.info(f"[TOOL: fetch_github_issue] Fetching issue #{issue_number}...")
    try:
        data = _fetch_issue_details(issue_number)
        summary = (
            f"Issue #{issue_number}\nTitle: {data.get('title')}\n"
            f"State: {data.get('state')}\nDescription:\n{data.get('body')}"
        )
        logger.info(f"[TOOL: fetch_github_issue] Successfully fetched issue #{issue_number}")
        return summary
    except (GitHubIssueNotFoundError, RuntimeError) as e:
        logger.error("[TOOL: fetch_github_issue] Failed: %s", e)
        return f"Failed to fetch GitHub issue: {e}"

@tool
def list_files(path: str = ".") -> str:
    """List files under a repository-relative directory (default: repository root)."""
    logger.info("[TOOL: list_files] Searching '%s'...", path)
    try:
        directory_path = _resolve_repo_path(path)
        if not directory_path.is_dir():
            return f"Error listing files: Not a directory: {path}"

        file_paths = []
        for root, dirs, files in os.walk(directory_path):
            dirs[:] = [
                name for name in dirs
                if name not in {".git", "__pycache__", ".pytest_cache"}
            ]
            for file in files:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(STATE.repo_root).as_posix()
                file_paths.append(rel_path)
        
        output = "Repository Files:\n" + "\n".join(file_paths)
        logger.info(f"[TOOL: list_files] Found {len(file_paths)} files.")
        return output
    except (OSError, RuntimeError, ValueError) as e:
        logger.error("[TOOL: list_files] Failed: %s", e)
        return f"Error listing files: {str(e)}"

@tool
def read_file(path: str) -> str:
    """Read a repository-relative file, rejecting paths outside the repository."""
    logger.info(f"[TOOL: read_file] Reading '{path}'...")
    try:
        resolved_path = _resolve_repo_path(path)
        if not resolved_path.is_file():
            return f"Error reading file {path}: Not a file."
        with open(resolved_path, "r", encoding="utf-8") as f:
            content = f.read()
            logger.info(f"[TOOL: read_file] Read {len(content)} bytes.")
            STATE.read_files.add(resolved_path)
            return content
    except (OSError, RuntimeError, ValueError) as e:
        logger.error("[TOOL: read_file] Failed: %s", e)
        return f"Error reading file {path}: {str(e)}"

@tool
def write_file(path: str, content: str) -> str:
    """Replace a repository-relative file when an existing implementation must change."""
    logger.info(f"[TOOL: write_file] Writing updates to '{path}'...")
    
    # LLM Safeguard: Local models often wrap code in markdown (```python ... ```). Strip it.
    if content.strip().startswith("```"):
        lines = content.strip().split('\n')
        if len(lines) > 1 and lines[-1].strip() == "```":
            content = '\n'.join(lines[1:-1])

    try:
        if STATE.no_changes_reason is not None:
            return "Error: Agent already selected the no-change outcome; refusing edits."
        resolved_path = _resolve_repo_path(path)
        if resolved_path.is_file() and resolved_path.read_bytes() == content.encode("utf-8"):
            return f"No changes made: {path} already has the requested content."
        if resolved_path.is_file() and resolved_path.suffix in {".py", ".pyi"}:
            existing_content = resolved_path.read_text(encoding="utf-8")
            omitted_definitions = _omitted_python_definitions(
                existing_content, content
            )
            if omitted_definitions:
                omitted = ", ".join(omitted_definitions)
                return (
                    f"Error: Refusing to replace {path}; this would remove existing "
                    f"definitions: {omitted}. Preserve those definitions in the "
                    "replacement, or use append_file for an additive change."
                )
        if STATE.repo_root is None or not STATE.issue_branch:
            return "Error: Issue branch has not been initialized; refusing to edit files."
        current_branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=STATE.repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        if current_branch != STATE.issue_branch:
            _checkout_issue_branch(STATE.repo_root, STATE.issue_branch)

        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved_path, "w", encoding="utf-8") as f:
            f.write(content)
        STATE.written_files[resolved_path] = hashlib.sha256(
            resolved_path.read_bytes()
        ).hexdigest()
        STATE.full_tests_passed = False
        STATE.commit_created = False
        msg = f"Successfully updated {path}. Now run tests to verify."
        logger.info(f"[TOOL: write_file] {msg}")
        return msg
    except (
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as e:
        logger.error("[TOOL: write_file] Failed: %s", e)
        return f"Error writing to file {path}: {str(e)}"

@tool
def append_file(path: str, content: str) -> str:
    """Append a new function or test to a repository file without replacing existing content."""
    logger.info("[TOOL: append_file] Appending to '%s'...", path)
    try:
        if STATE.no_changes_reason is not None:
            return "Error: Agent already selected the no-change outcome; refusing edits."
        resolved_path = _resolve_repo_path(path)
        if resolved_path not in STATE.read_files:
            return f"Error: Read {path} before appending to it."
        if STATE.repo_root is None or not STATE.issue_branch:
            return "Error: Issue branch has not been initialized; refusing to edit files."

        addition = content.strip()
        if not addition:
            return "Error: Refusing to append empty content."

        existing = (
            resolved_path.read_text(encoding="utf-8")
            if resolved_path.is_file()
            else ""
        )
        if addition in existing:
            return f"No changes made: requested content is already present in {path}."

        current_branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=STATE.repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        if current_branch != STATE.issue_branch:
            _checkout_issue_branch(STATE.repo_root, STATE.issue_branch)

        separator = "" if not existing or existing.endswith("\n") else "\n"
        updated = f"{existing}{separator}\n{addition}\n"
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path.write_text(updated, encoding="utf-8")
        STATE.written_files[resolved_path] = hashlib.sha256(
            resolved_path.read_bytes()
        ).hexdigest()
        STATE.full_tests_passed = False
        STATE.commit_created = False
        return f"Successfully appended new content to {path}. Existing content was preserved."
    except (
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as e:
        logger.error("[TOOL: append_file] Failed: %s", e)
        return f"Error appending to file {path}: {e}"

@tool
def finish_without_changes(reason: str) -> str:
    """Finish without edits or a pull request when the issue is already satisfied or not actionable."""
    if STATE.written_files:
        return "Error: Changes have already been made; run tests and complete the PR workflow."
    if not STATE.read_files:
        return "Error: Inspect at least one repository file before deciding no changes are needed."
    STATE.no_changes_reason = reason.strip() or "The issue requires no code changes."
    raise AgentFinishedWithoutChanges(STATE.no_changes_reason)

@tool
def run_tests(test_file: str = "") -> str:
    """Run tests using pytest to verify fixes. You can leave test_file empty to run all tests."""
    STATE.full_tests_passed = False
    target = test_file if test_file else "all tests"
    logger.info(f"[TOOL: run_tests] Executing unit tests with pytest on {target}...")
    try:
        test_path = _resolve_repo_path(test_file) if test_file else None
        if test_path is not None and not test_path.exists():
            return f"Error running tests: Test path does not exist: {test_file}"
        cmd = [sys.executable, "-m", "pytest"]
        if test_file:
            cmd.append(str(test_path))
            
        result = subprocess.run(
            cmd, cwd=STATE.repo_root, capture_output=True, text=True, timeout=120
        )
        output = f"Test Output:\n{result.stdout}\n{result.stderr}\nExit Code: {result.returncode}"
        
        if result.returncode == 0:
            logger.info("[TOOL: run_tests] All tests PASSED.")
            STATE.full_tests_passed = not test_file
        else:
            logger.warning(f"[TOOL: run_tests] Tests FAILED with exit code {result.returncode}.")
        return output
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as e:
        logger.error("[TOOL: run_tests] Failed: %s", e)
        return f"Error running tests: {str(e)}"

@tool
def create_pull_request(title: str, body: str) -> str:
    """Commit only files written by this run and open a PR after the full test suite passes."""
    issue_number = STATE.issue_number
    commit_message = title or f"Fix issue #{issue_number}"
    pr_title = title or f"Fix issue #{issue_number}"
    pr_body = body or f"Fixes #{issue_number}"

    if issue_number < 1:
        return "Error: Issue number has not been initialized for this run."

    branch_name = f"fix-issue-{issue_number}"
    logger.info(f"[TOOL: create_pull_request] Preparing to push changes to '{branch_name}'...")
    
    try:
        if STATE.no_changes_reason is not None:
            return f"No pull request created: {STATE.no_changes_reason}"
        if STATE.repo_root is None:
            return "Error: Target repository has not been initialized."
        if not STATE.full_tests_passed:
            return "Error: Run the full test suite successfully after the last edit before creating a PR."
        if not STATE.written_files:
            return "Error: No files were written by this agent run; refusing to create an empty PR."

        current_branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=STATE.repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        if current_branch != branch_name:
            return f"Error: Expected branch '{branch_name}', currently on '{current_branch}'."

        relative_paths = []
        for path, expected_hash in STATE.written_files.items():
            if not path.is_file():
                return f"Error: Written file no longer exists: {path.relative_to(STATE.repo_root)}"
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                return (
                    f"Error: File changed after the agent wrote it; refusing to include "
                    f"unexpected contents: {path.relative_to(STATE.repo_root)}"
                )
            relative_paths.append(path.relative_to(STATE.repo_root).as_posix())

        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=STATE.repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.splitlines()
        if staged:
            return (
                "Error: The Git index contains staged changes not created by this run; "
                "refusing to commit them."
            )

        if not STATE.commit_created:
            subprocess.run(
                ["git", "add", "--", *relative_paths],
                cwd=STATE.repo_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            staged = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                cwd=STATE.repo_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            ).stdout.splitlines()
            if not staged or not set(staged).issubset(set(relative_paths)):
                return (
                    "Error: Staged files were empty or included files outside this run's "
                    "written-file list; refusing to commit."
                )

            subprocess.run(
                ["git", "commit", "-m", commit_message],
                cwd=STATE.repo_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            STATE.commit_created = True
        subprocess.run(
            ["git", "push", "-u", "origin", branch_name],
            cwd=STATE.repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )

        pr_cmd = ["gh", "pr", "create", "--title", pr_title, "--body", pr_body, "--head", branch_name]
        result = subprocess.run(
            pr_cmd,
            cwd=STATE.repo_root,
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        if result.returncode == 0:
            success_msg = f"Pull Request successfully created! URL: {result.stdout.strip()}"
            STATE.pr_created = True
            logger.info(f"[TOOL: create_pull_request] {success_msg}")
            return success_msg
        else:
            err_output = result.stderr.strip()
            if "already exists" in err_output.lower():
                success_msg = f"Branch '{branch_name}' updated. The existing Pull Request was automatically updated with the new code!"
                STATE.pr_created = True
                logger.info(f"[TOOL: create_pull_request] {success_msg}")
                return success_msg
            return f"Code pushed to branch, but PR command failed: {err_output}"
        
    except subprocess.CalledProcessError as e:
        err = e.stderr or str(e)
        logger.error(f"[TOOL: create_pull_request] Git failed: {err}")
        return f"Git failed: {err}"
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as e:
        logger.error(f"[TOOL: create_pull_request] Error: {str(e)}")
        return f"Error during PR execution: {str(e)}"
