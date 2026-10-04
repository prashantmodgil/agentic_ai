import ast
import json
import subprocess
from pathlib import Path

from .state import (
    STATE,
    GitHubIssueNotFoundError,
    GitHubRepositoryUnavailableError,
)


def _resolve_repo_path(path: str) -> Path:
    if STATE.repo_root is None:
        raise RuntimeError("The target repository has not been initialized.")

    candidate = (STATE.repo_root / path).resolve()
    if not candidate.is_relative_to(STATE.repo_root):
        raise ValueError(f"Path is outside the target repository: {path}")
    return candidate

def _omitted_python_definitions(existing: str, replacement: str) -> list[str]:
    try:
        existing_tree = ast.parse(existing)
        replacement_tree = ast.parse(replacement)
    except SyntaxError:
        return []

    def definitions(tree: ast.Module) -> set[str]:
        return {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }

    return sorted(definitions(existing_tree) - definitions(replacement_tree))

def _prepare_repository(repo_path: str) -> Path:
    requested_root = Path(repo_path).expanduser().resolve()
    if not requested_root.is_dir():
        raise ValueError(f"Repository directory does not exist: {requested_root}")

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=requested_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as e:
        raise ValueError(
            f"Directory is not a Git repository: {requested_root}"
        ) from e
    repo_root = Path(result.stdout.strip()).resolve()
    if repo_root != requested_root:
        raise ValueError(
            f"Pass the repository root, not a subdirectory: {repo_root}"
        )

    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Unable to inspect Git repository status: {e.stderr}") from e
    if status.stdout.strip():
        raise RuntimeError(
            "Repository has pre-existing changes. Commit, stash, or remove them "
            "before starting the agent so they cannot be included accidentally."
        )

    return repo_root

def _checkout_issue_branch(repo_root: Path, branch_name: str) -> None:
    branch_exists = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"],
        cwd=repo_root,
        capture_output=True,
        timeout=30,
    ).returncode == 0
    checkout_cmd = (
        ["git", "checkout", branch_name]
        if branch_exists
        else ["git", "checkout", "-b", branch_name]
    )
    subprocess.run(
        checkout_cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )

def _fetch_issue_details(issue_number: int) -> dict[str, str]:
    if STATE.repo_root is None:
        raise RuntimeError("The target repository has not been initialized.")

    try:
        result = subprocess.run(
            [
                "gh",
                "issue",
                "view",
                str(issue_number),
                "--json",
                "number,title,body,state,url",
            ],
            cwd=STATE.repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as e:
        error = (e.stderr or e.stdout or str(e)).strip()
        if "repository" in error.lower() and "not found" in error.lower():
            raise GitHubRepositoryUnavailableError(
                f"GitHub repository is unavailable or inaccessible: {error}"
            ) from e
        if any(
            marker in error.lower()
            for marker in (
                "could not resolve to a repository",
                "repository does not exist",
            )
        ):
            raise GitHubRepositoryUnavailableError(
                f"GitHub repository is unavailable or inaccessible: {error}"
            ) from e
        if any(
            marker in error.lower()
            for marker in (
                "could not resolve to an issue",
                "issue not found",
                "no issue found",
            )
        ):
            raise GitHubIssueNotFoundError(
                f"GitHub issue #{issue_number} does not exist in this repository."
            ) from e
        raise RuntimeError(f"Unable to fetch GitHub issue #{issue_number}: {error}") from e
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"Unable to fetch GitHub issue #{issue_number}: {e}") from e

    try:
        details = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"GitHub returned invalid data for issue #{issue_number}.") from e

    if not details.get("title") and not details.get("body"):
        raise GitHubIssueNotFoundError(
            f"GitHub issue #{issue_number} does not exist in this repository."
        )
    return details
