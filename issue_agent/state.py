from dataclasses import dataclass, field
from pathlib import Path

from langgraph.errors import GraphBubbleUp


@dataclass
class AgentState:
    repo_root: Path | None = None
    written_files: dict[Path, str] = field(default_factory=dict)
    read_files: set[Path] = field(default_factory=set)
    full_tests_passed: bool = False
    issue_number: int = 0
    issue_branch: str = ""
    no_changes_reason: str | None = None
    pr_created: bool = False
    commit_created: bool = False

    def reset(self) -> None:
        self.repo_root = None
        self.written_files.clear()
        self.read_files.clear()
        self.full_tests_passed = False
        self.issue_number = 0
        self.issue_branch = ""
        self.no_changes_reason = None
        self.pr_created = False
        self.commit_created = False


STATE = AgentState()


class GitHubIssueNotFoundError(ValueError):
    """Raised when GitHub confirms that the requested issue does not exist."""


class GitHubRepositoryUnavailableError(RuntimeError):
    """Raised when the GitHub repository cannot be found or accessed."""


class AgentFinishedWithoutChanges(GraphBubbleUp):
    """Stops the LangGraph run after the agent declares no work is needed."""
