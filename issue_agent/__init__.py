"""GitHub issue fixing agent package."""

import subprocess

from .config import MAX_AGENT_TOOL_CALLS, logger
from .state import (
    STATE,
    AgentFinishedWithoutChanges,
    GitHubIssueNotFoundError,
    GitHubRepositoryUnavailableError,
)
from .repository import (
    _checkout_issue_branch,
    _fetch_issue_details,
    _omitted_python_definitions,
    _prepare_repository,
    _resolve_repo_path,
)
from .tool_calls import (
    _extract_text_tool_call,
    _message_text,
    _missing_required_tool_arguments,
    _normalize_tool_arguments,
)
from .tools import (
    append_file,
    create_pull_request,
    fetch_github_issue,
    finish_without_changes,
    list_files,
    read_file,
    run_tests,
    write_file,
)
from .runner import ChatOllama, create_agent, run_agentic_fix

__all__ = [
    "STATE", "AgentFinishedWithoutChanges", "GitHubIssueNotFoundError",
    "GitHubRepositoryUnavailableError", "MAX_AGENT_TOOL_CALLS", "logger",
    "subprocess", "_checkout_issue_branch", "_fetch_issue_details",
    "_omitted_python_definitions", "_prepare_repository", "_resolve_repo_path",
    "_extract_text_tool_call", "_message_text",
    "_missing_required_tool_arguments", "_normalize_tool_arguments",
    "append_file", "create_pull_request", "fetch_github_issue",
    "finish_without_changes", "list_files", "read_file", "run_tests",
    "write_file", "ChatOllama", "create_agent", "run_agentic_fix",
]
