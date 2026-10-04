"""Backward-compatible command-line entry point for the issue agent."""

from issue_agent import *  # noqa: F401,F403
from issue_agent import (
    _checkout_issue_branch,
    _extract_text_tool_call,
    _fetch_issue_details,
    _message_text,
    _missing_required_tool_arguments,
    _normalize_tool_arguments,
    _omitted_python_definitions,
    _prepare_repository,
    _resolve_repo_path,
)
from issue_agent.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
