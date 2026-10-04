import logging
import sys

from .config import logger
from .runner import run_agentic_fix


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) < 2:
        logger.error("Usage: python agent.py <GITHUB_ISSUE_NUMBER> <PATH_TO_TARGET_REPO>")
        return 1
    try:
        issue_number = int(arguments[0])
    except ValueError:
        logger.error("Issue number must be a positive integer.")
        return 2
    return run_agentic_fix(issue_number, arguments[1])


if __name__ == "__main__":
    raise SystemExit(main())
