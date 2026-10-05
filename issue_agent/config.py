import os
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("agent_execution.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("AgenticAI")
MAX_AGENT_TOOL_CALLS = 50
MODEL_NAME = "llama3.1"
LANGSMITH_DEFAULT_PROJECT = "github-issue-agent"


def configure_langsmith_tracing() -> bool:
    tracing_enabled = os.getenv("LANGSMITH_TRACING", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if not tracing_enabled:
        logger.info("LangSmith tracing is disabled.")
        return False

    project = os.getenv("LANGSMITH_PROJECT", "").strip()
    if not project:
        project = LANGSMITH_DEFAULT_PROJECT
        os.environ["LANGSMITH_PROJECT"] = project

    if not os.getenv("LANGSMITH_API_KEY"):
        logger.warning(
            "LangSmith tracing is enabled, but LANGSMITH_API_KEY is not set; "
            "traces may not be uploaded."
        )

    logger.info("LangSmith tracing is enabled for project '%s'.", project)
    return True
