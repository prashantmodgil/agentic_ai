import logging
import sys

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
