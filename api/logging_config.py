"""Request and claim correlation for backend log records."""

import logging
import os
import sys
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_LOG_PATH = PROJECT_ROOT / "server.log"
request_id_context: ContextVar[str] = ContextVar("request_id", default="-")
claim_id_context: ContextVar[str] = ContextVar("claim_id", default="-")

LOG_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s "
    "request_id=%(request_id)s claim_id=%(claim_id)s %(message)s"
)


class CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = request_id_context.get()
        if not hasattr(record, "claim_id"):
            record.claim_id = claim_id_context.get()
        return True


def configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(f"Unsupported LOG_LEVEL: {level_name}")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if not root_logger.handlers:
        root_logger.addHandler(logging.StreamHandler(sys.stderr))

    if not any(
        handler.get_name() == "insuranceai-server-file"
        for handler in root_logger.handlers
    ):
        file_handler = RotatingFileHandler(
            SERVER_LOG_PATH,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.set_name("insuranceai-server-file")
        root_logger.addHandler(file_handler)

    for handler in root_logger.handlers:
        if not any(isinstance(item, CorrelationFilter) for item in handler.filters):
            handler.addFilter(CorrelationFilter())
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
