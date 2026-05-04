import json
import logging
import logging.handlers
import os
import time
from logging.config import dictConfig
from pathlib import Path
from app.config.logging_config import (
    ERROR_FILE_NAME,
    FILE_PERMISSIONS,
    LOG_FILE_NAME,
    get_log_level,
    get_retention_days,
    use_json_logs,
)

logger = logging.getLogger(__name__)


class StructuredJsonFormatter(logging.Formatter):
    # Formats log records into structured JSON for machine-readable logging.

    def format(self, record: logging.LogRecord) -> str:
        ts = record.created
        utc = time.gmtime(ts)
        millis = int((ts - int(ts)) * 1000)

        timestamp = "{}.{:03d}Z".format(
            time.strftime("%Y-%m-%dT%H:%M:%S", utc),
            millis,
        )
        payload = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.thread,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False)


# Resolve project root from env or fallback to repository structure.
def _project_root() -> Path:
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


# Resolve and create log directory from env or fallback to project logs folder.
def _resolve_log_dir(project_root: Path) -> tuple[Path, str | None]:
    global warning
    env_dir = os.getenv("LOG_DIR")
    if env_dir:
        try:
            candidate = Path(env_dir).expanduser().resolve()
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate, None
        except Exception as exc:
            warning = f"Invalid LOG_DIR, falling back to default: {exc}"
    fallback = (project_root / "logs").resolve()
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback, warning if env_dir else None


# Apply secure file permissions to log files if they exist.
def _apply_file_permissions(*paths: Path) -> None:
    for path in paths:
        try:
            if path.exists():
                path.chmod(FILE_PERMISSIONS)
        except OSError:
            logger.debug("Failed to set permissions for %s", path)


# Ensure log files exist before applying permissions.
def _ensure_files_exist(*paths: Path) -> None:
    for path in paths:
        try:
            path.touch(exist_ok=True)
        except OSError:
            logger.debug("Failed to create file %s", path)


# Clear existing root handlers to prevent duplicate logging.
def _clear_root_handlers() -> list[str]:
    root = logging.getLogger()
    close_errors = []
    for handler in root.handlers[:]:
        try:
            handler.close()
        except Exception as exc:
            close_errors.append(f"{handler}: {exc}")
    root.handlers.clear()
    return close_errors


def _rotating_file_handler_base(formatter: str, retention_days: int) -> dict:
    return {
        "class": "logging.handlers.TimedRotatingFileHandler",
        "formatter": formatter,
        "when": "midnight",
        "interval": 1,
        "backupCount": retention_days,
        "encoding": "utf-8",
        "delay": True,
        "utc": True,
    }


def _build_formatters_config() -> dict:
    return {
        "text": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%SZ",
        },
        "json": {
            "()": StructuredJsonFormatter,
        },
    }


def _build_handlers_config(
    formatter: str,
    log_level: str,
    log_file: Path,
    error_file: Path,
    retention_days: int,
) -> dict:
    rotating_base = _rotating_file_handler_base(formatter, retention_days)

    return {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": formatter,
            "level": log_level,
            "stream": "ext://sys.stdout",
        },
        "app_file": {
            **rotating_base,
            "level": log_level,
            "filename": str(log_file),
        },
        "error_file": {
            **rotating_base,
            "level": "ERROR",
            "filename": str(error_file),
        },
    }


def _build_loggers_config(log_level: str) -> dict:
    return {
        "uvicorn": {
            "level": log_level,
            "handlers": ["console", "app_file", "error_file"],
            "propagate": False,
        },
        "uvicorn.error": {
            "level": log_level,
            "handlers": ["console", "app_file", "error_file"],
            "propagate": False,
        },
        "uvicorn.access": {
            "level": "INFO",
            "handlers": ["console", "app_file"],
            "propagate": False,
        },
    }


def _build_logging_config(
    formatter: str,
    log_level: str,
    log_file: Path,
    error_file: Path,
    retention_days: int,
) -> dict:
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": _build_formatters_config(),
        "handlers": _build_handlers_config(
            formatter,
            log_level,
            log_file,
            error_file,
            retention_days,
        ),
        "root": {
            "level": log_level,
            "handlers": ["console", "app_file", "error_file"],
        },
        "loggers": _build_loggers_config(log_level),
    }


def setup_logging() -> None:
    logging.Formatter.converter = time.gmtime

    log_level = get_log_level()
    retention_days = get_retention_days()
    formatter = "json" if use_json_logs() else "text"

    project_root = _project_root()
    log_dir, log_dir_warning = _resolve_log_dir(project_root)
    log_file = log_dir / LOG_FILE_NAME
    error_file = log_dir / ERROR_FILE_NAME

    handler_close_errors = _clear_root_handlers()

    config = _build_logging_config(
        formatter,
        log_level,
        log_file,
        error_file,
        retention_days,
    )
    dictConfig(config)

    if log_dir_warning:
        logger.warning(log_dir_warning)
    for close_error in handler_close_errors:
        logger.debug("Failed to close existing log handler: %s", close_error)

    _ensure_files_exist(log_file, error_file)
    _apply_file_permissions(log_file, error_file)
    logger.info(
        "Logging initialised | level=%s | format=%s | dir=%s | retention=%d days",
        log_level, formatter, log_dir, retention_days,
    )