import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from lina_core.security.secrets import redact


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(str(record.msg))
        if record.args:
            record.args = (
                tuple(redact(str(item)) for item in record.args)
                if isinstance(record.args, tuple)
                else record.args
            )
        return True


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("lina_core")
    logger.setLevel(level)
    logger.propagate = False
    if getattr(logger, "_lina_configured", False):
        return logger
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.addFilter(RedactingFilter())
    logger.addHandler(console)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(formatter)
        handler.addFilter(RedactingFilter())
        logger.addHandler(handler)
    logger._lina_configured = True  # type: ignore[attr-defined]
    return logger
