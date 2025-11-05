import logging
from os import environ
import sys
import traceback
from types import TracebackType
from typing import Optional, Type, Union
from pathlib import Path
from logging.handlers import RotatingFileHandler
import inspect


class OMRLogger(logging.Logger):
    """Custom logger class that handles formatting, file output, and uncaught exceptions."""

    LOG_FORMAT = (
        "%(asctime)s [%(levelname)s] (%(name)s:%(funcName)s:%(lineno)d): %(message)s"
    )
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self, name: str, level: int = logging.INFO):
        super().__init__(name, level)
        if not self.handlers:
            self._configure_handlers()

    def _configure_handlers(self) -> None:
        formatter = logging.Formatter(self.LOG_FORMAT, datefmt=self.DATE_FORMAT)

        # Log to stderr (visible to C# and most runtime logs)
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        self.addHandler(stream_handler)

        # Log to file (with rotation)
        log_dir = Path(environ.get("OMR_LOG_DIR", "logs"))
        log_dir.mkdir(exist_ok=True, parents=True)
        file_handler = RotatingFileHandler(
            log_dir / "omr.log", maxBytes=2_000_000, backupCount=3
        )
        file_handler.setFormatter(formatter)
        self.addHandler(file_handler)

        self.propagate = False

    def handle_exception(
        self,
        exc_type: Type[BaseException],
        exc_value: BaseException,
        tb: Optional[TracebackType],
    ) -> None:
        """Handles uncaught exceptions globally and logs their origin."""
        if issubclass(exc_type, KeyboardInterrupt):
            self.warning("Execution interrupted by user (KeyboardInterrupt).")
            return

        last_frame = self._get_last_traceback_frame(tb)
        origin = self._format_origin(last_frame)
        self.critical(
            f"Unhandled exception in {origin}", exc_info=(exc_type, exc_value, tb)
        )

    def _get_last_traceback_frame(
        self, tb: Optional[TracebackType]
    ) -> Optional[TracebackType]:
        """Return the deepest traceback frame."""
        if tb is None:
            return None
        while tb.tb_next:
            tb = tb.tb_next
        return tb

    def _format_origin(self, tb: Optional[TracebackType]) -> str:
        """Return 'module.function:line' for the origin of an exception."""
        if tb is None:
            return "<unknown>"
        frame = tb.tb_frame
        module = inspect.getmodule(frame)
        module_name = module.__name__ if module else "<no module>"
        func_name = frame.f_code.co_name
        line = tb.tb_lineno
        return f"{module_name}.{func_name}:{line}"


def _global_excepthook(
    exc_type: Type[BaseException],
    exc_value: BaseException,
    tb: Optional[TracebackType],
) -> None:
    """Global exception hook that routes to OMRLogger."""
    logger = logging.getLogger("unhandled")
    if isinstance(logger, OMRLogger):
        logger.handle_exception(exc_type, exc_value, tb)
    else:
        # fallback in case logging.setLoggerClass wasn't called yet
        logging.critical("Unhandled exception", exc_info=(exc_type, exc_value, tb))
    traceback.print_exception(exc_type, exc_value, tb, file=sys.stderr)


def setup_logging(log_level: Union[int, str, None] = None) -> None:
    """Initialize OMRLogger globally."""
    logging.setLoggerClass(OMRLogger)
    sys.excepthook = _global_excepthook
    if log_level is not None:
        logging.basicConfig(level=log_level)
