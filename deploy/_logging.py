import os
import logging
import sys
import time

from loguru import logger


LOG_LEVEL = logging.getLevelName(os.environ.get("LOG_LEVEL", "DEBUG"))
JSON_LOGS = True if os.environ.get("JSON_LOGS", "0") == "1" else False
pid = str(os.getpid())


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(log_file=None):
    # intercept everything at the root logger
    # logger.remove()
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(LOG_LEVEL)

    # remove every other logger's handlers
    # and propagate to root logger
    for name in logging.root.manager.loggerDict.keys():
        # print(name)
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True
        # for pref in ["gunicorn", "uvicorn", "fastapi"]:
        #     if name.startswith(pref):
        #         logging.getLogger(name).handlers = []
        #         logging.getLogger(name).propagate = True

    # configure loguru

    def retention(files):
        stats = [(file, os.stat(file)) for file in files]
        stats.sort(key=lambda s: -s[1].st_mtime)  # Sort files from newest to oldest
        while sum(s[1].st_size for s in stats) > 2 * 1000 ** 3:  # 2 GB
            file, _ = stats.pop()
            os.remove(file)

    sys.stdout.reconfigure(encoding="utf-8")
    logger.configure(handlers=[{"sink": sys.stdout, "serialize": JSON_LOGS}])
    common_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <yellow>{process}</yellow> |<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    if log_file is not None:
        logger.add(
            log_file,
            rotation="200 MB",
            retention=retention,
            compression="zip",
            level=logging.INFO,
            encoding="utf8",
            format=common_format,
        )