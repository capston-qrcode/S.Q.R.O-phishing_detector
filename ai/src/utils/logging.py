import logging

from ai.src.core.settings import GeneralSettings


def setup_logging(settings: GeneralSettings) -> logging.Logger:
    logger = logging.getLogger("ai-main")
    formatter = logging.Formatter("%(asctime)s-%(levelname)s:%(message)s")

    handler = logging.StreamHandler()

    handler.setFormatter(formatter)
    logger.setLevel(settings.LOGGING_LEVEL)

    return logger
