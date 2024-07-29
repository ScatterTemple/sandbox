import logging

import optuna


_default_handler: logging.Handler | None = None


def _configure_handler():
    global _default_handler
    if _default_handler:
        # This library has already configured the library root logger.
        return
    _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.

    # Apply our default configuration to the library root logger.
    library_root_logger: logging.Logger = logging.getLogger('log')
    library_root_logger.addHandler(_default_handler)
    library_root_logger.setLevel(logging.INFO)


def get_logger():
    _configure_handler()
    logger = logging.getLogger("log")
    logger.setLevel(logging.DEBUG)
    return logger
