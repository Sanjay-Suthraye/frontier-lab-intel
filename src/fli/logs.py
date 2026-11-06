from __future__ import annotations

import logging
import os


def setup_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=(level or os.environ.get("FLI_LOG_LEVEL", "INFO")).upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # tweepy logs full request headers (incl. the bearer token) at DEBUG - never let it go below WARNING
    for noisy in ("httpx", "httpcore", "anthropic", "urllib3", "tweepy", "requests_oauthlib", "oauthlib",
                  "apscheduler.executors.default"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
