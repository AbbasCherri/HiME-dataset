"""Minimal CLI entry point.

Thread-limiting environment variables are set here, before numpy (or
anything that transitively imports it) is imported in this process, so
BLAS-backed libraries never see more than one thread -- required for
bit-stable KMeans fits and distance sums across machines with a different
core count. Stages are added task by task; today this only proves the
entry point and the thread setup it must perform first.
"""

from __future__ import annotations

import logging
import os
import sys

_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def set_single_threaded_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Force OMP/MKL/OPENBLAS to one thread each; return what was set.

    Must run before numpy is imported in this process, because these
    libraries read the variable once, at load time.
    """
    target = os.environ if env is None else env
    for name in _THREAD_ENV_VARS:
        target[name] = "1"
    return {name: target[name] for name in _THREAD_ENV_VARS}


def main(argv: list[str] | None = None) -> int:
    values = set_single_threaded_env()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("notodata.cli")
    for name, value in values.items():
        logger.info("%s=%s", name, value)

    import numpy as np  # noqa: F401  (imported only after threads are pinned)

    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        logger.info("stage %r is not implemented yet", args[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
