"""Record the current PIPELINE_VERSION and source hash after a deliberate version bump.

Usage: update_pipeline_hash.py [lock_path]   (default: the tracked lock file)
"""

import sys
from pathlib import Path

from notodata import cache

if __name__ == "__main__":
    lock = Path(sys.argv[1]) if len(sys.argv) > 1 else cache.LOCK_PATH
    try:
        cache.update_pipeline_lock(lock_path=lock)
    except cache.PipelineVersionError as exc:
        sys.exit(str(exc))
    print(f"recorded {lock}")
