"""Record the current PIPELINE_VERSION and chips.py hash after a deliberate version bump."""

from notodata import cache

if __name__ == "__main__":
    cache.update_pipeline_lock()
    print(f"recorded {cache.LOCK_PATH}")
