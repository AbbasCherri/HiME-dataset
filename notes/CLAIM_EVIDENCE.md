# Claim-to-evidence map

One row per testable claim. `Status` uses: `Passed`, `Failed`, `Blocked`,
`Partially verified`, `Not attempted`. Rows for issues D1-D8 are added by
the tasks that implement them; this file starts with the scaffold's own
claims.

| Claim | Code location | Test | Status |
| --- | --- | --- | --- |
| Every pinned constant lives in one frozen module and cannot be mutated at runtime | `src/notodata/config.py` | `tests/test_config_frozen.py` | Passed |
| A write that is interrupted mid-way never leaves a half-written file in place of the original | `src/notodata/io.py::atomic_write` | `tests/test_io.py` | Passed |
| Two parquet writes of the same data produce byte-identical files; rows are sorted and the pandas index is dropped | `src/notodata/io.py::write_parquet_deterministic` | `tests/test_io.py` | Passed |
| The CLI pins BLAS thread counts to 1 before numpy can be imported | `src/notodata/cli.py` | `tests/test_cli_threads.py` | Passed |
| No Hugging Face token literal is committed anywhere in the tree | repo-wide | `tests/test_no_token_literal_in_repo.py` | Passed |
| `.gitignore` and `.dockerignore` exclude `data/`, `.env` and the audit key before it is unblinded | `.gitignore`, `.dockerignore` | `tests/test_gitignore.py` | Passed |
| `test-unit`/`test` containers never see `HF_TOKEN`; every bind mount is SELinux-labelled | `docker-compose.yml` | `tests/test_compose_hermetic.py` | Passed |
| The committed lockfile matches what `uv` resolves and exports today | `uv.lock`, `requirements.lock` | `tests/test_locks_in_sync.py` | Passed |
| Every non-brief number used by the pipeline is logged in `DATASET.md` before first use | `DATASET.md` | `tests/test_deviations_logged.py` | Passed |
| Only the designated module imports its external dependency (`requests`, `huggingface_hub`, `datasets`) | `src/notodata/*.py` | `tests/test_import_boundaries.py` | Passed (vacuous: neither module exists yet) |
| D1-D8 | -- | -- | Not attempted (implemented in later tasks) |
