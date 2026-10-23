# notodata

Builds a tested, documented per-building table for the Noto 2024 earthquake
from a pinned Hugging Face dataset: spatial folds, per-seed splits and
clients, GSI image chips and an embedding cache.

## Requirements

- Docker and Docker Compose.
- A Hugging Face token with read access to the source dataset.

## Setup

1. Copy `.env.example` to `.env` and set `HF_TOKEN`. Keep the file private
   (`chmod 600 .env`); it is git-ignored and never read by the test suite.
2. Build the toolchain image:

   ```sh
   docker compose build
   ```

## Running

```sh
make check     # lint + offline unit tests
make test      # the full suite, including tests marked `real`
make build     # run every pipeline stage except the two that need a person
               # or an embedding encoder
make verify    # rebuild into a fresh cache reusing only the tile cache, and
               # compare hashes against MANIFEST.sha256
```

`NOTODATA_HOME` (default `./data`) holds every cache and is git-ignored.
Only `DATASET.md`, `MANIFEST.sha256` and the summary CSVs under `reports/`
are committed.

See `DATASET.md` for the full build record: sources, label meanings, class
counts, imagery coverage, the fold table, the seed register and the
reproducibility record.
