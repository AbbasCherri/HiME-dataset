# Ambiguity log

Each entry records a decision made where the dataset brief and the build
plan leave a number or a procedure open, with a classification:

- `SPECIFIED` -- the brief or plan fixes the value; this just restates it.
- `PARTIALLY_SPECIFIED` -- the brief gives a rule but leaves one input open.
- `UNSPECIFIED` -- the brief is silent; this project decided and logs why.

| # | Issue | Decision | Class | Cost if wrong |
| --- | --- | --- | --- | --- |
| 1 | "Zero skips" and "`real` tests fail when data is missing" collide on a fresh clone | Two pytest targets: `test-unit` runs `pytest -m "not real"` (the per-change gate); `test` runs the full suite including `real` tests and is the final gate. `real` tests are expected to fail until the building table and clients exist. | PARTIALLY_SPECIFIED | Low |
| 2 | Source dataset changed mid-project | The dataset id is `AbbasABC/HiME-dataset`; the old pin belonged to a different repository and is dropped. The new revision is resolved once the dataset is inspected, and `config.py` holds `PENDING-TASK-2` until then. | UNSPECIFIED | If the new schema differs from the old one, later modules adjust |
| 3 | Earth radius | 6,371,008.8 m (IUGG mean); any standard radius changes distances by under 0.2%. | UNSPECIFIED | Low |
| 4 | Client counts | N = 50, 100, 200. | UNSPECIFIED | Low |
| 5 | Seed ranges | Development 1000-1004, tuning 2000-2004, confirmation 3000-3019, replication 4000-4019, kept disjoint across uses. | UNSPECIFIED | Low |
| 6 | Calibration draw | `default_rng([seed, client_id])`, one draw per client. | UNSPECIFIED | Low |
| 7 | Blank-chip rule | A chip is blank if its per-channel pixel standard deviation is below 2.0 grey levels, or at least half its pixels sit within 3 levels of black on every channel. | UNSPECIFIED | Medium -- a bug here reintroduces the old black-fill failure |
| 8 | GSI request limits | 8 concurrent requests, a 10 requests/second token bucket, 5 retries with exponential backoff on timeouts and 5xx, immediate skip (cached as missing) on 404. | UNSPECIFIED | Medium -- too aggressive risks a ban; too slow risks missing the build window |
| 9 | Candidate embedding encoders | `resnet50.tv_in1k_v2` (2048-d), `dinov2_vits14` (384-d), `openclip_vitb32_laion2b` (512-d), `satlas_aerial_swinb_si` (aerial-pretrained); chips resized to 224 px, bicubic. Which one is used downstream is chosen later, on validation data. | UNSPECIFIED | Low -- only affects which caches get built |
| 10 | Cognitive-complexity gate threshold | `complexipy --max-complexity-allowed 10`, enforced in the lint step; a function above it is refactored, never exempted. | SPECIFIED | Low |
