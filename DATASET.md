# Noto 2024 building-collapse dataset (HFL rewrite)

Status: scaffold only (Task 1) · Pipeline version: 1 · Commit: <filled by `notodata report`>

## 1. Summary

One paragraph: what the table is, how many buildings, collapsed share,
imagery setting (post layer, pre layer, pre_available), L, seeds built.
Filled once the pipeline has run.

## 2. Deviations from the brief

| # | Item | Brief value | Used value | Reason | Date |
| --- | --- | --- | --- | --- | --- |
| 1 | Raw source | `AbbasABC/HFL-Dataset` | Zenodo `10.5281/zenodo.11055711` (`Noto_Peninsula_Damage_2_5.gpkg`, MD5 `280e9c53cb45086786c20ed0b2c21ffb`) plus USGS ShakeMap `us6000m0xl` v10 | The HF dataset was unusable as a raw source; per-building labels come from the published ESSD dataset and the seismic columns are sampled from the ShakeMap grid | 2026-10-23 |
| 2 | Code 99 meaning | `unknown` until a named source | `missing_inconsistent` | Vescovo et al. 2025 section 2.2: 99 = "missing or inconsistent" footprint | 2026-10-23 |
| 3 | Earth radius | not fixed by the brief | 6371008.8 m (IUGG mean) | Any standard radius changes distances by under 0.2% | 2026-10-23 |
| 4 | Client counts | not fixed by the brief | N = 50, 100 and 200 | About 1,200, 600 and 300 training buildings per client; 600 is the planning figure used downstream | 2026-10-23 |
| 5 | Seed ranges | not fixed by the brief | Development 1000-1004, tuning 2000-2004, confirmation 3000-3019, replication 4000-4019 | Four disjoint ranges are needed; 20 seeds in confirmation and replication give enough margin for a Holm correction over 3 client counts | 2026-10-23 |
| 6 | Calibration draw | not fixed by the brief | `default_rng([s, client_id])`, one draw per client | Exact 20% stratification, reproducible per client without a shared generator | 2026-10-23 |
| 7 | Blank-chip rule | not fixed by the brief | A chip is `blank` if its per-channel pixel standard deviation is below 2.0 grey levels, or at least half its pixels are within 3 levels of black on every channel | The old pipeline's bug was a black-fill chip passing as real imagery; this catches both a flat chip and a mostly-black one | 2026-10-23 |
| 8 | GSI request limits | not fixed by the brief | 8 concurrent requests, a 10 requests/second token bucket, 5 retries with exponential backoff on timeouts and 5xx, 404 cached as a negative entry | Polite load against a public tile server; about 12 hours for 400,000 requests | 2026-10-23 |
| 9 | Candidate embedding encoders | not fixed by the brief | `resnet50.tv_in1k_v2` (2048-d), `dinov2_vits14` (384-d), `openclip_vitb32_laion2b` (512-d), `satlas_aerial_swinb_si` (aerial-pretrained); chips resized to 224 px, bicubic | Two general encoders, one language-aligned, one trained on aerial imagery; which one is used downstream is chosen later, on validation data | 2026-10-23 |
| 10 | Cognitive-complexity gate | not fixed by the brief | `complexipy --max-complexity-allowed 10` | A function above the threshold is refactored into smaller pieces, never exempted | 2026-10-23 |

## 3. Sources

- Zenodo `10.5281/zenodo.11055711`, v2.5, `Noto_Peninsula_Damage_2_5.gpkg`
  (CC-BY-4.0; Vescovo et al. 2025, doi:10.5194/essd-17-5259-2025)
- USGS ShakeMap `us6000m0xl`, version 10 (US Public Domain)
  <!-- AUTO:source_files --> file, size, sha256 <!-- /AUTO -->
- Dataset card and every source it cites (URL, date accessed)
- GSI Noto page, GSI tile list, GSI content-use terms
- Old repository @ `<commit>` (reference only; evidence lines quoted)
- HiME design doc, sections 2, 5, 6, 13

## 4. Columns and units

Seismic columns with units and provenance; `MMI_shape` definition (quoted)
or "undefined in sources checked: `<list>`"; counts of non-positive values.

## 5. Label meanings

| Code | Meaning | Quote | Source |
| --- | --- | --- | --- |

Code-99 branch used: `<branch>`, because `<evidence>`.

## 6. Class counts

<!-- AUTO:class_counts_raw --> raw vs brief, with differences <!-- /AUTO -->
<!-- AUTO:class_counts_mapped --> by label_raw x y_collapse x observable <!-- /AUTO -->

## 7. Imagery

### 7.1 Layers

<!-- AUTO:layers --> id, role, capture dates, districts, template, found in <!-- /AUTO -->

### 7.2 Coverage per layer

<!-- AUTO:coverage_by_layer --> <!-- /AUTO -->

### 7.3 Blinded audit

Rater, date, procedure, then <!-- AUTO:audit --> counts by layer x class x mark <!-- /AUTO -->
Decision: n_seam = `<k>`/20, n_post = `<k>`/20 -> rule row `<r>`.
Post layer `<id>` (`<date>`), pre layer `<id or none>`, pre_available = `<bool>`.
Dropped by post coverage: <!-- AUTO:dropped --> <!-- /AUTO -->

### 7.4 Coverage per image pattern

<!-- AUTO:coverage_by_pattern --> IP / I / none, overall and per class <!-- /AUTO -->

### 7.5 Chip definition

Zoom 18, 128 px, window maths, status rules, blank threshold, hash definition.

### 7.6 Tile fingerprints

<!-- AUTO:tile_fingerprints --> layer, tile count, sha256 <!-- /AUTO -->

## 8. Spatial design

Projection reference (lat0, lon0, R).
<!-- AUTO:moran --> bin, n, W, I <!-- /AUTO -->
L = `<km>` (rule case: `<first bin below 0.05 | clamped>`).
<!-- AUTO:folds --> per fold: blocks, buildings, collapsed, survived, unlabelled, G <!-- /AUTO -->
<!-- AUTO:splits_summary --> per seed: roles x class <!-- /AUTO -->
<!-- AUTO:clients_summary --> per (N, seed) <!-- /AUTO -->

## 9. Statistics computed on all folds

Moran's I sample, fold balancing, audit sample: why each is dataset-level.

## 10. Licence and redistribution

HF dataset licence (quoted). GSI terms (quoted attribution wording).
Policy: tiles and chips are fetched at build time, cached locally,
git-ignored, never redistributed.

## 11. Reproducibility

Python 3.12, image base `python:3.12-slim-bookworm`
(`sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70`),
thread settings (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`
all 1), <!-- AUTO:pip_freeze --> <!-- /AUTO -->
<!-- AUTO:seed_register --> <!-- /AUTO -->
How to rebuild: `make build`; how to verify: `notodata verify --clean`.
<!-- AUTO:verify_result --> date, mode, pass or fail per file <!-- /AUTO -->

## 12. Known limitations

Cloud or snow in January layers, automatic-orthophoto distortions,
seamlessphoto composite dates, equirectangular error over 60 km,
anything the audit marked `cannot_tell`.
