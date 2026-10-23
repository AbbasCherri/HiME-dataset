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
| 7 | Blank-chip rule | not fixed by the brief | A chip is `blank` if its pixel standard deviation over all channels is below 2.0 grey levels, or at least half its pixels are within 3 levels of black on every channel | The old pipeline's bug was a black-fill chip passing as real imagery; this catches both a flat chip and a mostly-black one | 2026-10-23 |
| 8 | GSI request limits | not fixed by the brief | 8 concurrent requests, a 10 requests/second token bucket, 5 retries with exponential backoff (429/503 back off from 5 s doubling, capped at 60 s; 1 s base otherwise; an empty 200 body counts as a retryable failure), 404 cached as a negative entry | Polite load against a public tile server; about 12 hours for 400,000 requests | 2026-10-23 |
| 9 | Candidate embedding encoders | not fixed by the brief | `resnet50.tv_in1k_v2` (2048-d), `dinov2_vits14` (384-d), `openclip_vitb32_laion2b` (512-d), `satlas_aerial_swinb_si` (aerial-pretrained); chips resized to 224 px, bicubic | Two general encoders, one language-aligned, one trained on aerial imagery; which one is used downstream is chosen later, on validation data | 2026-10-23 |
| 10 | Cognitive-complexity gate | not fixed by the brief | `complexipy --max-complexity-allowed 10` | A function above the threshold is refactored into smaller pieces, never exempted | 2026-10-23 |
| 11 | Cache sidecar key `building_ids_sha256` | the brief's sidecar keys: `chip_pipeline_version`, `layer_ids`, `zoom`, `chip_size`, `encoder_id`, `row_count` | the same keys plus `building_ids_sha256` (SHA-256 of the sorted building id list) | The same row count with different buildings must also force a recompute | 2026-10-23 |

## 3. Sources

- Zenodo `10.5281/zenodo.11055711`, v2.5, `Noto_Peninsula_Damage_2_5.gpkg`
  (CC-BY-4.0; Vescovo et al. 2025, doi:10.5194/essd-17-5259-2025)
- USGS ShakeMap `us6000m0xl`, version 10 (US Public Domain)
  <!-- AUTO:source_files -->
| file | size | sha256 |
| --- | --- | --- |
| Noto_Peninsula_Damage_2_5.gpkg | 47853568 | 1adf3ff883951db42728448e6f338f93c9d0343d11de11fd8274dabfc27aaf9d |
| grid.xml | 7559752 | c93c08868b5e79d6141bef44d141dee6c8930ef40805c24803480d046b072a81 |
<!-- /AUTO -->
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

<!-- AUTO:class_counts_raw -->
| code | count | in_bbox | old_count | difference |
| --- | --- | --- | --- | --- |
| 0 | 112275 | 110208 | 104917 | 7358 |
| 1 | 3461 | 3440 | 3407 | 54 |
| 9 | 9437 | 8280 | 8365 | 1072 |
| 99 | 15035 | 14873 | 12154 | 2881 |
| total | 140208 | 136801 | 128843 | 11365 |
<!-- /AUTO -->
<!-- AUTO:class_counts_mapped -->
| label_raw | y_collapse | observable | count |
| --- | --- | --- | --- |
| 0 | 0 | 1 | 112275 |
| 1 | 1 | 1 | 3461 |
| 9 | NaN | 0 | 9437 |
| 99 | NaN | 0 | 15035 |
<!-- /AUTO -->

## 7. Imagery

### 7.1 Layers

<!-- AUTO:layers -->
| id | role | capture dates | districts | template | found in |
| --- | --- | --- | --- | --- | --- |
| seamlessphoto | baseline | composite; per-pixel date unknown | All Japan | https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg | GSI tile list page (ichiran.html) |
| nendophoto2018 | pre | fiscal year 2018 | Varies by year | https://cyberjapandata.gsi.go.jp/xyz/nendophoto2018/{z}/{x}/{y}.png | GSI tile list page (ichiran.html) |
| nendophoto2019 | pre | fiscal year 2019 | Varies by year | https://cyberjapandata.gsi.go.jp/xyz/nendophoto2019/{z}/{x}/{y}.png | GSI tile list page (ichiran.html) |
| nendophoto2020 | pre | fiscal year 2020 | Varies by year | https://cyberjapandata.gsi.go.jp/xyz/nendophoto2020/{z}/{x}/{y}.png | GSI tile list page (ichiran.html) |
| nendophoto2021 | pre | fiscal year 2021 | Varies by year | https://cyberjapandata.gsi.go.jp/xyz/nendophoto2021/{z}/{x}/{y}.png | GSI tile list page (ichiran.html) |
| nendophoto2022 | pre | fiscal year 2022 | Varies by year | https://cyberjapandata.gsi.go.jp/xyz/nendophoto2022/{z}/{x}/{y}.png | GSI tile list page (ichiran.html) |
| 20240102noto_0405_0426do | post | 2024-04-05 to 2024-04-26 | Whole Noto peninsula | https://cyberjapandata.gsi.go.jp/xyz/20240102noto_0405_0426do/{z}/{x}/{y}.png | GSI tile list page (ichiran.html) |
| 20240102noto_wazimanishi_0117do | post | 2024-01-17 | Wajima West | https://cyberjapandata.gsi.go.jp/xyz/20240102noto_wazimanishi_0117do/{z}/{x}/{y}.png | GSI tile list page (ichiran.html) |
| 20240102noto_anamizu_0117do | post | 2024-01-17 | Anamizu | https://cyberjapandata.gsi.go.jp/xyz/20240102noto_anamizu_0117do/{z}/{x}/{y}.png | GSI tile list page (ichiran.html) |
| 20240102noto_nanao_0117do | post | 2024-01-17 | Nanao | https://cyberjapandata.gsi.go.jp/xyz/20240102noto_nanao_0117do/{z}/{x}/{y}.png | GSI tile list page (ichiran.html) |
| 20240102noto_suzu_0114do | post | 2024-01-14 | Suzu | https://cyberjapandata.gsi.go.jp/xyz/20240102noto_suzu_0114do/{z}/{x}/{y}.png | GSI tile list page (ichiran.html) |
| 20240102noto_wazimahigashi_0114do | post | 2024-01-14 | Wajima East | https://cyberjapandata.gsi.go.jp/xyz/20240102noto_wazimahigashi_0114do/{z}/{x}/{y}.png | GSI tile list page (ichiran.html) |
| 20240102noto_anamizu_0114do | post | 2024-01-14 | Anamizu | https://cyberjapandata.gsi.go.jp/xyz/20240102noto_anamizu_0114do/{z}/{x}/{y}.png | GSI tile list page (ichiran.html) |
| 20240102noto_wazimanaka_0111do | post | 2024-01-11 | Wajima Central | https://cyberjapandata.gsi.go.jp/xyz/20240102noto_wazimanaka_0111do/{z}/{x}/{y}.png | GSI tile list page (ichiran.html) |
| 20240102noto_wazimanishi_0111do | post | 2024-01-11 | Wajima West | https://cyberjapandata.gsi.go.jp/xyz/20240102noto_wazimanishi_0111do/{z}/{x}/{y}.png | GSI tile list page (ichiran.html) |
| 20240102noto_anamizu_0111do | post | 2024-01-11 | Anamizu | https://cyberjapandata.gsi.go.jp/xyz/20240102noto_anamizu_0111do/{z}/{x}/{y}.png | GSI tile list page (ichiran.html) |
| 20240102_noto_suzu_0105do | post | 2024-01-05 | Suzu | https://cyberjapandata.gsi.go.jp/xyz/20240102_noto_suzu_0105do/{z}/{x}/{y}.png | GSI tile list page (ichiran.html) |
| 20240102_noto_wazimanaka_0105do | post | 2024-01-05 | Wajima Central | https://cyberjapandata.gsi.go.jp/xyz/20240102_noto_wazimanaka_0105do/{z}/{x}/{y}.png | GSI tile list page (ichiran.html) |
| 20240102_noto_anamizu_0105do | post | 2024-01-05 | Anamizu | https://cyberjapandata.gsi.go.jp/xyz/20240102_noto_anamizu_0105do/{z}/{x}/{y}.png | GSI tile list page (ichiran.html) |
| 20240102_noto_nanao_0105do | post | 2024-01-05 | Nanao | https://cyberjapandata.gsi.go.jp/xyz/20240102_noto_nanao_0105do/{z}/{x}/{y}.png | GSI tile list page (ichiran.html) |
| 20240102noto_suzu_0102do | post | 2024-01-02 | Suzu | https://cyberjapandata.gsi.go.jp/xyz/20240102noto_suzu_0102do/{z}/{x}/{y}.png | GSI tile list page (ichiran.html) |
| 20240102noto_wazimahigashi_0102do | post | 2024-01-02 | Wajima East | https://cyberjapandata.gsi.go.jp/xyz/20240102noto_wazimahigashi_0102do/{z}/{x}/{y}.png | GSI tile list page (ichiran.html) |
| 20240102noto_wazimanaka_0102do | post | 2024-01-02 | Wajima Central | https://cyberjapandata.gsi.go.jp/xyz/20240102noto_wazimanaka_0102do/{z}/{x}/{y}.png | GSI tile list page (ichiran.html) |
<!-- /AUTO -->

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
all 1), <!-- AUTO:pip_freeze -->
```
PyYAML==6.0.3
Pygments==2.21.0
aiohappyeyeballs==2.7.1
aiohttp==3.14.5
aiosignal==1.4.0
anyio==4.15.1
attrs==26.1.0
certifi==2026.10.19
charset-normalizer==3.5.2
click==8.5.0
cloudpickle==3.1.2
complexipy==8.1.0
datasets==5.1.0
dill==0.4.1
filelock==4.2.4
frozenlist==1.8.0
fsspec==2026.7.0
geopandas==1.2.0
h11==0.16.0
hf-xet==1.7.0
httpcore2==2.16.0
httpx2==2.16.0
huggingface_hub==2.2.0
idna==3.20
iniconfig==2.3.1
joblib==1.6.0
multidict==7.1.0
multiprocess==0.70.19
narwhals==2.27.1
numpy==2.5.4
packaging==26.3
pandas==3.1.0
pillow==13.0.0
pluggy==1.7.0
propcache==0.5.4
pyarrow==26.0.0
pyogrio==0.13.0
pyproj==3.8.0
pytest==9.1.1
python-dateutil==3.0.0
requests==2.34.2
ruff==0.17.2
scikit-learn==1.9.1
scipy==1.18.1
shapely==2.2.0
six==1.17.0
threadpoolctl==3.7.0
tqdm==4.70.1
truststore==0.10.5
typing_extensions==4.16.0
tzdata==2026.5
urllib3==2.8.0
xxhash==4.0.2
yarl==1.25.1
```
<!-- /AUTO -->
<!-- AUTO:seed_register -->
Default seeds: 1000, 1001, 1002, 1003, 1004. Folds: 5; audit seed 0.
<!-- /AUTO -->
How to rebuild: `make build`; how to verify: `notodata verify --clean`.
<!-- AUTO:verify_result --> date, mode, pass or fail per file <!-- /AUTO -->

## 12. Known limitations

Cloud or snow in January layers, automatic-orthophoto distortions,
seamlessphoto composite dates, equirectangular error over 60 km,
anything the audit marked `cannot_tell`.
