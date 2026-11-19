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
| 9 | Candidate embedding encoders | not fixed by the brief | `resnet50.tv_in1k_v2` (2048-d), `dinov2_vits14` (384-d), `openclip_vitb32_laion2b` (512-d), `satlas_aerial_swinb_si` (aerial-pretrained); chips resized to 224 px, bicubic. `dinov2_vits14` is the encoder built for the release (timm `vit_small_patch14_dinov2.lvd142m` created with `img_size=224`, so timm resamples the position embedding from its default 518, pooled features, ImageNet mean/std 0.485/0.456/0.406 and 0.229/0.224/0.225, PIL bicubic resize of the uint8 chip, no L2 normalisation, GPU build recorded as non-deterministic); the other three are registered only | Two general encoders, one language-aligned, one trained on aerial imagery; only DINOv2 is built, and its model name, `img_size` and preprocessing are written to the embedding sidecar, and the timm and torch versions to a separate provenance file | 2026-11-17 |
| 10 | Cognitive-complexity gate | not fixed by the brief | `complexipy --max-complexity-allowed 10` | A function above the threshold is refactored into smaller pieces, never exempted | 2026-10-23 |
| 11 | Cache sidecar key `building_ids_sha256` | the brief's sidecar keys: `chip_pipeline_version`, `layer_ids`, `zoom`, `chip_size`, `encoder_id`, `row_count` | the same keys plus `building_ids_sha256` (SHA-256 of the sorted building id list) | The same row count with different buildings must also force a recompute | 2026-10-23 |
| 12 | Published dataset repo and card | not fixed by the brief | Hugging Face dataset repo `AbbasABC/HiME-dataset`, created private by `notodata publish` (visibility is flipped by hand on the Hub); dataset card licence `cc-by-4.0`, `pretty_name` "HiME Noto 2024 building collapse", task category `image-classification`; only the small artifacts are uploaded unless `--include-imagery`; `audit/` and `key.csv` are never uploaded | The old HF dataset id was overridden (row 1), so the release goes to a new repo; the card licence follows the CC-BY-4.0 source dataset | 2026-11-18 |

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

<!-- AUTO:coverage_by_layer -->
| layer_id | group | value | n | ok | missing | partial | blank | error | coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seamlessphoto | overall | all | 140208 | 140208 | 0 | 0 | 0 | 0 | 1.0 |
| seamlessphoto | label_raw | 0 | 112275 | 112275 | 0 | 0 | 0 | 0 | 1.0 |
| seamlessphoto | label_raw | 1 | 3461 | 3461 | 0 | 0 | 0 | 0 | 1.0 |
| seamlessphoto | label_raw | 9 | 9437 | 9437 | 0 | 0 | 0 | 0 | 1.0 |
| seamlessphoto | label_raw | 99 | 15035 | 15035 | 0 | 0 | 0 | 0 | 1.0 |
| nendophoto2018 | overall | all | 140208 | 0 | 140208 | 0 | 0 | 0 | 0.0 |
| nendophoto2018 | label_raw | 0 | 112275 | 0 | 112275 | 0 | 0 | 0 | 0.0 |
| nendophoto2018 | label_raw | 1 | 3461 | 0 | 3461 | 0 | 0 | 0 | 0.0 |
| nendophoto2018 | label_raw | 9 | 9437 | 0 | 9437 | 0 | 0 | 0 | 0.0 |
| nendophoto2018 | label_raw | 99 | 15035 | 0 | 15035 | 0 | 0 | 0 | 0.0 |
| nendophoto2019 | overall | all | 140208 | 0 | 140208 | 0 | 0 | 0 | 0.0 |
| nendophoto2019 | label_raw | 0 | 112275 | 0 | 112275 | 0 | 0 | 0 | 0.0 |
| nendophoto2019 | label_raw | 1 | 3461 | 0 | 3461 | 0 | 0 | 0 | 0.0 |
| nendophoto2019 | label_raw | 9 | 9437 | 0 | 9437 | 0 | 0 | 0 | 0.0 |
| nendophoto2019 | label_raw | 99 | 15035 | 0 | 15035 | 0 | 0 | 0 | 0.0 |
| nendophoto2020 | overall | all | 140208 | 61686 | 77969 | 553 | 0 | 0 | 0.43996062992125984 |
| nendophoto2020 | label_raw | 0 | 112275 | 47683 | 64115 | 477 | 0 | 0 | 0.42469828545980853 |
| nendophoto2020 | label_raw | 1 | 3461 | 1499 | 1961 | 1 | 0 | 0 | 0.4331118173938168 |
| nendophoto2020 | label_raw | 9 | 9437 | 3589 | 5805 | 43 | 0 | 0 | 0.38031153968422166 |
| nendophoto2020 | label_raw | 99 | 15035 | 8915 | 6088 | 32 | 0 | 0 | 0.592949783837712 |
| nendophoto2021 | overall | all | 140208 | 33624 | 105578 | 1006 | 0 | 0 | 0.23981513180417666 |
| nendophoto2021 | label_raw | 0 | 112275 | 24279 | 87206 | 790 | 0 | 0 | 0.21624582498329994 |
| nendophoto2021 | label_raw | 1 | 3461 | 1449 | 1941 | 71 | 0 | 0 | 0.41866512568621783 |
| nendophoto2021 | label_raw | 9 | 9437 | 1955 | 7456 | 26 | 0 | 0 | 0.20716329341951892 |
| nendophoto2021 | label_raw | 99 | 15035 | 5941 | 8975 | 119 | 0 | 0 | 0.3951446624542734 |
| nendophoto2022 | overall | all | 140208 | 64568 | 74030 | 1610 | 0 | 0 | 0.4605158050895812 |
| nendophoto2022 | label_raw | 0 | 112275 | 53455 | 57446 | 1374 | 0 | 0 | 0.47610777109775104 |
| nendophoto2022 | label_raw | 1 | 3461 | 1585 | 1794 | 82 | 0 | 0 | 0.457960127130887 |
| nendophoto2022 | label_raw | 9 | 9437 | 3737 | 5613 | 87 | 0 | 0 | 0.3959944897742927 |
| nendophoto2022 | label_raw | 99 | 15035 | 5791 | 9177 | 67 | 0 | 0 | 0.3851679414699036 |
| 20240102noto_0405_0426do | overall | all | 140208 | 140208 | 0 | 0 | 0 | 0 | 1.0 |
| 20240102noto_0405_0426do | label_raw | 0 | 112275 | 112275 | 0 | 0 | 0 | 0 | 1.0 |
| 20240102noto_0405_0426do | label_raw | 1 | 3461 | 3461 | 0 | 0 | 0 | 0 | 1.0 |
| 20240102noto_0405_0426do | label_raw | 9 | 9437 | 9437 | 0 | 0 | 0 | 0 | 1.0 |
| 20240102noto_0405_0426do | label_raw | 99 | 15035 | 15035 | 0 | 0 | 0 | 0 | 1.0 |
| 20240102noto_wazimanishi_0117do | overall | all | 140208 | 10876 | 129219 | 113 | 0 | 0 | 0.07757046673513637 |
| 20240102noto_wazimanishi_0117do | label_raw | 0 | 112275 | 9494 | 102678 | 103 | 0 | 0 | 0.08456023157425963 |
| 20240102noto_wazimanishi_0117do | label_raw | 1 | 3461 | 62 | 3399 | 0 | 0 | 0 | 0.01791389771742271 |
| 20240102noto_wazimanishi_0117do | label_raw | 9 | 9437 | 282 | 9154 | 1 | 0 | 0 | 0.02988237787432447 |
| 20240102noto_wazimanishi_0117do | label_raw | 99 | 15035 | 1038 | 13988 | 9 | 0 | 0 | 0.06903890921183904 |
| 20240102noto_anamizu_0117do | overall | all | 140208 | 24401 | 115678 | 129 | 0 | 0 | 0.17403429190916353 |
| 20240102noto_anamizu_0117do | label_raw | 0 | 112275 | 17571 | 94604 | 100 | 0 | 0 | 0.156499665998664 |
| 20240102noto_anamizu_0117do | label_raw | 1 | 3461 | 240 | 3218 | 3 | 0 | 0 | 0.06934412019647501 |
| 20240102noto_anamizu_0117do | label_raw | 9 | 9437 | 3923 | 5511 | 3 | 0 | 0 | 0.4157041432658684 |
| 20240102noto_anamizu_0117do | label_raw | 99 | 15035 | 2667 | 12345 | 23 | 0 | 0 | 0.1773860991020951 |
| 20240102noto_nanao_0117do | overall | all | 140208 | 26752 | 113063 | 393 | 0 | 0 | 0.19080223667693713 |
| 20240102noto_nanao_0117do | label_raw | 0 | 112275 | 22016 | 89897 | 362 | 0 | 0 | 0.19608995769316412 |
| 20240102noto_nanao_0117do | label_raw | 1 | 3461 | 20 | 3440 | 1 | 0 | 0 | 0.005778676683039584 |
| 20240102noto_nanao_0117do | label_raw | 9 | 9437 | 3256 | 6178 | 3 | 0 | 0 | 0.34502490198156194 |
| 20240102noto_nanao_0117do | label_raw | 99 | 15035 | 1460 | 13548 | 27 | 0 | 0 | 0.09710675091453276 |
| 20240102noto_suzu_0114do | overall | all | 140208 | 9903 | 126858 | 3447 | 0 | 0 | 0.07063077713111948 |
| 20240102noto_suzu_0114do | label_raw | 0 | 112275 | 7601 | 101775 | 2899 | 0 | 0 | 0.06769984413270987 |
| 20240102noto_suzu_0114do | label_raw | 1 | 3461 | 786 | 2450 | 225 | 0 | 0 | 0.22710199364345565 |
| 20240102noto_suzu_0114do | label_raw | 9 | 9437 | 654 | 8783 | 0 | 0 | 0 | 0.0693016848574759 |
| 20240102noto_suzu_0114do | label_raw | 99 | 15035 | 862 | 13850 | 323 | 0 | 0 | 0.05733288992351181 |
| 20240102noto_wazimahigashi_0114do | overall | all | 140208 | 2191 | 137960 | 57 | 0 | 0 | 0.015626783065160334 |
| 20240102noto_wazimahigashi_0114do | label_raw | 0 | 112275 | 1255 | 110982 | 38 | 0 | 0 | 0.01117791137831218 |
| 20240102noto_wazimahigashi_0114do | label_raw | 1 | 3461 | 20 | 3441 | 0 | 0 | 0 | 0.005778676683039584 |
| 20240102noto_wazimahigashi_0114do | label_raw | 9 | 9437 | 40 | 9397 | 0 | 0 | 0 | 0.0042386351594786474 |
| 20240102noto_wazimahigashi_0114do | label_raw | 99 | 15035 | 876 | 14140 | 19 | 0 | 0 | 0.05826405054871965 |
| 20240102noto_anamizu_0114do | overall | all | 140208 | 14911 | 124917 | 380 | 0 | 0 | 0.10634913842291453 |
| 20240102noto_anamizu_0114do | label_raw | 0 | 112275 | 11550 | 100390 | 335 | 0 | 0 | 0.10287241148964596 |
| 20240102noto_anamizu_0114do | label_raw | 1 | 3461 | 158 | 3300 | 3 | 0 | 0 | 0.045651545796012716 |
| 20240102noto_anamizu_0114do | label_raw | 9 | 9437 | 1385 | 8049 | 3 | 0 | 0 | 0.14676274239694817 |
| 20240102noto_anamizu_0114do | label_raw | 99 | 15035 | 1818 | 13178 | 39 | 0 | 0 | 0.12091785833056203 |
| 20240102noto_wazimanaka_0111do | overall | all | 140208 | 13571 | 126167 | 470 | 0 | 0 | 0.09679190916352848 |
| 20240102noto_wazimanaka_0111do | label_raw | 0 | 112275 | 11283 | 100594 | 398 | 0 | 0 | 0.10049432197728791 |
| 20240102noto_wazimanaka_0111do | label_raw | 1 | 3461 | 697 | 2732 | 32 | 0 | 0 | 0.2013868824039295 |
| 20240102noto_wazimanaka_0111do | label_raw | 9 | 9437 | 589 | 8828 | 20 | 0 | 0 | 0.06241390272332309 |
| 20240102noto_wazimanaka_0111do | label_raw | 99 | 15035 | 1002 | 14013 | 20 | 0 | 0 | 0.06664449617559029 |
| 20240102noto_wazimanishi_0111do | overall | all | 140208 | 23312 | 116396 | 500 | 0 | 0 | 0.16626726007075202 |
| 20240102noto_wazimanishi_0111do | label_raw | 0 | 112275 | 18676 | 93140 | 459 | 0 | 0 | 0.1663415720329548 |
| 20240102noto_wazimanishi_0111do | label_raw | 1 | 3461 | 406 | 3044 | 11 | 0 | 0 | 0.11730713666570355 |
| 20240102noto_wazimanishi_0111do | label_raw | 9 | 9437 | 607 | 8822 | 8 | 0 | 0 | 0.06432128854508848 |
| 20240102noto_wazimanishi_0111do | label_raw | 99 | 15035 | 3623 | 11390 | 22 | 0 | 0 | 0.24097106750914532 |
| 20240102noto_anamizu_0111do | overall | all | 140208 | 8785 | 131326 | 97 | 0 | 0 | 0.06265690973410933 |
| 20240102noto_anamizu_0111do | label_raw | 0 | 112275 | 7892 | 104338 | 45 | 0 | 0 | 0.07029169450011133 |
| 20240102noto_anamizu_0111do | label_raw | 1 | 3461 | 20 | 3440 | 1 | 0 | 0 | 0.005778676683039584 |
| 20240102noto_anamizu_0111do | label_raw | 9 | 9437 | 222 | 9181 | 34 | 0 | 0 | 0.023524425135106497 |
| 20240102noto_anamizu_0111do | label_raw | 99 | 15035 | 651 | 14367 | 17 | 0 | 0 | 0.04329896907216495 |
| 20240102_noto_suzu_0105do | overall | all | 140208 | 8743 | 130685 | 5 | 775 | 0 | 0.062357354787173344 |
| 20240102_noto_suzu_0105do | label_raw | 0 | 112275 | 6934 | 104935 | 5 | 401 | 0 | 0.06175907370296148 |
| 20240102_noto_suzu_0105do | label_raw | 1 | 3461 | 674 | 2717 | 0 | 70 | 0 | 0.19474140421843397 |
| 20240102_noto_suzu_0105do | label_raw | 9 | 9437 | 381 | 8786 | 0 | 270 | 0 | 0.040372999894034124 |
| 20240102_noto_suzu_0105do | label_raw | 99 | 15035 | 754 | 14247 | 0 | 34 | 0 | 0.050149650814765544 |
| 20240102_noto_wazimanaka_0105do | overall | all | 140208 | 13984 | 126144 | 80 | 0 | 0 | 0.09973753280839895 |
| 20240102_noto_wazimanaka_0105do | label_raw | 0 | 112275 | 11633 | 100594 | 48 | 0 | 0 | 0.10361166778000445 |
| 20240102_noto_wazimanaka_0105do | label_raw | 1 | 3461 | 729 | 2732 | 0 | 0 | 0 | 0.21063276509679282 |
| 20240102_noto_wazimanaka_0105do | label_raw | 9 | 9437 | 603 | 8811 | 23 | 0 | 0 | 0.06389742502914061 |
| 20240102_noto_wazimanaka_0105do | label_raw | 99 | 15035 | 1019 | 14007 | 9 | 0 | 0 | 0.06777519122048553 |
| 20240102_noto_anamizu_0105do | overall | all | 140208 | 24284 | 115678 | 129 | 117 | 0 | 0.17319981741412757 |
| 20240102_noto_anamizu_0105do | label_raw | 0 | 112275 | 17530 | 94604 | 100 | 41 | 0 | 0.1561344912046315 |
| 20240102_noto_anamizu_0105do | label_raw | 1 | 3461 | 240 | 3218 | 3 | 0 | 0 | 0.06934412019647501 |
| 20240102_noto_anamizu_0105do | label_raw | 9 | 9437 | 3860 | 5511 | 3 | 63 | 0 | 0.4090282928896895 |
| 20240102_noto_anamizu_0105do | label_raw | 99 | 15035 | 2654 | 12345 | 23 | 13 | 0 | 0.1765214499501164 |
| 20240102_noto_nanao_0105do | overall | all | 140208 | 56895 | 83035 | 95 | 183 | 0 | 0.405789969188634 |
| 20240102_noto_nanao_0105do | label_raw | 0 | 112275 | 50048 | 62158 | 69 | 0 | 0 | 0.44576263638387886 |
| 20240102_noto_nanao_0105do | label_raw | 1 | 3461 | 130 | 3331 | 0 | 0 | 0 | 0.037561398439757296 |
| 20240102_noto_nanao_0105do | label_raw | 9 | 9437 | 3409 | 5826 | 19 | 183 | 0 | 0.36123768146656776 |
| 20240102_noto_nanao_0105do | label_raw | 99 | 15035 | 3308 | 11720 | 7 | 0 | 0 | 0.22001995344196873 |
| 20240102noto_suzu_0102do | overall | all | 140208 | 22074 | 118087 | 47 | 0 | 0 | 0.15743752139678194 |
| 20240102noto_suzu_0102do | label_raw | 0 | 112275 | 16897 | 95343 | 35 | 0 | 0 | 0.15049654865286127 |
| 20240102noto_suzu_0102do | label_raw | 1 | 3461 | 1761 | 1700 | 0 | 0 | 0 | 0.5088124819416354 |
| 20240102noto_suzu_0102do | label_raw | 9 | 9437 | 812 | 8618 | 7 | 0 | 0 | 0.08604429373741655 |
| 20240102noto_suzu_0102do | label_raw | 99 | 15035 | 2604 | 12426 | 5 | 0 | 0 | 0.1731958762886598 |
| 20240102noto_wazimahigashi_0102do | overall | all | 140208 | 10095 | 130009 | 104 | 0 | 0 | 0.07200017117425539 |
| 20240102noto_wazimahigashi_0102do | label_raw | 0 | 112275 | 5842 | 106385 | 48 | 0 | 0 | 0.05203295479848586 |
| 20240102noto_wazimahigashi_0102do | label_raw | 1 | 3461 | 319 | 3142 | 0 | 0 | 0 | 0.09216989309448137 |
| 20240102noto_wazimahigashi_0102do | label_raw | 9 | 9437 | 292 | 9122 | 23 | 0 | 0 | 0.03094203666419413 |
| 20240102noto_wazimahigashi_0102do | label_raw | 99 | 15035 | 3642 | 11360 | 33 | 0 | 0 | 0.24223478550049884 |
| 20240102noto_wazimanaka_0102do | overall | all | 140208 | 11958 | 128196 | 43 | 11 | 0 | 0.08528757274905854 |
| 20240102noto_wazimanaka_0102do | label_raw | 0 | 112275 | 9860 | 102366 | 41 | 8 | 0 | 0.08782008461367179 |
| 20240102noto_wazimanaka_0102do | label_raw | 1 | 3461 | 718 | 2743 | 0 | 0 | 0 | 0.20745449292112106 |
| 20240102noto_wazimanaka_0102do | label_raw | 9 | 9437 | 541 | 8893 | 0 | 3 | 0 | 0.05732754053194871 |
| 20240102noto_wazimanaka_0102do | label_raw | 99 | 15035 | 839 | 14194 | 2 | 0 | 0 | 0.05580312603924177 |
<!-- /AUTO -->

### 7.3 Blinded audit

Rater, date, procedure, then <!-- AUTO:audit -->
| layer_id | label_raw | mark | n |
| --- | --- | --- | --- |
| 20240102noto_0405_0426do | 0 | cannot_tell | 2 |
| 20240102noto_0405_0426do | 0 | no_visible_damage | 16 |
| 20240102noto_0405_0426do | 0 | visible_damage | 2 |
| 20240102noto_0405_0426do | 1 | cannot_tell | 2 |
| 20240102noto_0405_0426do | 1 | no_visible_damage | 9 |
| 20240102noto_0405_0426do | 1 | visible_damage | 9 |
| nendophoto2022 | 0 | cannot_tell | 2 |
| nendophoto2022 | 0 | no_visible_damage | 18 |
| nendophoto2022 | 1 | cannot_tell | 3 |
| nendophoto2022 | 1 | no_visible_damage | 17 |
| seamlessphoto | 0 | cannot_tell | 2 |
| seamlessphoto | 0 | no_visible_damage | 16 |
| seamlessphoto | 0 | visible_damage | 2 |
| seamlessphoto | 1 | cannot_tell | 2 |
| seamlessphoto | 1 | no_visible_damage | 7 |
| seamlessphoto | 1 | visible_damage | 11 |
<!-- /AUTO -->
Decision: n_seam = `<k>`/20, n_post = `<k>`/20 -> rule row `<r>`.
Post layer `<id>` (`<date>`), pre layer `<id or none>`, pre_available = `<bool>`.
Dropped by post coverage: <!-- AUTO:dropped -->
| label_raw | dropped |
| --- | --- |
<!-- /AUTO -->

### 7.4 Coverage per image pattern

<!-- AUTO:coverage_by_pattern -->
| group | value | pattern | count | share |
| --- | --- | --- | --- | --- |
| overall | all | IP | 0 | 0.0 |
| overall | all | I | 140208 | 1.0 |
| overall | all | none | 0 | 0.0 |
| label_raw | 0 | IP | 0 | 0.0 |
| label_raw | 0 | I | 112275 | 1.0 |
| label_raw | 0 | none | 0 | 0.0 |
| label_raw | 1 | IP | 0 | 0.0 |
| label_raw | 1 | I | 3461 | 1.0 |
| label_raw | 1 | none | 0 | 0.0 |
| label_raw | 9 | IP | 0 | 0.0 |
| label_raw | 9 | I | 9437 | 1.0 |
| label_raw | 9 | none | 0 | 0.0 |
| label_raw | 99 | IP | 0 | 0.0 |
| label_raw | 99 | I | 15035 | 1.0 |
| label_raw | 99 | none | 0 | 0.0 |
| y_collapse | 0 | IP | 0 | 0.0 |
| y_collapse | 0 | I | 112275 | 1.0 |
| y_collapse | 0 | none | 0 | 0.0 |
| y_collapse | 1 | IP | 0 | 0.0 |
| y_collapse | 1 | I | 3461 | 1.0 |
| y_collapse | 1 | none | 0 | 0.0 |
| y_collapse | unlabelled | IP | 0 | 0.0 |
| y_collapse | unlabelled | I | 24472 | 1.0 |
| y_collapse | unlabelled | none | 0 | 0.0 |
| chip_status | seamlessphoto | ok | 140208 |  |
| chip_status | seamlessphoto | missing | 0 |  |
| chip_status | seamlessphoto | partial | 0 |  |
| chip_status | seamlessphoto | blank | 0 |  |
| chip_status | seamlessphoto | error | 0 |  |
<!-- /AUTO -->

### 7.5 Chip definition

Zoom 18, 128 px, window maths, status rules, blank threshold, hash definition.

### 7.6 Tile fingerprints

<!-- AUTO:tile_fingerprints -->
| layer | tiles | sha256 |
| --- | --- | --- |
| seamlessphoto | 25612 | 11b716e8d45fd57b93af0ce68529712567bba8580ed3ca1b35577e7ac309cb80 |
| nendophoto2018 | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| nendophoto2019 | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| nendophoto2020 | 13410 | 192cc9b880a855f8fc19ead7e647cd515bd036dab152d0cfcc00c00d62a608bc |
| nendophoto2021 | 8242 | f9c3c140c4603eabccd3c9ac25286e07e008cc326a441175b10c4473e8ffc00b |
| nendophoto2022 | 11826 | d5e44c4c7c38a9089e3bb7220153b3e32162198ac8bf763eec9447d0580374e8 |
| 20240102noto_0405_0426do | 25612 | 3a12d0570b2a300c434f90ad9adb4850fa473de8ac9654a00e52f0666ad309a9 |
| 20240102noto_wazimanishi_0117do | 2323 | f830153e8c243b5f1829d624bf7897a23c03e4ed3588cd3e1bcf3b1be02ff272 |
| 20240102noto_anamizu_0117do | 5881 | 5c73e925362e04a8893bbff7adb453ea9a59c55b89752b72bca3a4ee86c682a9 |
| 20240102noto_nanao_0117do | 4812 | a3dea3f3f8dbba26c962d3f1f5cee4355a4060d4f2da2e9072c0ff8a4bf2b46e |
| 20240102noto_suzu_0114do | 1918 | 7e5238e81338d234fb929c615fc7b7675f0fed4088459216a3e7285a9b72cab8 |
| 20240102noto_wazimahigashi_0114do | 852 | 9c0295b3786ef9a5f10dc9c167e5f9f9cfb37851cab4f46932109aa6cbf270dd |
| 20240102noto_anamizu_0114do | 3413 | e14ab021d8229d82f6636b16d37ea819619ee35cc97bbee260e104ea9c434c46 |
| 20240102noto_wazimanaka_0111do | 2555 | c3755dac1c017eba657290404c12640cd41abe35037dd4edc17246e866dd696e |
| 20240102noto_wazimanishi_0111do | 5073 | 9998d98c58726e4fc5b1f89fbbf8e71027013090f1b2f7d1e822804bb5a62b07 |
| 20240102noto_anamizu_0111do | 1919 | b504c6582af437b2786c9406b17f6643561f31c78dfeddda58715085f4765764 |
| 20240102_noto_suzu_0105do | 1463 | 7e21d047efda5fdd0d71fc3bc757991558569730df65e02db250d5f520807bf8 |
| 20240102_noto_wazimanaka_0105do | 2557 | 60438b28ce0a596c04db53c90fbc662dd2004d9ccd0ac9d83bb1853e25f495fd |
| 20240102_noto_anamizu_0105do | 5881 | 1886c3e7b258fe61d9ca56a3c3bc788da1452ac273bfbf3dafa0a9fd94dadd12 |
| 20240102_noto_nanao_0105do | 9030 | 354f3faccc88987b432502fac89c1c114884855b294cd2db325a7ef7bf710119 |
| 20240102noto_suzu_0102do | 3758 | ec6aa863552f37638ad1f5540eab8e75c43f37b5cab81ad9ea2697152dfc8097 |
| 20240102noto_wazimahigashi_0102do | 2691 | 7e534e814b5752690df2e909037671c95587d3d96a6f6c387fc20ba1f4058c1f |
| 20240102noto_wazimanaka_0102do | 1667 | 714a28ff6f4246e3309181900a18de123c78784c580a2007098a0409a9eaf313 |
<!-- /AUTO -->

## 8. Spatial design

Projection reference (lat0, lon0, R).
<!-- AUTO:moran -->
| bin_lo_km | bin_hi_km | n | W | I | expected |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.5 | 20000 | 1817324 | 0.23284632845746353 | -5.000250012500625e-05 |
| 0.5 | 1.0 | 20000 | 3167872 | 0.07703184368966058 | -5.000250012500625e-05 |
| 1.0 | 2.0 | 20000 | 6979498 | 0.025330609149691777 | -5.000250012500625e-05 |
| 2.0 | 3.0 | 20000 | 6758540 | 0.03771859555527842 | -5.000250012500625e-05 |
| 3.0 | 5.0 | 20000 | 13025926 | 0.028540427921116846 | -5.000250012500625e-05 |
| 5.0 | 8.0 | 20000 | 23713044 | 0.030521281261571956 | -5.000250012500625e-05 |
| 8.0 | 12.0 | 20000 | 25845920 | 0.019988887470038685 | -5.000250012500625e-05 |
| 12.0 | 20.0 | 20000 | 61435310 | 0.017167436180888375 | -5.000250012500625e-05 |
<!-- /AUTO -->
L = `<km>` (rule case: `<first bin below 0.05 | clamped>`).
<!-- AUTO:folds -->
| fold | blocks | buildings | collapsed | survived | unlabelled | prevalence | G_I |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 19 | 14344 | 693 | 10544 | 3107 | 0.06167126457239477 | 14344 |
| 1 | 102 | 31466 | 692 | 24555 | 6219 | 0.027409197132332554 | 31466 |
| 2 | 116 | 31466 | 692 | 25279 | 5495 | 0.026645104154634013 | 31466 |
| 3 | 100 | 31466 | 692 | 26457 | 4317 | 0.02548896828612472 | 31466 |
| 4 | 116 | 31466 | 692 | 25440 | 5334 | 0.026480942905250267 | 31466 |
<!-- /AUTO -->
<!-- AUTO:splits_summary -->
| seed | role | buildings | collapsed | survived | unlabelled |
| --- | --- | --- | --- | --- | --- |
| 1000 | test | 14344 | 693 | 10544 | 3107 |
| 1000 | val | 31466 | 692 | 24555 | 6219 |
| 1000 | train_pool | 74265 | 1595 | 60483 | 12187 |
| 1000 | buffer_dropped | 20133 | 481 | 16693 | 2959 |
| 1001 | test | 31466 | 692 | 24555 | 6219 |
| 1001 | val | 31466 | 692 | 25279 | 5495 |
| 1001 | train_pool | 49825 | 1447 | 39926 | 8452 |
| 1001 | buffer_dropped | 27451 | 630 | 22515 | 4306 |
| 1002 | test | 31466 | 692 | 25279 | 5495 |
| 1002 | val | 31466 | 692 | 26457 | 4317 |
| 1002 | train_pool | 52908 | 1368 | 40087 | 11453 |
| 1002 | buffer_dropped | 24368 | 709 | 20452 | 3207 |
| 1003 | test | 31466 | 692 | 26457 | 4317 |
| 1003 | val | 31466 | 692 | 25440 | 5334 |
| 1003 | train_pool | 51546 | 1357 | 40446 | 9743 |
| 1003 | buffer_dropped | 25730 | 720 | 19932 | 5078 |
| 1004 | test | 31466 | 692 | 25440 | 5334 |
| 1004 | val | 14344 | 693 | 10544 | 3107 |
| 1004 | train_pool | 65877 | 1428 | 52585 | 11864 |
| 1004 | buffer_dropped | 28521 | 648 | 23706 | 4167 |
| 2000 | test | 14344 | 693 | 10544 | 3107 |
| 2000 | val | 31466 | 692 | 24555 | 6219 |
| 2000 | train_pool | 74265 | 1595 | 60483 | 12187 |
| 2000 | buffer_dropped | 20133 | 481 | 16693 | 2959 |
| 2001 | test | 31466 | 692 | 24555 | 6219 |
| 2001 | val | 31466 | 692 | 25279 | 5495 |
| 2001 | train_pool | 49825 | 1447 | 39926 | 8452 |
| 2001 | buffer_dropped | 27451 | 630 | 22515 | 4306 |
| 2002 | test | 31466 | 692 | 25279 | 5495 |
| 2002 | val | 31466 | 692 | 26457 | 4317 |
| 2002 | train_pool | 52908 | 1368 | 40087 | 11453 |
| 2002 | buffer_dropped | 24368 | 709 | 20452 | 3207 |
| 2003 | test | 31466 | 692 | 26457 | 4317 |
| 2003 | val | 31466 | 692 | 25440 | 5334 |
| 2003 | train_pool | 51546 | 1357 | 40446 | 9743 |
| 2003 | buffer_dropped | 25730 | 720 | 19932 | 5078 |
| 2004 | test | 31466 | 692 | 25440 | 5334 |
| 2004 | val | 14344 | 693 | 10544 | 3107 |
| 2004 | train_pool | 65877 | 1428 | 52585 | 11864 |
| 2004 | buffer_dropped | 28521 | 648 | 23706 | 4167 |
| 3000 | test | 14344 | 693 | 10544 | 3107 |
| 3000 | val | 31466 | 692 | 24555 | 6219 |
| 3000 | train_pool | 74265 | 1595 | 60483 | 12187 |
| 3000 | buffer_dropped | 20133 | 481 | 16693 | 2959 |
| 3001 | test | 31466 | 692 | 24555 | 6219 |
| 3001 | val | 31466 | 692 | 25279 | 5495 |
| 3001 | train_pool | 49825 | 1447 | 39926 | 8452 |
| 3001 | buffer_dropped | 27451 | 630 | 22515 | 4306 |
| 3002 | test | 31466 | 692 | 25279 | 5495 |
| 3002 | val | 31466 | 692 | 26457 | 4317 |
| 3002 | train_pool | 52908 | 1368 | 40087 | 11453 |
| 3002 | buffer_dropped | 24368 | 709 | 20452 | 3207 |
| 3003 | test | 31466 | 692 | 26457 | 4317 |
| 3003 | val | 31466 | 692 | 25440 | 5334 |
| 3003 | train_pool | 51546 | 1357 | 40446 | 9743 |
| 3003 | buffer_dropped | 25730 | 720 | 19932 | 5078 |
| 3004 | test | 31466 | 692 | 25440 | 5334 |
| 3004 | val | 14344 | 693 | 10544 | 3107 |
| 3004 | train_pool | 65877 | 1428 | 52585 | 11864 |
| 3004 | buffer_dropped | 28521 | 648 | 23706 | 4167 |
| 3005 | test | 14344 | 693 | 10544 | 3107 |
| 3005 | val | 31466 | 692 | 24555 | 6219 |
| 3005 | train_pool | 74265 | 1595 | 60483 | 12187 |
| 3005 | buffer_dropped | 20133 | 481 | 16693 | 2959 |
| 3006 | test | 31466 | 692 | 24555 | 6219 |
| 3006 | val | 31466 | 692 | 25279 | 5495 |
| 3006 | train_pool | 49825 | 1447 | 39926 | 8452 |
| 3006 | buffer_dropped | 27451 | 630 | 22515 | 4306 |
| 3007 | test | 31466 | 692 | 25279 | 5495 |
| 3007 | val | 31466 | 692 | 26457 | 4317 |
| 3007 | train_pool | 52908 | 1368 | 40087 | 11453 |
| 3007 | buffer_dropped | 24368 | 709 | 20452 | 3207 |
| 3008 | test | 31466 | 692 | 26457 | 4317 |
| 3008 | val | 31466 | 692 | 25440 | 5334 |
| 3008 | train_pool | 51546 | 1357 | 40446 | 9743 |
| 3008 | buffer_dropped | 25730 | 720 | 19932 | 5078 |
| 3009 | test | 31466 | 692 | 25440 | 5334 |
| 3009 | val | 14344 | 693 | 10544 | 3107 |
| 3009 | train_pool | 65877 | 1428 | 52585 | 11864 |
| 3009 | buffer_dropped | 28521 | 648 | 23706 | 4167 |
| 3010 | test | 14344 | 693 | 10544 | 3107 |
| 3010 | val | 31466 | 692 | 24555 | 6219 |
| 3010 | train_pool | 74265 | 1595 | 60483 | 12187 |
| 3010 | buffer_dropped | 20133 | 481 | 16693 | 2959 |
| 3011 | test | 31466 | 692 | 24555 | 6219 |
| 3011 | val | 31466 | 692 | 25279 | 5495 |
| 3011 | train_pool | 49825 | 1447 | 39926 | 8452 |
| 3011 | buffer_dropped | 27451 | 630 | 22515 | 4306 |
| 3012 | test | 31466 | 692 | 25279 | 5495 |
| 3012 | val | 31466 | 692 | 26457 | 4317 |
| 3012 | train_pool | 52908 | 1368 | 40087 | 11453 |
| 3012 | buffer_dropped | 24368 | 709 | 20452 | 3207 |
| 3013 | test | 31466 | 692 | 26457 | 4317 |
| 3013 | val | 31466 | 692 | 25440 | 5334 |
| 3013 | train_pool | 51546 | 1357 | 40446 | 9743 |
| 3013 | buffer_dropped | 25730 | 720 | 19932 | 5078 |
| 3014 | test | 31466 | 692 | 25440 | 5334 |
| 3014 | val | 14344 | 693 | 10544 | 3107 |
| 3014 | train_pool | 65877 | 1428 | 52585 | 11864 |
| 3014 | buffer_dropped | 28521 | 648 | 23706 | 4167 |
| 3015 | test | 14344 | 693 | 10544 | 3107 |
| 3015 | val | 31466 | 692 | 24555 | 6219 |
| 3015 | train_pool | 74265 | 1595 | 60483 | 12187 |
| 3015 | buffer_dropped | 20133 | 481 | 16693 | 2959 |
| 3016 | test | 31466 | 692 | 24555 | 6219 |
| 3016 | val | 31466 | 692 | 25279 | 5495 |
| 3016 | train_pool | 49825 | 1447 | 39926 | 8452 |
| 3016 | buffer_dropped | 27451 | 630 | 22515 | 4306 |
| 3017 | test | 31466 | 692 | 25279 | 5495 |
| 3017 | val | 31466 | 692 | 26457 | 4317 |
| 3017 | train_pool | 52908 | 1368 | 40087 | 11453 |
| 3017 | buffer_dropped | 24368 | 709 | 20452 | 3207 |
| 3018 | test | 31466 | 692 | 26457 | 4317 |
| 3018 | val | 31466 | 692 | 25440 | 5334 |
| 3018 | train_pool | 51546 | 1357 | 40446 | 9743 |
| 3018 | buffer_dropped | 25730 | 720 | 19932 | 5078 |
| 3019 | test | 31466 | 692 | 25440 | 5334 |
| 3019 | val | 14344 | 693 | 10544 | 3107 |
| 3019 | train_pool | 65877 | 1428 | 52585 | 11864 |
| 3019 | buffer_dropped | 28521 | 648 | 23706 | 4167 |
| 4000 | test | 14344 | 693 | 10544 | 3107 |
| 4000 | val | 31466 | 692 | 24555 | 6219 |
| 4000 | train_pool | 74265 | 1595 | 60483 | 12187 |
| 4000 | buffer_dropped | 20133 | 481 | 16693 | 2959 |
| 4001 | test | 31466 | 692 | 24555 | 6219 |
| 4001 | val | 31466 | 692 | 25279 | 5495 |
| 4001 | train_pool | 49825 | 1447 | 39926 | 8452 |
| 4001 | buffer_dropped | 27451 | 630 | 22515 | 4306 |
| 4002 | test | 31466 | 692 | 25279 | 5495 |
| 4002 | val | 31466 | 692 | 26457 | 4317 |
| 4002 | train_pool | 52908 | 1368 | 40087 | 11453 |
| 4002 | buffer_dropped | 24368 | 709 | 20452 | 3207 |
| 4003 | test | 31466 | 692 | 26457 | 4317 |
| 4003 | val | 31466 | 692 | 25440 | 5334 |
| 4003 | train_pool | 51546 | 1357 | 40446 | 9743 |
| 4003 | buffer_dropped | 25730 | 720 | 19932 | 5078 |
| 4004 | test | 31466 | 692 | 25440 | 5334 |
| 4004 | val | 14344 | 693 | 10544 | 3107 |
| 4004 | train_pool | 65877 | 1428 | 52585 | 11864 |
| 4004 | buffer_dropped | 28521 | 648 | 23706 | 4167 |
| 4005 | test | 14344 | 693 | 10544 | 3107 |
| 4005 | val | 31466 | 692 | 24555 | 6219 |
| 4005 | train_pool | 74265 | 1595 | 60483 | 12187 |
| 4005 | buffer_dropped | 20133 | 481 | 16693 | 2959 |
| 4006 | test | 31466 | 692 | 24555 | 6219 |
| 4006 | val | 31466 | 692 | 25279 | 5495 |
| 4006 | train_pool | 49825 | 1447 | 39926 | 8452 |
| 4006 | buffer_dropped | 27451 | 630 | 22515 | 4306 |
| 4007 | test | 31466 | 692 | 25279 | 5495 |
| 4007 | val | 31466 | 692 | 26457 | 4317 |
| 4007 | train_pool | 52908 | 1368 | 40087 | 11453 |
| 4007 | buffer_dropped | 24368 | 709 | 20452 | 3207 |
| 4008 | test | 31466 | 692 | 26457 | 4317 |
| 4008 | val | 31466 | 692 | 25440 | 5334 |
| 4008 | train_pool | 51546 | 1357 | 40446 | 9743 |
| 4008 | buffer_dropped | 25730 | 720 | 19932 | 5078 |
| 4009 | test | 31466 | 692 | 25440 | 5334 |
| 4009 | val | 14344 | 693 | 10544 | 3107 |
| 4009 | train_pool | 65877 | 1428 | 52585 | 11864 |
| 4009 | buffer_dropped | 28521 | 648 | 23706 | 4167 |
| 4010 | test | 14344 | 693 | 10544 | 3107 |
| 4010 | val | 31466 | 692 | 24555 | 6219 |
| 4010 | train_pool | 74265 | 1595 | 60483 | 12187 |
| 4010 | buffer_dropped | 20133 | 481 | 16693 | 2959 |
| 4011 | test | 31466 | 692 | 24555 | 6219 |
| 4011 | val | 31466 | 692 | 25279 | 5495 |
| 4011 | train_pool | 49825 | 1447 | 39926 | 8452 |
| 4011 | buffer_dropped | 27451 | 630 | 22515 | 4306 |
| 4012 | test | 31466 | 692 | 25279 | 5495 |
| 4012 | val | 31466 | 692 | 26457 | 4317 |
| 4012 | train_pool | 52908 | 1368 | 40087 | 11453 |
| 4012 | buffer_dropped | 24368 | 709 | 20452 | 3207 |
| 4013 | test | 31466 | 692 | 26457 | 4317 |
| 4013 | val | 31466 | 692 | 25440 | 5334 |
| 4013 | train_pool | 51546 | 1357 | 40446 | 9743 |
| 4013 | buffer_dropped | 25730 | 720 | 19932 | 5078 |
| 4014 | test | 31466 | 692 | 25440 | 5334 |
| 4014 | val | 14344 | 693 | 10544 | 3107 |
| 4014 | train_pool | 65877 | 1428 | 52585 | 11864 |
| 4014 | buffer_dropped | 28521 | 648 | 23706 | 4167 |
| 4015 | test | 14344 | 693 | 10544 | 3107 |
| 4015 | val | 31466 | 692 | 24555 | 6219 |
| 4015 | train_pool | 74265 | 1595 | 60483 | 12187 |
| 4015 | buffer_dropped | 20133 | 481 | 16693 | 2959 |
| 4016 | test | 31466 | 692 | 24555 | 6219 |
| 4016 | val | 31466 | 692 | 25279 | 5495 |
| 4016 | train_pool | 49825 | 1447 | 39926 | 8452 |
| 4016 | buffer_dropped | 27451 | 630 | 22515 | 4306 |
| 4017 | test | 31466 | 692 | 25279 | 5495 |
| 4017 | val | 31466 | 692 | 26457 | 4317 |
| 4017 | train_pool | 52908 | 1368 | 40087 | 11453 |
| 4017 | buffer_dropped | 24368 | 709 | 20452 | 3207 |
| 4018 | test | 31466 | 692 | 26457 | 4317 |
| 4018 | val | 31466 | 692 | 25440 | 5334 |
| 4018 | train_pool | 51546 | 1357 | 40446 | 9743 |
| 4018 | buffer_dropped | 25730 | 720 | 19932 | 5078 |
| 4019 | test | 31466 | 692 | 25440 | 5334 |
| 4019 | val | 14344 | 693 | 10544 | 3107 |
| 4019 | train_pool | 65877 | 1428 | 52585 | 11864 |
| 4019 | buffer_dropped | 28521 | 648 | 23706 | 4167 |
<!-- /AUTO -->
<!-- AUTO:clients_summary -->
| N | seed | clients_kept | clients_dropped | buildings_dropped | dropped_collapsed | dropped_survived | dropped_unlabelled | size_median | size_min | size_max | clients_zero_collapsed_calib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 1000 | 100 | 0 | 0 | 0 | 0 | 0 | 566.0 | 113 | 3422 | 63 |
| 100 | 1001 | 100 | 0 | 0 | 0 | 0 | 0 | 308.5 | 40 | 2995 | 69 |
| 100 | 1002 | 100 | 0 | 0 | 0 | 0 | 0 | 317.0 | 25 | 4019 | 67 |
| 100 | 1003 | 100 | 0 | 0 | 0 | 0 | 0 | 364.5 | 42 | 2686 | 67 |
| 100 | 1004 | 100 | 0 | 0 | 0 | 0 | 0 | 469.5 | 105 | 3612 | 67 |
| 100 | 2000 | 100 | 0 | 0 | 0 | 0 | 0 | 563.0 | 91 | 4863 | 62 |
| 100 | 2001 | 100 | 0 | 0 | 0 | 0 | 0 | 315.5 | 28 | 3855 | 69 |
| 100 | 2002 | 100 | 0 | 0 | 0 | 0 | 0 | 347.0 | 56 | 2929 | 66 |
| 100 | 2003 | 100 | 0 | 0 | 0 | 0 | 0 | 346.0 | 22 | 2560 | 70 |
| 100 | 2004 | 100 | 0 | 0 | 0 | 0 | 0 | 503.0 | 105 | 2609 | 67 |
| 100 | 3000 | 100 | 0 | 0 | 0 | 0 | 0 | 571.5 | 113 | 4861 | 64 |
| 100 | 3001 | 100 | 0 | 0 | 0 | 0 | 0 | 313.5 | 42 | 2995 | 69 |
| 100 | 3002 | 100 | 0 | 0 | 0 | 0 | 0 | 326.5 | 26 | 2923 | 66 |
| 100 | 3003 | 100 | 0 | 0 | 0 | 0 | 0 | 354.0 | 42 | 2585 | 70 |
| 100 | 3004 | 100 | 0 | 0 | 0 | 0 | 0 | 514.0 | 67 | 3530 | 65 |
| 100 | 3005 | 100 | 0 | 0 | 0 | 0 | 0 | 571.5 | 99 | 4830 | 64 |
| 100 | 3006 | 100 | 0 | 0 | 0 | 0 | 0 | 283.0 | 29 | 4567 | 71 |
| 100 | 3007 | 100 | 0 | 0 | 0 | 0 | 0 | 330.5 | 26 | 3707 | 67 |
| 100 | 3008 | 100 | 0 | 0 | 0 | 0 | 0 | 358.5 | 46 | 3083 | 70 |
| 100 | 3009 | 100 | 0 | 0 | 0 | 0 | 0 | 503.0 | 106 | 3125 | 64 |
| 100 | 3010 | 100 | 0 | 0 | 0 | 0 | 0 | 580.0 | 113 | 3434 | 65 |
| 100 | 3011 | 100 | 0 | 0 | 0 | 0 | 0 | 328.5 | 56 | 2995 | 69 |
| 100 | 3012 | 100 | 0 | 0 | 0 | 0 | 0 | 321.0 | 26 | 3547 | 66 |
| 100 | 3013 | 100 | 0 | 0 | 0 | 0 | 0 | 358.0 | 38 | 2567 | 70 |
| 100 | 3014 | 100 | 0 | 0 | 0 | 0 | 0 | 474.0 | 78 | 4912 | 66 |
| 100 | 3015 | 100 | 0 | 0 | 0 | 0 | 0 | 542.0 | 42 | 4898 | 62 |
| 100 | 3016 | 100 | 0 | 0 | 0 | 0 | 0 | 281.5 | 28 | 4567 | 71 |
| 100 | 3017 | 100 | 0 | 0 | 0 | 0 | 0 | 344.5 | 26 | 2942 | 64 |
| 100 | 3018 | 100 | 0 | 0 | 0 | 0 | 0 | 368.0 | 42 | 2607 | 68 |
| 100 | 3019 | 100 | 0 | 0 | 0 | 0 | 0 | 476.0 | 55 | 3346 | 67 |
| 100 | 4000 | 100 | 0 | 0 | 0 | 0 | 0 | 562.0 | 78 | 3267 | 64 |
| 100 | 4001 | 100 | 0 | 0 | 0 | 0 | 0 | 331.0 | 57 | 3029 | 69 |
| 100 | 4002 | 100 | 0 | 0 | 0 | 0 | 0 | 307.5 | 26 | 3628 | 66 |
| 100 | 4003 | 100 | 0 | 0 | 0 | 0 | 0 | 373.5 | 33 | 2560 | 69 |
| 100 | 4004 | 100 | 0 | 0 | 0 | 0 | 0 | 482.0 | 105 | 3764 | 65 |
| 100 | 4005 | 100 | 0 | 0 | 0 | 0 | 0 | 580.0 | 113 | 3162 | 61 |
| 100 | 4006 | 100 | 0 | 0 | 0 | 0 | 0 | 301.5 | 56 | 4553 | 69 |
| 100 | 4007 | 100 | 0 | 0 | 0 | 0 | 0 | 325.0 | 25 | 3733 | 66 |
| 100 | 4008 | 100 | 0 | 0 | 0 | 0 | 0 | 358.5 | 46 | 1825 | 68 |
| 100 | 4009 | 100 | 0 | 0 | 0 | 0 | 0 | 481.5 | 51 | 4916 | 65 |
| 100 | 4010 | 100 | 0 | 0 | 0 | 0 | 0 | 563.5 | 86 | 4858 | 63 |
| 100 | 4011 | 100 | 0 | 0 | 0 | 0 | 0 | 313.5 | 28 | 2994 | 68 |
| 100 | 4012 | 100 | 0 | 0 | 0 | 0 | 0 | 330.5 | 26 | 3255 | 69 |
| 100 | 4013 | 100 | 0 | 0 | 0 | 0 | 0 | 363.5 | 46 | 2513 | 69 |
| 100 | 4014 | 100 | 0 | 0 | 0 | 0 | 0 | 477.5 | 71 | 4854 | 68 |
| 100 | 4015 | 100 | 0 | 0 | 0 | 0 | 0 | 555.5 | 95 | 3107 | 61 |
| 100 | 4016 | 100 | 0 | 0 | 0 | 0 | 0 | 289.5 | 28 | 4623 | 70 |
| 100 | 4017 | 100 | 0 | 0 | 0 | 0 | 0 | 330.5 | 26 | 2555 | 64 |
| 100 | 4018 | 100 | 0 | 0 | 0 | 0 | 0 | 364.5 | 63 | 2574 | 67 |
| 100 | 4019 | 100 | 0 | 0 | 0 | 0 | 0 | 466.5 | 106 | 4930 | 68 |
| 200 | 1000 | 200 | 0 | 0 | 0 | 0 | 0 | 256.5 | 29 | 2078 | 148 |
| 200 | 1001 | 197 | 3 | 45 | 0 | 35 | 10 | 162.0 | 20 | 1471 | 151 |
| 200 | 1002 | 199 | 1 | 19 | 0 | 14 | 5 | 173.0 | 20 | 1556 | 153 |
| 200 | 1003 | 195 | 5 | 56 | 0 | 25 | 31 | 194.0 | 22 | 1568 | 148 |
| 200 | 1004 | 200 | 0 | 0 | 0 | 0 | 0 | 223.0 | 37 | 2105 | 156 |
| 200 | 2000 | 200 | 0 | 0 | 0 | 0 | 0 | 254.5 | 27 | 2070 | 149 |
| 200 | 2001 | 198 | 2 | 31 | 0 | 22 | 9 | 161.5 | 28 | 1630 | 156 |
| 200 | 2002 | 198 | 2 | 33 | 1 | 18 | 14 | 172.5 | 20 | 1536 | 154 |
| 200 | 2003 | 195 | 5 | 59 | 0 | 38 | 21 | 203.0 | 20 | 1311 | 151 |
| 200 | 2004 | 199 | 1 | 19 | 0 | 8 | 11 | 243.0 | 27 | 2026 | 157 |
| 200 | 3000 | 200 | 0 | 0 | 0 | 0 | 0 | 254.0 | 27 | 1906 | 148 |
| 200 | 3001 | 198 | 2 | 31 | 0 | 22 | 9 | 164.0 | 28 | 1499 | 157 |
| 200 | 3002 | 197 | 3 | 39 | 1 | 21 | 17 | 167.0 | 22 | 1515 | 152 |
| 200 | 3003 | 196 | 4 | 54 | 1 | 19 | 34 | 175.5 | 27 | 1681 | 153 |
| 200 | 3004 | 200 | 0 | 0 | 0 | 0 | 0 | 235.5 | 42 | 1828 | 157 |
| 200 | 3005 | 200 | 0 | 0 | 0 | 0 | 0 | 249.0 | 27 | 2195 | 150 |
| 200 | 3006 | 199 | 1 | 14 | 0 | 14 | 0 | 166.0 | 23 | 1306 | 157 |
| 200 | 3007 | 198 | 2 | 32 | 1 | 18 | 13 | 168.0 | 20 | 1472 | 154 |
| 200 | 3008 | 195 | 5 | 53 | 1 | 24 | 28 | 197.0 | 21 | 1407 | 151 |
| 200 | 3009 | 200 | 0 | 0 | 0 | 0 | 0 | 228.5 | 35 | 2026 | 157 |
| 200 | 3010 | 200 | 0 | 0 | 0 | 0 | 0 | 265.0 | 27 | 1828 | 152 |
| 200 | 3011 | 198 | 2 | 33 | 0 | 33 | 0 | 157.0 | 28 | 1637 | 155 |
| 200 | 3012 | 199 | 1 | 19 | 0 | 14 | 5 | 179.0 | 20 | 1695 | 154 |
| 200 | 3013 | 197 | 3 | 43 | 0 | 16 | 27 | 196.0 | 31 | 1305 | 150 |
| 200 | 3014 | 199 | 1 | 19 | 0 | 8 | 11 | 246.0 | 41 | 2029 | 154 |
| 200 | 3015 | 200 | 0 | 0 | 0 | 0 | 0 | 255.0 | 27 | 2190 | 148 |
| 200 | 3016 | 198 | 2 | 31 | 0 | 22 | 9 | 176.5 | 20 | 1591 | 154 |
| 200 | 3017 | 199 | 1 | 19 | 0 | 14 | 5 | 179.0 | 20 | 1700 | 156 |
| 200 | 3018 | 196 | 4 | 43 | 1 | 14 | 28 | 196.0 | 20 | 1481 | 151 |
| 200 | 3019 | 198 | 2 | 30 | 0 | 11 | 19 | 230.5 | 38 | 1752 | 157 |
| 200 | 4000 | 200 | 0 | 0 | 0 | 0 | 0 | 265.0 | 27 | 2164 | 148 |
| 200 | 4001 | 200 | 0 | 0 | 0 | 0 | 0 | 171.0 | 22 | 1297 | 156 |
| 200 | 4002 | 197 | 3 | 39 | 1 | 18 | 20 | 171.0 | 20 | 1412 | 152 |
| 200 | 4003 | 196 | 4 | 41 | 0 | 21 | 20 | 194.0 | 20 | 1681 | 154 |
| 200 | 4004 | 199 | 1 | 19 | 0 | 8 | 11 | 241.0 | 20 | 1831 | 156 |
| 200 | 4005 | 200 | 0 | 0 | 0 | 0 | 0 | 261.0 | 24 | 1947 | 149 |
| 200 | 4006 | 198 | 2 | 31 | 0 | 22 | 9 | 164.0 | 20 | 1480 | 155 |
| 200 | 4007 | 198 | 2 | 36 | 0 | 30 | 6 | 178.5 | 20 | 1699 | 155 |
| 200 | 4008 | 198 | 2 | 24 | 0 | 11 | 13 | 193.0 | 20 | 1170 | 152 |
| 200 | 4009 | 200 | 0 | 0 | 0 | 0 | 0 | 233.5 | 35 | 2187 | 159 |
| 200 | 4010 | 200 | 0 | 0 | 0 | 0 | 0 | 243.5 | 27 | 1876 | 148 |
| 200 | 4011 | 198 | 2 | 34 | 0 | 31 | 3 | 162.0 | 27 | 1403 | 157 |
| 200 | 4012 | 198 | 2 | 38 | 0 | 26 | 12 | 186.5 | 20 | 1296 | 157 |
| 200 | 4013 | 195 | 5 | 55 | 1 | 25 | 29 | 190.0 | 20 | 1438 | 151 |
| 200 | 4014 | 199 | 1 | 19 | 0 | 8 | 11 | 244.0 | 38 | 2040 | 155 |
| 200 | 4015 | 200 | 0 | 0 | 0 | 0 | 0 | 257.0 | 27 | 2018 | 147 |
| 200 | 4016 | 197 | 3 | 40 | 0 | 24 | 16 | 166.0 | 22 | 1420 | 155 |
| 200 | 4017 | 198 | 2 | 32 | 1 | 18 | 13 | 172.5 | 20 | 1623 | 152 |
| 200 | 4018 | 197 | 3 | 34 | 0 | 21 | 13 | 203.0 | 21 | 1371 | 150 |
| 200 | 4019 | 200 | 0 | 0 | 0 | 0 | 0 | 240.0 | 21 | 1893 | 158 |
| 50 | 1000 | 50 | 0 | 0 | 0 | 0 | 0 | 1030.5 | 361 | 9204 | 20 |
| 50 | 1001 | 50 | 0 | 0 | 0 | 0 | 0 | 648.0 | 153 | 4632 | 29 |
| 50 | 1002 | 50 | 0 | 0 | 0 | 0 | 0 | 722.0 | 122 | 4074 | 23 |
| 50 | 1003 | 50 | 0 | 0 | 0 | 0 | 0 | 619.5 | 147 | 4769 | 29 |
| 50 | 1004 | 50 | 0 | 0 | 0 | 0 | 0 | 965.0 | 250 | 6280 | 23 |
| 50 | 2000 | 50 | 0 | 0 | 0 | 0 | 0 | 1156.5 | 333 | 5091 | 22 |
| 50 | 2001 | 50 | 0 | 0 | 0 | 0 | 0 | 660.5 | 155 | 5425 | 26 |
| 50 | 2002 | 50 | 0 | 0 | 0 | 0 | 0 | 696.5 | 83 | 4684 | 25 |
| 50 | 2003 | 50 | 0 | 0 | 0 | 0 | 0 | 605.5 | 147 | 4768 | 27 |
| 50 | 2004 | 50 | 0 | 0 | 0 | 0 | 0 | 980.0 | 316 | 6207 | 22 |
| 50 | 3000 | 50 | 0 | 0 | 0 | 0 | 0 | 1276.5 | 333 | 7689 | 20 |
| 50 | 3001 | 50 | 0 | 0 | 0 | 0 | 0 | 643.0 | 153 | 4632 | 28 |
| 50 | 3002 | 50 | 0 | 0 | 0 | 0 | 0 | 733.5 | 168 | 6962 | 26 |
| 50 | 3003 | 50 | 0 | 0 | 0 | 0 | 0 | 723.5 | 147 | 4770 | 27 |
| 50 | 3004 | 50 | 0 | 0 | 0 | 0 | 0 | 1034.5 | 298 | 6075 | 23 |
| 50 | 3005 | 50 | 0 | 0 | 0 | 0 | 0 | 1074.0 | 315 | 6183 | 22 |
| 50 | 3006 | 50 | 0 | 0 | 0 | 0 | 0 | 632.0 | 94 | 4632 | 30 |
| 50 | 3007 | 50 | 0 | 0 | 0 | 0 | 0 | 734.0 | 155 | 5379 | 24 |
| 50 | 3008 | 50 | 0 | 0 | 0 | 0 | 0 | 638.5 | 143 | 4768 | 27 |
| 50 | 3009 | 50 | 0 | 0 | 0 | 0 | 0 | 997.0 | 153 | 6280 | 23 |
| 50 | 3010 | 50 | 0 | 0 | 0 | 0 | 0 | 1060.0 | 317 | 7955 | 21 |
| 50 | 3011 | 50 | 0 | 0 | 0 | 0 | 0 | 632.0 | 88 | 4632 | 26 |
| 50 | 3012 | 50 | 0 | 0 | 0 | 0 | 0 | 707.5 | 168 | 4204 | 23 |
| 50 | 3013 | 50 | 0 | 0 | 0 | 0 | 0 | 650.0 | 72 | 5187 | 26 |
| 50 | 3014 | 50 | 0 | 0 | 0 | 0 | 0 | 1029.0 | 250 | 6054 | 23 |
| 50 | 3015 | 50 | 0 | 0 | 0 | 0 | 0 | 1078.0 | 361 | 8100 | 21 |
| 50 | 3016 | 50 | 0 | 0 | 0 | 0 | 0 | 639.5 | 149 | 4632 | 27 |
| 50 | 3017 | 50 | 0 | 0 | 0 | 0 | 0 | 732.5 | 160 | 6146 | 24 |
| 50 | 3018 | 50 | 0 | 0 | 0 | 0 | 0 | 591.0 | 147 | 4747 | 28 |
| 50 | 3019 | 50 | 0 | 0 | 0 | 0 | 0 | 1038.0 | 274 | 6280 | 24 |
| 50 | 4000 | 50 | 0 | 0 | 0 | 0 | 0 | 1099.0 | 359 | 6458 | 21 |
| 50 | 4001 | 50 | 0 | 0 | 0 | 0 | 0 | 648.5 | 142 | 4632 | 29 |
| 50 | 4002 | 50 | 0 | 0 | 0 | 0 | 0 | 694.0 | 168 | 5720 | 25 |
| 50 | 4003 | 50 | 0 | 0 | 0 | 0 | 0 | 619.5 | 147 | 4658 | 26 |
| 50 | 4004 | 50 | 0 | 0 | 0 | 0 | 0 | 1052.0 | 316 | 6078 | 24 |
| 50 | 4005 | 50 | 0 | 0 | 0 | 0 | 0 | 1089.5 | 520 | 7163 | 22 |
| 50 | 4006 | 50 | 0 | 0 | 0 | 0 | 0 | 732.0 | 88 | 4834 | 26 |
| 50 | 4007 | 50 | 0 | 0 | 0 | 0 | 0 | 671.0 | 157 | 5346 | 24 |
| 50 | 4008 | 50 | 0 | 0 | 0 | 0 | 0 | 630.0 | 105 | 4902 | 26 |
| 50 | 4009 | 50 | 0 | 0 | 0 | 0 | 0 | 989.0 | 247 | 6280 | 23 |
| 50 | 4010 | 50 | 0 | 0 | 0 | 0 | 0 | 1211.5 | 305 | 6705 | 21 |
| 50 | 4011 | 50 | 0 | 0 | 0 | 0 | 0 | 652.5 | 88 | 4834 | 29 |
| 50 | 4012 | 50 | 0 | 0 | 0 | 0 | 0 | 724.0 | 188 | 4114 | 26 |
| 50 | 4013 | 50 | 0 | 0 | 0 | 0 | 0 | 628.5 | 142 | 6098 | 29 |
| 50 | 4014 | 50 | 0 | 0 | 0 | 0 | 0 | 973.0 | 316 | 6077 | 24 |
| 50 | 4015 | 50 | 0 | 0 | 0 | 0 | 0 | 1064.5 | 284 | 8989 | 21 |
| 50 | 4016 | 50 | 0 | 0 | 0 | 0 | 0 | 546.0 | 133 | 4632 | 30 |
| 50 | 4017 | 50 | 0 | 0 | 0 | 0 | 0 | 706.5 | 168 | 5344 | 27 |
| 50 | 4018 | 50 | 0 | 0 | 0 | 0 | 0 | 583.0 | 147 | 4776 | 28 |
| 50 | 4019 | 50 | 0 | 0 | 0 | 0 | 0 | 1069.0 | 355 | 6054 | 23 |
<!-- /AUTO -->

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
Jinja2==3.1.6
MarkupSafe==3.0.4
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
cuda-bindings==13.4.4
cuda-pathfinder==1.8.4
cuda-toolkit==13.0.3.0
datasets==5.1.0
dill==0.4.1
filelock==4.2.4
frozenlist==1.8.0
fsspec==2026.7.0
ftfy==6.3.1
geopandas==1.2.0
h11==0.16.0
hf-xet==1.7.0
httpcore2==2.16.0
httpx2==2.16.0
huggingface_hub==2.2.0
idna==3.20
iniconfig==2.3.1
joblib==1.6.0
mpmath==1.3.0
multidict==7.1.0
multiprocess==0.70.19
narwhals==2.27.1
networkx==3.7
notodata==0.1.0
numpy==2.5.4
nvidia-cublas==13.1.1.3
nvidia-cuda-cupti==13.0.85
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96
nvidia-cudnn-cu13==9.24.0.43
nvidia-cufft==12.0.0.61
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-cusolver==12.0.4.66
nvidia-cusparse==12.6.3.3
nvidia-cusparselt-cu13==0.8.1
nvidia-nccl-cu13==2.30.7
nvidia-nvjitlink==13.4.92
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
open_clip_torch==3.3.0
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
regex==2026.9.29
requests==2.34.2
ruff==0.17.2
safetensors==0.8.0
scikit-learn==1.9.1
scipy==1.18.1
setuptools==84.2.1
shapely==2.2.0
six==1.17.0
sympy==1.14.0
threadpoolctl==3.7.0
timm==1.0.30
torch==2.14.1
torchvision==0.29.1
tqdm==4.70.1
triton==3.8.0
truststore==0.10.5
typing_extensions==4.16.0
tzdata==2026.5
urllib3==2.8.0
wcwidth==0.9.3
xxhash==4.0.2
yarl==1.25.1
```
<!-- /AUTO -->
<!-- AUTO:seed_register -->
Default seeds: 1000, 1001, 1002, 1003, 1004. Folds: 5; audit seed 0.
<!-- /AUTO -->
How to rebuild: `make build`; how to verify: `notodata verify --clean`.
<!-- AUTO:verify_result -->
2026-11-19, mode `reuse-tiles`: **pass**

| file | result |
| --- | --- |
| chips/seamlessphoto/@tree | pass |
| chips/seamlessphoto/index.parquet | pass |
| derived/buildings.parquet | pass |
| derived/clients/N100_seed_1000.parquet | pass |
| derived/clients/N100_seed_1000_meta.parquet | pass |
| derived/clients/N100_seed_1001.parquet | pass |
| derived/clients/N100_seed_1001_meta.parquet | pass |
| derived/clients/N100_seed_1002.parquet | pass |
| derived/clients/N100_seed_1002_meta.parquet | pass |
| derived/clients/N100_seed_1003.parquet | pass |
| derived/clients/N100_seed_1003_meta.parquet | pass |
| derived/clients/N100_seed_1004.parquet | pass |
| derived/clients/N100_seed_1004_meta.parquet | pass |
| derived/clients/N200_seed_1000.parquet | pass |
| derived/clients/N200_seed_1000_meta.parquet | pass |
| derived/clients/N200_seed_1001.parquet | pass |
| derived/clients/N200_seed_1001_meta.parquet | pass |
| derived/clients/N200_seed_1002.parquet | pass |
| derived/clients/N200_seed_1002_meta.parquet | pass |
| derived/clients/N200_seed_1003.parquet | pass |
| derived/clients/N200_seed_1003_meta.parquet | pass |
| derived/clients/N200_seed_1004.parquet | pass |
| derived/clients/N200_seed_1004_meta.parquet | pass |
| derived/clients/N50_seed_1000.parquet | pass |
| derived/clients/N50_seed_1000_meta.parquet | pass |
| derived/clients/N50_seed_1001.parquet | pass |
| derived/clients/N50_seed_1001_meta.parquet | pass |
| derived/clients/N50_seed_1002.parquet | pass |
| derived/clients/N50_seed_1002_meta.parquet | pass |
| derived/clients/N50_seed_1003.parquet | pass |
| derived/clients/N50_seed_1003_meta.parquet | pass |
| derived/clients/N50_seed_1004.parquet | pass |
| derived/clients/N50_seed_1004_meta.parquet | pass |
| derived/splits/seed_1000.parquet | pass |
| derived/splits/seed_1001.parquet | pass |
| derived/splits/seed_1002.parquet | pass |
| derived/splits/seed_1003.parquet | pass |
| derived/splits/seed_1004.parquet | pass |
| reports/audit.csv | pass |
| reports/class_counts_mapped.csv | pass |
| reports/class_counts_raw.csv | pass |
| reports/clients_summary.csv | pass |
| reports/coverage_by_layer.csv | pass |
| reports/coverage_by_pattern.csv | pass |
| reports/dropped_by_post_coverage.csv | pass |
| reports/folds.csv | pass |
| reports/imagery_decision.json | pass |
| reports/moran.csv | pass |
| reports/source_checks.csv | pass |
| reports/splits_summary.csv | pass |
<!-- /AUTO -->

## 12. Known limitations

Cloud or snow in January layers, automatic-orthophoto distortions,
seamlessphoto composite dates, equirectangular error over 60 km,
anything the audit marked `cannot_tell`.
