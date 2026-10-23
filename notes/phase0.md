# Phase 0: reconnaissance

Date run: 2026-10-23. Worktree `build/notodata`, HEAD at the time of the run `358f002`.

Every claim below cites a file and line, a URL, or the output of a command that is
reproduced verbatim. Where evidence could not be obtained, the item says
**Blocked** and why; nothing is filled in from memory or plausibility.

Shell commands are shown as they are written for the repository
(`docker compose run --rm <service> …`). They were invoked from the worktree root.

## 0. Status at a glance

| Phase 0 item | Status | One-line result |
| --- | --- | --- |
| §1 Old repository quotes | **Blocked** | No old repository exists; nothing to quote |
| §2 D1 bug reproduction | **Resolved** | 6.2050% of buildings inside their chip, median offset 97.13 m |
| §3 New dataset recon | **Blocked / suspended** | `AbbasABC/HiME-dataset` is an empty repository; no card, no data, no schema to compare |
| §4 HiME design doc | **Resolved** | Doc read directly; all seven requirements in the build plan are confirmed against it |
| §5 GSI imagery registry | **Resolved** | All 17 post layers named on GSI's tile list and all 17 templates resolve; the 5 pre layers are a template, not 5 named layers |
| §6 Licence terms | **Resolved** | GSI attribution wording quoted; four encoder licences identified, one encoder id does not exist |

Two decisions are held open for the human at Stop 1: the code-99 branch and
amendment A1. Recommendations are in section 7; neither has been written into
`config.py` or the Deviations table.

## 1. Old repository (spec §1) — Blocked: no old repository available

**Blocked: no old repository available.** There is no `hflsim`/`uavbench`
repository to clone, so no commit hash, no file and no line number can be
recorded for any of the seven items the build plan lists. Nothing in this
document quotes that code, and no line numbers for it appear anywhere in this
file. The spec's Phase 0 exit criteria cannot be met for this item and it should
be struck from the plan rather than deferred.

For the record, these are the claims the build plan makes *about* that code. They
are **what the build plan claims, not independently verified** — the citation is
the build plan itself, not the code it describes:

> `src/hflsim/data/loader.py::_get_building_chip` (integer tile, 512 x 512 mosaic, centre crop);
> the `StandardScaler` fit near line 532;
> the `label` to `damage_val` copy and the code comment defining 9 and 99;
> the tile cache path `TILE_CACHE/{z}_{x}_{y}.jpg` and the black-chip fallback with its 50% warning;
> `src/uavbench/fl/features.py::compute_feature_cache` and its shape-only validation;
> the K-means client construction and the 80/10/10 per-client split;
> the client counts N in `configs/paper_full.yaml` (for comparison with the decided N = 50, 100, 200).

— `docs/spec/noto-hfl-dataset-build-plan.md` lines 199–205.

Consequences for later phases:

- The old class counts (`0: 104,917; 1: 3,407; 9: 8,365; 99: 12,154; total 128,843`,
  spec line 41) have no verifiable provenance. They stay a comparison column only,
  as ruling R10 already requires (`docs/plans/notodata-execution.md` line 150).
- The code comment that the spec says defines codes 9 and 99 (spec line 201) cannot
  be read. This is the *only* reason the Decisions table gives for the code-99
  branch being undecided — "The only evidence today is a code comment, which D4
  does not accept as a source" (spec line 648) — and that comment is now
  unreachable as well as inadmissible. See section 7.
- The D1 regression check planned for `tests/test_d1_chips.py` under ruling R3
  (`docs/plans/notodata-execution.md` line 139) cannot be written against the old
  source. It can still be written against the reimplementation in section 2, whose
  numbers are reproduced below; the test should cite this document, not the old repo.

Note on the execution plan: the task brief asks for rulings "R10, P8". `R10` exists
(`docs/plans/notodata-execution.md` line 150). There is no ruling `P8` in that file;
its rulings run R1–R12 and `grep -n "P8" docs/plans/notodata-execution.md` returns
nothing. A ruling `P6b` is referenced in the Docker-runtime section (line 104) but is
also not in the table.

## 2. D1 reproduction (spec §2) — Resolved

The build plan describes the old chip geometry well enough to reimplement without
the old source: "integer tile, 512 x 512 mosaic, centre crop" (spec line 199) and
"Reimplement the old tile math in a throwaway script (not in the package), draw
20,000 points in the bounding box with `default_rng(0)`, and report the share of
buildings inside their chip and the median offset in metres. The brief reports 6.6%
and 97 m" (spec line 206).

The script is not in version control. It was written to the git-ignored `data/`
directory, run inside the toolchain image, and deleted after its output was copied
here. Its full text follows so the measurement can be reproduced from this document
alone.

### 2.1 Script

```python
"""Throwaway Phase 0 script: reproduce the D1 chip-centring bug.

Reimplements the old pipeline's chip geometry as the build plan describes it
(docs/spec/noto-hfl-dataset-build-plan.md lines 199 and 206):
integer tile indices, a 512x512 mosaic, then a centre crop.

Old algorithm, as reimplemented here:
  1. Web-Mercator global pixel coordinates at zoom 18, 256 px tiles.
  2. Keep only the INTEGER tile indices tx = floor(gx / 256), ty = floor(gy / 256);
     the sub-tile pixel offset of the building is discarded (this is the bug).
  3. Build a 512x512 mosaic from the 2x2 tile block (tx, ty), (tx+1, ty),
     (tx, ty+1), (tx+1, ty+1). Mosaic pixel (0, 0) is global pixel
     (tx * 256, ty * 256).
  4. Centre-crop CHIP = 128 px: mosaic rows/cols 192..319 inclusive, i.e. global
     pixels [tx*256+192, tx*256+320) x [ty*256+192, ty*256+320). The crop's
     nominal centre is global pixel (tx*256 + 256, ty*256 + 256) -- the corner
     where the four mosaic tiles meet, not the building.

Reported: the share of buildings whose true location falls inside their own chip
window, and the median ground distance from the building to the chip's nominal
centre.
"""

import math

import numpy as np

ZOOM = 18
TILE = 256
CHIP = 128
MOSAIC = 512
R_EARTH = 6_371_008.8
LAT_MIN, LAT_MAX = 37.005, 37.526
LON_MIN, LON_MAX = 136.677, 137.356
N = 20_000
SEED = 0


def global_pixel(lat, lon, zoom=ZOOM):
    """Web-Mercator global pixel coordinates (float), 256 px tiles."""
    n = TILE * (2.0**zoom)
    gx = (lon + 180.0) / 360.0 * n
    lat_rad = np.radians(lat)
    gy = (1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / np.pi) / 2.0 * n
    return gx, gy


def metres_per_pixel(lat, zoom=ZOOM):
    """Ground resolution of one Web-Mercator pixel at this latitude."""
    return (2.0 * math.pi * R_EARTH * np.cos(np.radians(lat))) / (TILE * 2.0**zoom)


def main():
    rng = np.random.default_rng(SEED)
    lat = rng.uniform(LAT_MIN, LAT_MAX, N)
    lon = rng.uniform(LON_MIN, LON_MAX, N)

    gx, gy = global_pixel(lat, lon)

    # The bug: only the integer tile index survives.
    tx = np.floor(gx / TILE).astype(np.int64)
    ty = np.floor(gy / TILE).astype(np.int64)

    # Mosaic origin in global pixels, then the centre crop of the 512 mosaic.
    ox = tx * TILE
    oy = ty * TILE
    lo = (MOSAIC - CHIP) // 2  # 192
    x0, x1 = ox + lo, ox + lo + CHIP  # [+192, +320)
    y0, y1 = oy + lo, oy + lo + CHIP

    inside = (gx >= x0) & (gx < x1) & (gy >= y0) & (gy < y1)

    # Chip's nominal centre in global pixels.
    cx = ox + MOSAIC / 2.0
    cy = oy + MOSAIC / 2.0
    mpp = metres_per_pixel(lat)
    offset_m = np.hypot(gx - cx, gy - cy) * mpp

    print(f"seed                      : numpy.random.default_rng({SEED})")
    print(f"points                    : {N}")
    print(f"bbox                      : lat {LAT_MIN}-{LAT_MAX}, lon {LON_MIN}-{LON_MAX}")
    print(f"zoom / tile / mosaic / chip: {ZOOM} / {TILE} / {MOSAIC} / {CHIP}")
    print(f"metres per pixel (mean)   : {mpp.mean():.6f}")
    print(f"inside own chip           : {inside.sum()} / {N} = {100.0 * inside.mean():.4f}%")
    print(f"offset to nominal centre  : median {np.median(offset_m):.4f} m")
    print(f"                            mean   {offset_m.mean():.4f} m")
    print(f"                            p05    {np.percentile(offset_m, 5):.4f} m")
    print(f"                            p95    {np.percentile(offset_m, 95):.4f} m")
    print(f"                            max    {offset_m.max():.4f} m")

    # For reference: the correct, float global-pixel chip puts the building at
    # the chip centre by construction, so every building is inside its chip.
    fx0, fx1 = gx - CHIP / 2.0, gx + CHIP / 2.0
    fy0, fy1 = gy - CHIP / 2.0, gy + CHIP / 2.0
    ok = (gx >= fx0) & (gx < fx1) & (gy >= fy0) & (gy < fy1)
    print(f"new float-pixel control   : inside {100.0 * ok.mean():.4f}%, offset 0.0 m")


if __name__ == "__main__":
    main()
```

### 2.2 Command and output

```console
$ docker compose run --rm --no-deps shell python /work/data/d1_repro.py
seed                      : numpy.random.default_rng(0)
points                    : 20000
bbox                      : lat 37.005-37.526, lon 136.677-137.356
zoom / tile / mosaic / chip: 18 / 256 / 512 / 128
metres per pixel (mean)   : 0.474705
inside own chip           : 1241 / 20000 = 6.2050%
offset to nominal centre  : median 97.1289 m
                            mean   93.0889 m
                            p05    30.6591 m
                            p95    145.3365 m
                            max    171.7021 m
new float-pixel control   : inside 100.0000%, offset 0.0 m
```

### 2.3 What this establishes

- **Measured: 6.2050% inside, median offset 97.1289 m**, seed
  `numpy.random.default_rng(0)`, 20,000 points, bounding box lat 37.005–37.526 /
  lon 136.677–137.356, zoom 18, 256 px tiles, 512 px mosaic, 128 px centre crop.
- The brief's figures are 6.6% and 97 m — "The brief reports 6.6% and 97 m"
  (`docs/spec/noto-hfl-dataset-build-plan.md` line 206). The median offset matches
  to the metre. The inside share is 0.4
  percentage points below the brief's figure and is not forced to match.
- The 6.2% is the expected value of the bug, not an artefact of this
  reimplementation. Discarding the sub-tile offset puts the building uniformly
  anywhere in its 256 px tile, while the crop keeps only mosaic pixels 192–319,
  i.e. the last 64 px of that tile on each axis: (64/256)² = 1/16 = 6.25%. The
  measured 6.2050% is that value within sampling noise.
- Independent corroboration from the HiME design doc, section 2: "In the old
  pipeline about 93% of GSI chips missed their building" (doc
  `aa9ed0a8-71ae-438b-9ace-ac281061f4f9`, block `ms5hv9v0r3d.353`). 100 − 6.2050 =
  93.795%, which agrees with that figure.
- The control line shows the regression check this becomes: computing the chip
  window from the *float* global-pixel position puts every building inside its
  chip at zero offset. A new-code test asserting "building at chip pixel (64, 64)
  to within one pixel" therefore disagrees with the old maths on roughly 15 of any
  16 buildings, which is what ruling R3 wants to pin down.

## 3. New dataset recon (spec §3) — Blocked: the dataset repository is empty

Per the revision-2 override (spec lines 2–3) the source is
`AbbasABC/HiME-dataset`, not `AbbasABC/HFL-Dataset`. Access was confirmed and the
repository was enumerated. It contains no data and no card.

**Mid-task note.** After these facts were reported, the schema-comparison half of
this item was suspended pending a scope clarification: the project is *building*
this dataset rather than loading an existing one. The raw findings are kept here
because they are evidence; the comparison against the build plan's column
assumptions is marked blocked rather than guessed.

### 3.1 Token and identity

The token lives in `.env` at the worktree root, which is git-ignored
(`.gitignore` line 1) and is read by the `hf` service through `env_file`
(`docker-compose.yml`, the `hf` service). Its value is not reproduced here or in
any committed file.

```console
$ docker compose run --rm hf auth whoami
✓ Logged in
  user: AbbasABC
```

### 3.2 Repository info and the revision pin

```console
$ docker compose run --rm hf datasets info AbbasABC/HiME-dataset
{
  "id": "AbbasABC/HiME-dataset",
  "author": "AbbasABC",
  "created_at": "2026-10-23T07:40:45+00:00",
  "disabled": false,
  "downloads": 0,
  "gated": false,
  "last_modified": "2026-10-23T07:40:45+00:00",
  "likes": 0,
  "private": false,
  "sha": "2f9a51479aab33ab0da631363d28a2fa5167976f",
  "siblings": [
    {
      "rfilename": ".gitattributes"
    }
  ],
  "tags": [
    "region:us"
  ],
  "used_storage": 0
}
```

Current commit, resolved 2026-10-23: **`2f9a51479aab33ab0da631363d28a2fa5167976f`**.

This hash is recorded as the resolved revision *of an empty repository*. It is not
yet usable as the build pin: pinning to it would pin the pipeline to zero rows.
The pin should be re-resolved once the dataset has content, and only then written
into `config.py` and the Deviations table alongside the superseded
`AbbasABC/HFL-Dataset@6cf97c900445e080e61cb45e1aa72515d3ff1de8` (spec line 37,
ruling R10 at `docs/plans/notodata-execution.md` line 150).

### 3.3 File listing

```console
$ docker compose run --rm hf datasets list AbbasABC/HiME-dataset --tree --recursive
2504  2026-10-23 07:40:45  └── .gitattributes
```

One file, 2,504 bytes, the default LFS attributes file. `used_storage` is 0.
No parquet, no CSV, no GeoPackage, no images. Nothing was downloaded into
`data/raw/`, because there is nothing to download, so no per-file SHA-256 table and
no `hf cache verify` run can be recorded.

### 3.4 Dataset card

```console
$ docker compose run --rm hf datasets card AbbasABC/HiME-dataset --text
Error: File not found in dataset 'AbbasABC/HiME-dataset'.
URL: https://huggingface.co/datasets/AbbasABC/HiME-dataset/resolve/main/README.md
```

There is no `README.md`, therefore no dataset card.

### 3.5 Comparison against the build plan's assumptions — Blocked

Every item the spec asks to extract and quote from the card (spec lines 207–211)
is unobtainable, because the card does not exist:

| Item the spec requires | Status |
| --- | --- |
| Meaning of label codes 0, 1, 9, 99 (decides the D4 branch) | **Blocked**: no card and no document it cites |
| Definition of `MMI_shape` | **Blocked** |
| Units of `PGA`, `PGV`, `SA_0_3`, `SA_1_0`, `SA_3_0` | **Blocked** |
| Provenance of the labels and the shaking fields (e.g. a ShakeMap version) | **Blocked** |
| Dataset licence | **Blocked** |
| Citation requirement | **Blocked** |

Likewise, the schema comparison the brief asks for cannot be made: there are no
columns, no label codes and no rows to compare against `damage_val`/`label`, codes
{0, 1, 9, 99}, the seven seismic columns and `chip_path` (spec lines 39–40,
brief line 16). The honest statement of the difference is the largest one possible:
**the dataset the plan assumes exists does not exist yet.** Phases 1–2 and Tasks 3,
17 and 18 cannot run against this repository in its present state.

### 3.6 The old dataset repository, for comparison only

The predecessor repository does exist and is populated. This is recorded because it
is the only concrete evidence about the data's shape, and because the revision the
spec pins belongs to it. It is **not** the source for this build (spec line 3) and
nothing was downloaded from it.

```console
$ docker compose run --rm hf datasets info AbbasABC/HFL-Dataset
{
  "id": "AbbasABC/HFL-Dataset",
  "author": "AbbasABC",
  "created_at": "2026-06-01T10:41:25+00:00",
  "disabled": false,
  "downloads": 24,
  "gated": false,
  "last_modified": "2026-06-03T05:58:06+00:00",
  "likes": 0,
  "private": true,
  "sha": "6cf97c900445e080e61cb45e1aa72515d3ff1de8",
  ...
  "used_storage": 25459710317
}

$ docker compose run --rm hf datasets list AbbasABC/HFL-Dataset --tree
38  2026-06-03 05:49:27  ├── .gitattributes
                         ├── Final_Dataset/
                         └── Images/

$ docker compose run --rm hf datasets card AbbasABC/HFL-Dataset --text
Error: File not found in dataset 'AbbasABC/HFL-Dataset'.
URL: https://huggingface.co/datasets/AbbasABC/HFL-Dataset/resolve/main/README.md
```

Facts worth carrying forward:

- Its `sha` is `6cf97c900445e080e61cb45e1aa72515d3ff1de8`, exactly the revision the
  spec pins at line 37. That pin was therefore real, and it refers to this
  repository.
- It is **private**, 25,459,710,317 bytes (about 25.5 GB), and has 24 downloads.
- The per-building table is `Final_Dataset/Noto_Buildings_Seismic.gpkg` — a
  GeoPackage, not parquet. The spec's Phase 1 plan to load the snapshot with
  `datasets.load_dataset` (spec line 224) does not read GeoPackage; if the new
  dataset is built from this file, Phase 1 needs a vector reader, which is a
  schema-level deviation from the plan.
- `Images/` holds per-building GeoTIFFs under Japanese district directory names,
  e.g. `Images/石川県、能登町、字時長山口/building_101644_z18.tif`. The
  `_z18` suffix matches the zoom the brief fixes.
- It has **no `README.md` either**, so the "named source" that D4 requires for the
  code-99 meaning does not exist in the old repository any more than in the new one.

## 4. HiME design doc (spec §4) — Resolved

The doc is the artifact linked at `docs/spec/noto-hfl-dataset-build-plan.md`
line 63, id `aa9ed0a8-71ae-438b-9ace-ac281061f4f9`. It is not a web page and cannot
be fetched over HTTP; it is a living document, read through its own service. It was
read successfully, so this item is **not** blocked
and the build plan's quoted summary did not have to be used as a substitute. The
doc is "HiME: Algorithm Design and Justification", revision 49, dated 2026-10-15,
sixteen sections. Block ids below are quotable addresses within it.

### 4.1 The build plan's requirement list, checked against the doc

The list is at `docs/spec/noto-hfl-dataset-build-plan.md` lines 65–71.

| Build-plan requirement (spec line) | Confirmed in the doc? | Evidence |
| --- | --- | --- |
| Binary collapse labels, obstructed views unlabelled, scored by AUPRC on spatially held-out blocks (65) | Yes | §2, block `ms5hv9v0r3d.1232`: "Binary collapse detection per building, scored by AUPRC on spatially held-out blocks. Collapsed is 1, survived is 0, and obstructed views carry no collapse label." |
| Seven raw seismic features; HiME logs PGA, PGV and the three SA columns; the pipeline reports how many values are zero or negative and does not transform them (66) | Yes | §5, block `ms5hv9v0r3d.6664`: "The features z_k are log PGA, log PGV, log SA(0.3 s), log SA(1.0 s), log SA(3.0 s), MMI_original and MMI_shape." §5 table `.6142` gives E_S input as "7 seismic features". |
| A post chip and a pre chip per building where they exist, and the image pattern G in {IP, I, none}; E_S on all labelled rows, E_I on rows with a post chip, E_Δ on pattern IP only (67) | Yes, with one sharpening | §6, block `ms5hv9v0r3d.8860`: "For building b, G_b is IP if both chips exist, I if only the post chip exists, and none otherwise; a pre chip without a post chip counts as none. The seismic features exist for every building." §5, block `.8281`: "Each expert trains on the buildings where its inputs exist and the collapse label is defined. A building without a pre-event chip trains E_S and E_I only." |
| Frozen-encoder embeddings computed once per chip, cached per encoder id, several candidates, V0 chooses on the validation fold (68) | Yes | §5, block `ms5hv9v0r3d.7714`: "φ is a frozen pretrained encoder chosen by a central validation study before federated training (Section 13). Each client computes its embeddings once, so federated training never runs an image forward pass." §13, block `.36853`: "Pre-study V0, run centrally: chooses the encoder (Section 5)…" |
| Per-seed spatial folds, 500 m buffer, clients from the training pool only, 20% stratified calibration set per client (69) | Yes | §13, block `ms5hv9v0r3d.36068`: "Spatial block cross-validation: blocks sized from the label correlogram, 5 folds, and for seed s the test fold is s mod 5 and the validation fold (s + 1) mod 5. Training buildings within 500 m of a validation or test building are dropped. Clients hold training-pool buildings only." §11 algorithm, block `.31365`: "each client holds out 20% of its labelled buildings as calibration set V_i, stratified by label with a fixed seed; the other 80% train the experts". |
| Per-client label counts per image pattern, so the server can compute π and every π_m (70) | Yes | §4, block `ms5hv9v0r3d.13032`: "Expert m uses π_m, the prevalence among its own training rows… The server computes every π_m and π from label counts each client reports once". §11 step 3: "each client reports label counts once; the server computes pi and every pi_m". |
| A written meaning for `MMI_shape`; HiME leaves its monotonicity unconstrained until the dataset documentation defines it (71) | Yes, verbatim | §5, block `ms5hv9v0r3d.6664`: "The six intensity measures are all but MMI_shape, which stays unconstrained until the dataset documentation defines it." |

The build plan's summary of what HiME needs is accurate. No requirement in it is
overstated.

### 4.2 Requirements found in the doc that the build plan's phases do not cover

These go on the open-questions list, not into code (spec line 212).

1. **Coordinates must not become model features.** §2, block `ms5hv9v0r3d.353`:
   "In the old pipeline about 93% of GSI chips missed their building, the imagery
   date was unverified, and coordinates leaked location into the model." The build
   plan keeps lat/lon in the building table (correctly — geometry needs them) but
   never marks them as non-features. A `feature_columns` list, or a note in
   `DATASET.md` section 4, would stop the leak recurring downstream.
2. **The degeneracy screen needs prevalence per fold.** §13, block
   `ms5hv9v0r3d.36351`: "Primary metric: test AUPRC, averaged over the last 10
   rounds. Any run whose AUPRC is below 3 times the prevalence is flagged as
   degenerate before scoring." The plan reports overall class counts and per-layer
   coverage per raw code, but no per-fold prevalence. Phase 7's fold table should
   carry it.
3. **Per-client label counts must be broken out by image pattern, not just by
   label.** Spec line 70 says this, and the doc agrees, but the only report the
   plan specifies for clients is the calibration report (spec lines 674–675).
   A per-client × pattern × label count file is implied and not named.
4. **The calibration upload is per-row, with the image pattern attached.** §11
   step 20, block `ms5hv9v0r3d.31365`: "each client uploads, for each b in V_i:
   G_b, the expert outputs, y_b". So `G` must be stable per building across the
   whole pipeline, including after any post-coverage restriction. The plan computes
   `G` in Phase 4c, after the imagery decision, which satisfies this — but the
   ordering is load-bearing and is not called out as such anywhere.
5. **The fine-tuning escape hatch needs the chips kept, not only the embeddings.**
   §5, block `ms5hv9v0r3d.7923`: "A fine-tuned encoder is admitted only if it
   beats the best frozen candidate by more than 5% relative validation AUPRC."
   The plan caches embeddings and git-ignores chips (spec line 25), which is
   correct, but a rebuild for fine-tuning then needs the whole tile fetch again
   (about 12 hours, spec line 660). Worth stating as a known limitation.
6. **`MMI_shape` is a blocker for E_S, not a documentation nicety.** The doc leaves
   its monotonicity unconstrained "until the dataset documentation defines it", and
   §5 models E_S as "Additive monotone splines" (table `.6142`). With the card
   missing (section 3.4), this definition cannot be supplied by Phase 0 at all. It
   must come from whoever builds the dataset.

## 5. GSI imagery (spec §5) — Resolved

### 5.1 The GSI disaster page

`https://www.gsi.go.jp/BOUSAI/20240101_noto_earthquake.html` is not reachable by
direct HTTP from this environment (`curl` returns exit 000, no connection); it was
read through a fetching proxy instead. It confirms the April 2024 full-coverage
orthophoto set (flights 5–26 April 2024), January sets, GSI's automatic-processing
caveat and the cloud/snow caveat, and states that its published information may be
reused including reproduction provided the source is credited. It names no layer
ids or URL templates, so the registry evidence below comes from the tile-list page,
which does.

### 5.2 The tile list page, fetched directly

```console
$ curl -s -o ichiran.html -w "HTTP %{http_code} bytes=%{size_download}\n" \
    https://maps.gsi.go.jp/development/ichiran.html
HTTP 200 bytes=464296
```

Counting the distinct Noto layer ids in that file:

```console
$ grep -o '20240102[_]*noto[a-z0-9_]*' ichiran.html | sort -u | wc -l
17
$ grep -o '20240102[_]*noto[a-z0-9_]*' ichiran.html | sort -u
20240102noto_0405_0426do
20240102_noto_anamizu_0105do
20240102noto_anamizu_0111do
20240102noto_anamizu_0114do
20240102noto_anamizu_0117do
20240102_noto_nanao_0105do
20240102noto_nanao_0117do
20240102noto_suzu_0102do
20240102_noto_suzu_0105do
20240102noto_suzu_0114do
20240102noto_wazimahigashi_0102do
20240102noto_wazimahigashi_0114do
20240102noto_wazimanaka_0102do
20240102_noto_wazimanaka_0105do
20240102noto_wazimanaka_0111do
20240102noto_wazimanishi_0111do
20240102noto_wazimanishi_0117do
```

**Exactly 17 ids, and they are exactly the 17 in the Phase 5a registry table**
(spec lines 302–318), including the four 5 January ids with the extra underscore
after `20240102`. The registry's layer ids need no correction.

Two details in the spec's registry notes are confirmed verbatim:

- The index names the 17 January Nanao layer as a 7 January flight
  (spec line 320). The page's index reads
  `令和6年能登半島地震 七尾地区 正射画像（2024年1月7日撮影）`, while the layer id
  is `20240102noto_nanao_0117do`. The spec's instruction to use 17 January stands.
- GSI's automatic-processing caveat, from the detail entry for the April layer:
  `写真から自動処理したものなので、構造物等の歪み、不連続等が発生している箇所があります。`
  ("because it is processed automatically from photographs, there are places where
  distortion and discontinuity of structures occur").
- The cloud caveat, from the detail entry for `20240102noto_suzu_0102do` and the
  other January layers: `また、雲の影響により地上が見えにくい場合があります。`
  ("also, cloud may make the ground hard to see"). The January entries carry this
  sentence; the April entry does not.

The April layer's detail entry also gives its coverage and zoom range:

```
令和6年能登半島地震 能登地区 正射画像（2024年4月5日～4月26日撮影）
URL：https://cyberjapandata.gsi.go.jp/xyz/20240102noto_0405_0426do/{z}/{x}/{y}.png
ズームレベル  10～18
提供範囲  石川県珠洲市、輪島市、能登町、穴水町、七尾市、志賀町、中能登町、羽咋市、
          宝達志水町、津幡町、かほく市、内灘町、金沢市
          富山県氷見市、高岡市、射水市、小矢部市、富山市、砺波市、南砺市
提供開始  令和6年6月24日
```

That is every affected municipality in the bounding box plus more, which supports
the Decisions-table expectation that this layer wins the coverage rule
(spec line 654). By contrast `20240102noto_wazimanishi_0117do` lists
`石川県輪島市、穴水町、七尾市、志賀町` and `20240102noto_suzu_0102do` lists
`石川県珠洲市、輪島市` — single-district coverage, as the risk table says
(spec line 631).

### 5.3 The five pre-event layers are a template, not five named layers

This is a correction to the Decisions table. Spec line 653 says of the post layers
"GSI's tile list names exactly these", which is true. But amendment A1 registers
`nendophoto2018` to `nendophoto2022` as five layers (spec line 681), and the tile
list does **not** name five such layers. It carries one entry with a templated
year:

```console
$ grep -o 'nendophoto[0-9]*' ichiran.html | sort -u
nendophoto
nendophoto2018
```

The single occurrence of a year is the page's own sample thumbnail. The entry reads:

```
年度別空中写真（2007年以降）
URL：https://cyberjapandata.gsi.go.jp/xyz/nendophoto{year}/{z}/{x}/{y}.png
ズームレベル  14～18
提供開始  令和2年5月21日
備考  この地理院タイルは、電子国土基本図（オルソ画像）、森林（国有林・民有林）の
      空中写真、地方公共団体等の空中写真、簡易空中写真、人工衛星画像を撮影年度ごとに
      まとめたものです。
      URLの{year}は年度の4桁の数字（「2007」など）に置き換えてください。
      また、2023年度の一部において、米国航空宇宙局（NASA）/米国地質調査所（USGS）の
      地球観測衛星Landsat-8の画像、Axelspace CorporationのGRUS画像を使用しております。
```

Three consequences for A1:

1. The five pre candidates are a convention this project adopts over a templated
   family, not five layers GSI enumerates. The registry's `found_in` field should
   say so honestly.
2. `nendophoto{year}` is itself a composite — basic-map orthophotos, forest aerial
   photos, local-government aerial photos, simple aerial photos and satellite
   imagery, grouped by fiscal year. It is dated at the fiscal-year level but is not
   a single flight, so A1's claim that it makes the "before" image "provably before
   1 January 2024" holds only at year granularity.
3. A1 excludes fiscal 2023 because it may include post-event flights
   (spec line 681). GSI's note gives a second, independent reason: part of fiscal
   2023 is Landsat-8 and GRUS **satellite** imagery, which also carries an extra
   attribution requirement. The exclusion is better founded than A1 states.

### 5.4 Live template tests

Every one of the 17 post layers resolves. The test tile for each is the sample tile
GSI's own page embeds for that layer, which is guaranteed to be inside its
coverage:

```console
$ while read -r u; do curl -sI --max-time 25 "$u" -o /dev/null \
    -w "HTTP %{http_code} ct=%{content_type}\n"; done < sample_tiles.txt
https://cyberjapandata.gsi.go.jp/xyz/20240102_noto_anamizu_0105do/15/28859/12729.png        HTTP 200 ct=image/png
https://cyberjapandata.gsi.go.jp/xyz/20240102_noto_nanao_0105do/15/28841/12744.png          HTTP 200 ct=image/png
https://cyberjapandata.gsi.go.jp/xyz/20240102_noto_suzu_0105do/15/28886/12701.png           HTTP 200 ct=image/png
https://cyberjapandata.gsi.go.jp/xyz/20240102_noto_wazimanaka_0105do/15/28844/12708.png     HTTP 200 ct=image/png
https://cyberjapandata.gsi.go.jp/xyz/20240102noto_0405_0426do/11/1802/796.png               HTTP 200 ct=image/png
https://cyberjapandata.gsi.go.jp/xyz/20240102noto_anamizu_0111do/15/28862/12723.png         HTTP 200 ct=image/png
https://cyberjapandata.gsi.go.jp/xyz/20240102noto_anamizu_0114do/15/28870/12715.png         HTTP 200 ct=image/png
https://cyberjapandata.gsi.go.jp/xyz/20240102noto_anamizu_0117do/15/28876/12715.png         HTTP 200 ct=image/png
https://cyberjapandata.gsi.go.jp/xyz/20240102noto_nanao_0117do/15/28830/12750.png           HTTP 200 ct=image/png
https://cyberjapandata.gsi.go.jp/xyz/20240102noto_suzu_0102do/15/28877/12704.png            HTTP 200 ct=image/png
https://cyberjapandata.gsi.go.jp/xyz/20240102noto_suzu_0114do/15/28884/12699.png            HTTP 200 ct=image/png
https://cyberjapandata.gsi.go.jp/xyz/20240102noto_wazimahigashi_0102do/15/28856/12704.png   HTTP 200 ct=image/png
https://cyberjapandata.gsi.go.jp/xyz/20240102noto_wazimahigashi_0114do/15/28864/12698.png   HTTP 200 ct=image/png
https://cyberjapandata.gsi.go.jp/xyz/20240102noto_wazimanaka_0102do/15/28844/12709.png      HTTP 200 ct=image/png
https://cyberjapandata.gsi.go.jp/xyz/20240102noto_wazimanaka_0111do/15/28844/12708.png      HTTP 200 ct=image/png
https://cyberjapandata.gsi.go.jp/xyz/20240102noto_wazimanishi_0111do/15/28829/12723.png     HTTP 200 ct=image/png
https://cyberjapandata.gsi.go.jp/xyz/20240102noto_wazimanishi_0117do/15/28826/12737.png     HTTP 200 ct=image/png
```

One zoom-18 tile per role, at Wajima city centre (lat 37.3906, lon 136.8996 →
z18 tile x = 230759, y = 101677), fetched in full:

```console
baseline  https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/18/230759/101677.jpg             HTTP 200 ct=image/jpeg bytes=11977
pre       https://cyberjapandata.gsi.go.jp/xyz/nendophoto2018/18/230759/101677.png            HTTP 404 ct=application/xml bytes=310
pre       https://cyberjapandata.gsi.go.jp/xyz/nendophoto2019/18/230759/101677.png            HTTP 404 ct=application/xml bytes=330
pre       https://cyberjapandata.gsi.go.jp/xyz/nendophoto2020/18/230759/101677.png            HTTP 200 ct=image/png  bytes=126607
pre       https://cyberjapandata.gsi.go.jp/xyz/nendophoto2021/18/230759/101677.png            HTTP 404 ct=application/xml bytes=330
pre       https://cyberjapandata.gsi.go.jp/xyz/nendophoto2022/18/230759/101677.png            HTTP 404 ct=application/xml bytes=330
pre       https://cyberjapandata.gsi.go.jp/xyz/nendophoto2023/18/230759/101677.png            HTTP 404 ct=application/xml bytes=330
post      https://cyberjapandata.gsi.go.jp/xyz/20240102noto_0405_0426do/18/230759/101677.png  HTTP 200 ct=image/png  bytes=114691
```

Three things are established. The baseline template is live and serves JPEG. The
expected post layer is live at zoom 18 inside the bounding box and serves PNG. And
each `nendophoto` year is **sparse**: at this location only fiscal 2020 exists.
A 404 for a dated pre layer is normal, not a fault.

### 5.5 Indicative pre-layer coverage probe

Because that sparseness decides amendment A1, one zoom-18 tile was tested per
fiscal year at six settlements spread across the box. **This is an indication, not
a coverage measurement** — six points, not 128,843 buildings. Real coverage is
Task 20's job.

```console
$ while IFS=$'\t' read -r town yr u; do printf "%-9s %s " "$town" "$yr"; \
    curl -sI --max-time 25 "$u" -o /dev/null -w "HTTP %{http_code}\n"; done < nendo_probe.txt
Wajima    2018 HTTP 404      Nanao     2018 HTTP 404
Wajima    2019 HTTP 404      Nanao     2019 HTTP 404
Wajima    2020 HTTP 200      Nanao     2020 HTTP 404
Wajima    2021 HTTP 404      Nanao     2021 HTTP 404
Wajima    2022 HTTP 404      Nanao     2022 HTTP 404
Wajima    2023 HTTP 404      Nanao     2023 HTTP 404
Suzu      2018 HTTP 404      Noto-cho  2018 HTTP 404
Suzu      2019 HTTP 404      Noto-cho  2019 HTTP 404
Suzu      2020 HTTP 404      Noto-cho  2020 HTTP 404
Suzu      2021 HTTP 200      Noto-cho  2021 HTTP 200
Suzu      2022 HTTP 200      Noto-cho  2022 HTTP 200
Suzu      2023 HTTP 404      Noto-cho  2023 HTTP 404
Anamizu   2018 HTTP 404      Shika     2018 HTTP 404
Anamizu   2019 HTTP 404      Shika     2019 HTTP 404
Anamizu   2020 HTTP 200      Shika     2020 HTTP 404
Anamizu   2021 HTTP 404      Shika     2021 HTTP 404
Anamizu   2022 HTTP 200      Shika     2022 HTTP 404
Anamizu   2023 HTTP 404      Shika     2023 HTTP 404
```

Settlement coordinates used: Wajima 37.3906/136.8996, Suzu 37.4361/137.2606,
Anamizu 37.2300/136.9100, Nanao 37.0430/136.9674, Noto-cho 37.3057/137.1485,
Shika 37.0060/136.7790.

Read-out, per fiscal year, of the six probe points:

| Year | Points with a tile | Which |
| --- | --- | --- |
| 2018 | 0 of 6 | — |
| 2019 | 0 of 6 | — |
| 2020 | 2 of 6 | Wajima, Anamizu |
| 2021 | 2 of 6 | Suzu, Noto-cho |
| 2022 | 3 of 6 | Suzu, Anamizu, Noto-cho |
| 2023 | 0 of 6 | — (excluded by A1 anyway) |
| Union 2018–2022 | 4 of 6 | Nanao and Shika have no pre tile in any year |

No single fiscal year reaches more than 3 of 6 points, and the best year, 2022,
misses Wajima — the district with the heaviest collapse. A1 requires the chosen
dated layer to cover "at least 80% of kept buildings" (spec line 681). On this
evidence that threshold is very unlikely to be met by any single year, so A1's own
fallback ("otherwise the brief's rule stands unchanged") would fire. See section 7.

## 6. Licence terms (spec §6) — Resolved

### 6.1 Which GSI category the layers fall into

The tile list page sorts tiles into three classes. Both the Noto orthophoto group
and `年度別空中写真` (the `nendophoto` family) are listed under class 2,
`基本測量成果以外で出典の記載のみで利用可能なもの` — "usable with a source credit
alone, other than basic survey results". So no application under the Survey Act is
needed; attribution is the whole requirement, plus any per-layer extra noted in
that layer's 備考.

### 6.2 The attribution wording, quoted

From `ichiran.html`, the page's own statement of terms:

> 地理院タイルは、国土地理院コンテンツ利用規約に従って利用できます。

("GSI tiles may be used in accordance with the GSI Content Use Terms.")

Class 2, in full:

> 基本測量成果以外で出典の記載のみで利用可能なものは、出典を記載いただくことで、申請なくご利用いただけます。出典は「国土地理院」または「地理院タイル」等と記載してください。ただし、タイルによっては「国土地理院」または「地理院タイル」等の出所明示に加えて、個別の出所の明示が必要な場合があります。詳細は各情報の備考をご覧ください。
>
> また、1.と同様に、地理院タイルをウェブサイトやソフトウェア、アプリケーション上でリアルタイムに読み込んで利用する場合、地理院タイルは出典の明示のみで申請不要でご利用いただけます。出典は、「国土地理院」または「地理院タイル」等と記載していただき、地理院タイル一覧ページ（https://maps.gsi.go.jp/development/ichiran.html）へのリンクを付けてください。

In English: tiles in this class may be used without any application provided the
source is stated; state the source as 「国土地理院」 or 「地理院タイル」; some tiles
additionally require their own individual source statement, given in that tile's
remarks; and when tiles are loaded in real time by a website, software or
application, a credit of 「国土地理院」 or 「地理院タイル」 plus a link to the tile
list page is sufficient and no application is needed.

Also noted on the same page: 「地理院地図」・「地理院タイル」は、国土地理院の登録商標です
(both names are registered trademarks of GSI).

The content-use terms themselves
(`https://www.gsi.go.jp/kikakuchousei/kikakuchousei40182.html`, also unreachable by
direct HTTP from here and read through a proxy) require the source to be stated,
give the model form `出典：国土地理院ウェブサイト （当該ページのURL）`, and require
that any editing or processing be stated too, with the model form
`地理院タイル（…）を加工して作成` for derivative works.

### 6.3 What this project must do

- `DATASET.md` section 10 carries the credit 「国土地理院」 (or 「地理院タイル」) with
  a link to `https://maps.gsi.go.jp/development/ichiran.html`, as spec line 320
  already requires.
- Because chips are cut from the tiles, the derivative-work form applies: the credit
  should say the chips were made by processing GSI tiles
  (`地理院タイルを加工して作成`), not merely that GSI is the source.
- The project still redistributes nothing: tiles are fetched at build time, cached
  under the git-ignored `data/` tree, and never committed (spec line 25,
  `.gitignore` line 2). This keeps the project clear of the per-layer extra
  attribution cases entirely, and is the reason fiscal 2023 `nendophoto` (which
  contains GRUS satellite imagery with its own attribution clause) would be a
  licensing complication as well as a dating one.

### 6.4 Candidate encoder licences

Spec line 677 names four candidates. A licence check, as asked — not a legal review.

| Candidate as the spec names it | Real identifier | Licence | Note |
| --- | --- | --- | --- |
| `resnet50.tv_in1k_v2` | **does not exist** | — | `https://huggingface.co/timm/resnet50.tv_in1k_v2` → HTTP 401 (no such repo). The two real timm repos are `resnet50.tv_in1k` and `resnet50.tv2_in1k`, both HTTP 200. "V2 torchvision weights" is `resnet50.tv2_in1k`. The spec's id must be corrected before Task 15 registers it. |
| `resnet50.tv_in1k` / `resnet50.tv2_in1k` | `timm/resnet50.tv_in1k`, `timm/resnet50.tv2_in1k` | BSD-3-Clause (weights originate from `github.com/pytorch/vision`) | Permissive; attribution only |
| `dinov2_vits14` | `facebookresearch/dinov2` | Apache-2.0 (code and weights) | Permissive |
| `openclip_vitb32_laion2b` | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`, loaded in open_clip as `ViT-B-32` + `laion2b_s34b_b79k` | MIT | Permissive |
| `satlas_aerial_swinb_si` | `allenai/satlas-pretrain` | **Weights ODC-BY**, repository code Apache-2.0 | ODC-BY is an attribution licence for data, not the same shape as the others; it is still permissive but requires attribution of the SatlasPretrain source wherever the embeddings are published. Flag for Task 15's registry notes. |

All four are usable. Two items for the record: the first id is wrong, and the
aerial-pretrained one is the only non-standard licence of the set.

## 7. Stop 1: findings and recommendations (not decided here)

Spec line 621: "After Phase 0: confirm the code-99 branch against its quoted
evidence, and approve or reject amendment A1."

Neither is decided in this document. `config.CODE99_MEANING` and the Deviations
table are untouched, and silence is not treated as approval.

### 7.1 Code-99 branch — recommendation: keep `unknown`

The D4 rule for what counts as evidence (spec line 237): "A named source is the
dataset card or a document it cites, stating the meaning in quotable words; a code
comment, a column name or the old repo does not count. The default is `unknown`
until a named source says otherwise."

What Phase 0 found:

- `AbbasABC/HiME-dataset` has no `README.md`, so there is no card and no document
  for a card to cite (section 3.4).
- `AbbasABC/HFL-Dataset` has no `README.md` either (section 3.6).
- There is no old repository, so even the inadmissible code comment is unreachable
  (section 1).
- Nothing in the HiME design doc defines the raw codes; it speaks only of collapsed
  = 1, survived = 0 and "obstructed views carry no collapse label" (section 4.1).

**No quotable named source exists. The recommendation is that `CODE99_MEANING`
stays `unknown`**, which means code 99 maps to `y_collapse = NaN`,
`observable = NaN`, and `y_collapse_sens = 1` (spec line 247). This is the
spec's own default, so confirming it requires no deviation row.

What the human needs to decide: confirm `unknown`, or supply a named source. If a
source is supplied later, note that the branch must be fixed before Phase 7 and
never changed afterwards without rebuilding everything downstream (spec line 253).

### 7.2 Amendment A1 — recommendation: approve the registry, expect the fallback to fire

A1 (spec lines 681–683) registers `nendophoto2018`–`nendophoto2022` as pre-event
candidates, takes the pre layer to be the dated layer with the highest coverage
(ties to the most recent year) provided it covers at least 80% of kept buildings,
otherwise leaves the brief's rule unchanged, and adds that layer to the audit
(40 buildings × 3 layers = 120 images).

Evidence bearing on it:

- The reasoning behind A1 is sound and is confirmed by GSI's own text.
  `seamlessphoto` genuinely has no per-pixel date, and `nendophoto{year}` genuinely
  is dated by fiscal year (section 5.3). The A1 exclusion of fiscal 2023 is
  correct and has a second justification GSI supplies (satellite imagery in part of
  that year).
- But the five layers are a template, not five layers GSI names (section 5.3), and
  each year is sparse. On a six-point probe, no single year covers more than 3 of 6
  settlements and the best year misses Wajima; two of the six settlements have no
  pre tile in any year 2018–2022 (section 5.5).
- So the 80% threshold is unlikely to be met by any single dated layer, and A1's own
  fallback clause would then apply, returning the brief's pre rule unchanged.

**Recommendation: approve A1 as written.** It costs little — the registry entries
and a third audit layer — and its threshold protects the pipeline from a bad pre
layer automatically. It should be approved *with the expectation recorded* that the
80% test will probably fail and the fallback will fire, so nobody later reads a
failed A1 as a pipeline fault. If A1 is approved, the audit is 120 images; if
rejected, the paragraph is deleted and the audit returns to 80 images
(40 buildings × 2 layers), per spec line 683.

One wrinkle the human should be aware of either way: ruling R4
(`docs/plans/notodata-execution.md` line 141) already notes that A1's rejection
clause ("if 3 or more of the 20 collapsed buildings show visible damage in it, it is
rejected for the next dated layer") assumes a second dated layer was audited, which
it was not. That ambiguity is unchanged by Phase 0.

### 7.3 A third matter for the same checkpoint

Not part of Stop 1 as the spec wrote it, but it blocks more than Stop 1 does: the
source dataset is empty (section 3). Phases 1–2 cannot run, the revision pin cannot
be fixed to anything meaningful, and the card-derived answers D4 needs do not exist.
Whoever signs off Stop 1 should say where the per-building rows, coordinates, labels
and seismic features are to come from, and whether this project is loading a dataset
or building one.

## 8. Open questions list

1. Where does the per-building data come from, now that `AbbasABC/HiME-dataset` is
   empty? (section 3)
2. If it comes from `Final_Dataset/Noto_Buildings_Seismic.gpkg`, Phase 1's
   `datasets.load_dataset` path does not read GeoPackage. Which reader? (section 3.6)
3. Who defines `MMI_shape`, and in what document, given that there is no card for
   D4 or HiME to cite? (sections 3.5, 4.2)
4. Same question for the units of PGA, PGV and the three SA columns, the label
   provenance, the shaking provenance, the licence and the citation requirement.
   (section 3.5)
5. The old-repository items in Phase 0 §1 are permanently unobtainable. Strike them
   from the plan, or restate them as claims about a system that is no longer
   available? (section 1)
6. `resnet50.tv_in1k_v2` does not exist. Replace with `resnet50.tv2_in1k`?
   (section 6.4)
7. SatlasPretrain weights are ODC-BY, not Apache/MIT/BSD like the other three.
   Acceptable for the embedding cache? (section 6.4)
8. The pre-layer registry is a template family, not five GSI-named layers. Should
   the Decisions table's "GSI's tile list names exactly these" be narrowed to the
   post layers only? (section 5.3)
9. Should lat/lon be marked explicitly as non-features, given that the HiME doc
   names coordinate leakage as a defect of the old pipeline? (section 4.2, item 1)
10. Per-fold prevalence for HiME's degeneracy screen, and per-client counts broken
    out by image pattern — add to Phase 7 and Phase 9 reports? (section 4.2,
    items 2 and 3)
11. There is no ruling `P8` in the execution plan, though the Task 2 brief cites
    one; `P6b` is referenced in the text but also absent from the rulings table.
    (section 1)

**STOP 1 is pending human sign-off.**
