# Cape Hatteras Wave Forecasting

Reproduction of **Ma, Ding, Ai, Liu & Bu (2025), "Prediction of wave parameters
under extreme sea conditions using a 3D U-Net deep learning model"**,
*Ocean Engineering* 340:122269 — transplanted from the Northwest Pacific to
Cape Hatteras / the Outer Banks, North Carolina.

This reproduction is a **scaffold, not the destination**. The intended
direction is a physics-informed neural operator; the point of reproducing a
3D U-Net first is to get a clean, trustworthy dataset, baseline and evaluation
harness that any later architecture can be measured against.

---

## The one finding that shaped the design

The paper's headline methodological claim is its "extreme sea conditions
dataset construction method": slice ERA5 down to the lifetime windows of
tropical cyclones from a storm track database, and train only on those.

Applied to Cape Hatteras with HURDAT2 (1979–2024), that strategy gives:

| | |
|---|---|
| storms within 500 km of Hatteras | **148** (3.2/yr) |
| record covered by padded storm windows | **3.0%** — 97% of data discarded |
| major hurricanes (Cat3+) at closest approach | **5** in 46 years |
| still tropical at closest approach | 66% — **22% had already gone extratropical** |
| events in Dec, Feb, Mar, Apr | **zero** |

The last row is decisive. Outer Banks wave climate is bimodal: tropical
cyclones in summer/autumn, and nor'easters from October to April. A
HURDAT2-only catalogue is blind to an entire season that does much of the
cumulative work on this coast. In the Northwest Pacific typhoons genuinely do
dominate the extreme tail, so the paper's approach is defensible there; here it
is not.

**Consequence:** we train on the *continuous* record and use the storm
catalogue only for stratified evaluation and optional loss weighting — never as
a data filter. Extreme events are detected from the SWH field itself
(`extremes.py`), which catches both regimes, and are then *labelled* TC- or
ET-origin using HURDAT2.

---

## Other deliberate departures from the paper

| Paper | Here | Why |
|---|---|---|
| No persistence baseline | Persistence + climatology reported at every lead time | SWH is strongly autocorrelated; without this reference an MAE of 0.14 m at +1 h means nothing |
| MWD min-max normalised (0–360) | MWD decomposed into sin/cos | 359° and 1° are adjacent in reality but map to 0.997 and 0.003 |
| Land filled with 0, "invisible to ReLU" | Explicit land-mask channel + masked loss | Zeros propagate through convolutions like any other value. **27.8% of this domain is land** |
| Random 80/20 train/val split | Split by year | With a 48 h lookback, adjacent samples overlap by 47 h; random splitting leaks |
| Normalisation stats over full record | Stats from train years only | Otherwise test-set information leaks into training |
| Single 1 h lead time | 1 / 6 / 12 / 24 / 48 h | 1 h alone is too easy to be informative |
| Target excluded from input channels | Both modes supported (`paper_*` vs `full`) | The paper predicts SWH without ever showing the model SWH history. Unusual; worth measuring |

---

## Domain

```
lat  28.5N .. 44.0N   step 0.5°  ->  32 points
lon  83.5W .. 60.0W   step 0.5°  ->  48 points
```

32 × 48 matches the native ERA5 wave grid and is divisible by 8, so three
rounds of 2× pooling are exact (48→24→12→6, 32→16→8→4). The box contains the
full storm approach corridor and the Gulf Stream front.

---

## Layout

```
src/
  config.py               domain, paths, split years, constants
  download_era5.py        chunked, resumable CDS download
  cds_queue.py            inspect/purge the CDS job queue
  preprocess.py           netCDF -> contiguous memmap + normalisation stats
  dataset.py              sliding windows, masked loss, year-based splits
  models/                 unet3d, cnn3d, convlstm, persistence, climatology
  train.py                training loop
  run_experiments.py      the experiment matrix, run unattended
  predict.py              test-set inference + comparison tables
  evaluate.py             stratified metrics
  baseline_table.py       persistence/climatology reference table
  ablation_masking.py     effect of the land mask on reported metrics
  smoke_test.py           model shapes/throughput on synthetic tensors
  pipeline_smoke_test.py  full pipeline on synthetic ERA5 files
notebooks/
  01_results.ipynb        all results and figures, generated inline
data/{raw,interim,processed}/
outputs/{figures,tables,checkpoints,logs}/
_archive/extreme_analysis/  storm catalogue and extreme-event work
```

> **The extreme-event analysis is archived, not deleted.** `hurdat2.py`,
> `eda_hurdat2.py` and `extremes.py` — along with their figures, tables and the
> HURDAT2 data — were moved to `_archive/extreme_analysis/` on 2026-08-05 so
> that the main tree covers only ordinary SWH forecasting. That directory's
> README lists the findings and gives the commands to restore everything.

Environment: conda env `oceanwave` (Python 3.12, PyTorch cu128 for the RTX
5090). Kept separate from Anaconda `base`, whose MKL conflicts with PyTorch's
OpenMP runtime.

---

## If you just cloned this

The 20 GB of ERA5 is **not** in the repository. What is here is enough to read
and understand the work, and to re-make every figure:

| you can | you cannot, without the data |
|---|---|
| open `notebooks/01_results.ipynb` and see all results | run `preprocess.py` |
| re-run every plotting cell (forecasts are cached in `outputs/predictions_h1.npz`) | run `train.py` or `baseline_table.py` |
| read all the code | run `predict.py` on new samples |

To get the data, create a free account at
<https://cds.climate.copernicus.eu>, put your token in `~/.cdsapirc` (see
[CDS notes](#cds-notes) below), then run steps 1 and 2 under *Running it*. The
download takes roughly a day; it is resumable, so interruptions are harmless.

### Environment

```bash
conda create -n oceanwave python=3.12 -y
conda activate oceanwave
pip install numpy pandas xarray netCDF4 scipy matplotlib cdsapi tqdm pyarrow ipykernel
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or your CUDA version
python -m ipykernel install --user --name oceanwave --display-name "Python (oceanwave)"
```

A separate environment matters on Windows: Anaconda's MKL and PyTorch each ship
an OpenMP runtime, and loading both in one process aborts with `OMP: Error #15`.

Check it works — this runs the whole pipeline on synthetic data in a few
minutes and needs no ERA5:

```bash
python src/pipeline_smoke_test.py     # prints "pipeline OK end to end"
```

---

## Running it

```bash
PY="D:/Anaconda/envs/oceanwave/python.exe"

# 1. ERA5, 2000-2025. Resumable; rerun the same command after any interruption.
$PY src/download_era5.py --years 2000 2025 --workers 4
$PY src/download_era5.py --status

# 2. build the training cache
$PY src/preprocess.py

# 3. the persistence / climatology reference table
$PY src/baseline_table.py

# 4. train -- either one run at a time
$PY src/train.py --model unet3d --target swh --lead 1
$PY src/train.py --model cnn3d  --target swh --channel-set paper_swh
# ... or the whole matrix unattended (skips anything already trained)
$PY src/run_experiments.py

# 5. compare on the held-out years. One lead time per invocation:
#    models trained for different leads are not evaluated on the same samples.
$PY src/predict.py --checkpoints unet3d_swh_full_lb48_h1 \
      cnn3d_swh_full_lb48_h1 convlstm_swh_full_lb48_h1 \
      unet3d_swh_paper_swh_lb48_h1 --stride 6
```

Then open `notebooks/01_results.ipynb` (kernel **Python (oceanwave)**) for the
tables and figures — it generates every plot from the cached forecasts.

Validate code changes without waiting on data:

```bash
$PY src/smoke_test.py            # model shapes, GPU throughput
$PY src/pipeline_smoke_test.py   # preprocess -> dataset -> train -> predict
```

### CDS notes

The download is chunked into two-month requests because CDS prices a request
by size against a ceiling of 121,000 — six variables over this grid cost 53,568
per month, so two months (101,952) fits and a whole year (630,720) is rejected.
CDS also caps queued jobs per dataset at about 6; killing a downloader leaves
its jobs occupying that cap, and `cds_queue.py --purge` clears them.

Credentials live in `%USERPROFILE%\.cdsapirc`. The 2024 platform migration
invalidated the old `UID:APIKEY` format — tokens are now UUIDs from
<https://cds.climate.copernicus.eu/profile>.

---

## Period and splits

The record is **2000–2025**. Two reasons:

- ERA5 wave fields are only weakly constrained by observations before the
  altimeter era (ERS-1 1991, TOPEX 1992). Earlier SWH is close to
  unconstrained wave-model output, which makes a poor supervision target.
- It matches the source paper's own training period (2000–2019), so the
  reproduction stays directly comparable.

Data for 1979–1997 was downloaded before this decision and is kept under
`data/raw/era5/archive_pre2000/` for a later "does more data help" ablation. It
is outside the glob `preprocess.py` uses, so it does not enter the cache.

- **Test**: 2021–2025 (5 years), held out entirely.
- **Validation**: 2004, 2009, 2014, 2019 — scattered rather than contiguous so
  both sets stay representative across the AMO phase and the trend in storm
  frequency.
- **Train**: the remaining 17 years.

Windows straddling a split boundary, or spanning a gap in the hourly record,
are dropped.
