"""Project-wide configuration: domain, paths, constants.

Domain choice rationale
-----------------------
The paper (Ma et al. 2025, Ocean Eng. 340:122269) uses an 80x80 grid over the
Northwest Pacific. Cape Hatteras is a far smaller target, so we instead take a
Western North Atlantic box that is large enough to (a) contain the full storm
approach corridor and the Gulf Stream front, and (b) survive three rounds of
2x max-pooling in a U-Net encoder.

    lat  28.5N .. 44.0N   step 0.5  ->  32 points
    lon  83.5W .. 60.0W   step 0.5  ->  48 points

32 x 48 is divisible by 8 in both dimensions.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"

for _p in (RAW, INTERIM, PROCESSED, FIGURES, TABLES):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- domain
GRID_RES = 0.5  # degrees, matches the native ERA5 wave grid

LAT_MIN, LAT_MAX = 28.5, 44.0
LON_MIN, LON_MAX = -83.5, -60.0

N_LAT = int(round((LAT_MAX - LAT_MIN) / GRID_RES)) + 1  # 32
N_LON = int(round((LON_MAX - LON_MIN) / GRID_RES)) + 1  # 48

# Cape Hatteras / Diamond Shoals, the point of interest
HATTERAS_LAT, HATTERAS_LON = 35.25, -75.53

# Radius used to decide whether a storm is "locally relevant" to Hatteras.
LOCAL_RADIUS_KM = 500.0

# ---------------------------------------------------------------- time
# ERA5 runs from 1940, but its wave fields are only weakly constrained by
# observations before the altimeter era (ERS-1 1991, TOPEX 1992): earlier SWH
# is close to unconstrained model output, which is a poor supervision target.
# 2000 onward is well covered by multiple altimeters and matches the source
# paper's own training period (2000-2019), which keeps the reproduction
# directly comparable.
#
# Data for 1979-1997 is archived under data/raw/era5/archive_pre2000/ and is
# available for a later "does more data help" ablation.
YEAR_START = 2000
YEAR_END = 2025

# Held out entirely for testing; never seen during training or validation.
TEST_YEARS = (2021, 2022, 2023, 2024, 2025)

# ---------------------------------------------------------------- ERA5
ERA5_WAVE_VARS = {
    "swh": "significant_height_of_combined_wind_waves_and_swell",
    "mwd": "mean_wave_direction",
    "mp2": "mean_zero_crossing_wave_period",
    "pp1d": "peak_wave_period",
}
ERA5_WIND_VARS = {
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
}

# ---------------------------------------------------------------- HURDAT2
# NHC renames this file on every reissue, so we pin a known-good version but
# fall back to scraping the directory index for the newest Atlantic file.
HURDAT2_DIR_URL = "https://www.nhc.noaa.gov/data/hurdat/"
HURDAT2_URL = HURDAT2_DIR_URL + "hurdat2-1851-2025-02272026.txt"
HURDAT2_RAW = RAW / "hurdat2_atlantic.txt"

# HURDAT2 status codes. TC = genuine tropical cyclone stages; the rest are
# either pre-genesis, post-transition, or non-tropical lows.
TROPICAL_STATUS = {"TD", "TS", "HU"}
SUBTROPICAL_STATUS = {"SD", "SS"}
EXTRATROPICAL_STATUS = {"EX"}
OTHER_STATUS = {"LO", "WV", "DB", "ET", "PT", "ST", "TY"}

# Saffir-Simpson thresholds on 1-min sustained wind, in knots.
SAFFIR_SIMPSON = [
    (137, "Cat5"),
    (113, "Cat4"),
    (96, "Cat3"),
    (83, "Cat2"),
    (64, "Cat1"),
    (34, "TS"),
    (0, "TD"),
]


def classify_intensity(wind_kt):
    """Map a max-sustained-wind value (knots) to a Saffir-Simpson label."""
    if wind_kt is None or wind_kt < 0:
        return "Unknown"
    for threshold, label in SAFFIR_SIMPSON:
        if wind_kt >= threshold:
            return label
    return "TD"
