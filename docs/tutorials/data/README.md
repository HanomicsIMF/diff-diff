# Tutorial Data Fixtures

This directory holds small datasets used by tutorials in `docs/tutorials/`. Files
here are committed to the repository so tutorials can be reproduced offline.

## `geolift_test.csv`

**Used by**: `docs/tutorials/18_geo_experiments.ipynb` (Section 6, cross-method
comparison against GeoLift's published Augmented Synthetic Control output).

**Source**: Extracted from `GeoLift_Test.rda`, the canonical example dataset
shipped with the [`facebookincubator/GeoLift`](https://github.com/facebookincubator/GeoLift)
R package.

**Original file**: <https://github.com/facebookincubator/GeoLift/blob/main/data/GeoLift_Test.rda>

**License**: MIT (matches the GeoLift R package). The original LICENSE is at
<https://github.com/facebookincubator/GeoLift/blob/main/LICENSE>.

**Extraction date**: 2026-04-11

**Extraction method**: Read the `.rda` file with the [`rdata`](https://pypi.org/project/rdata/)
Python package (pure Python, no R dependency), then exported to CSV with two
added columns:
- `day` (1-105): integer day index per location, sorted by date.
- `treated` (0/1): binary indicator. Set to 1 for `chicago` and `portland`
  (the canonical test markets in the GeoLift Walkthrough vignette), 0 for the
  other 38 cities.

The original `Y`, `location`, and `date` columns are unchanged.

**Schema**:

| Column     | Type   | Description                                            |
|------------|--------|--------------------------------------------------------|
| location   | string | City name, lowercase (e.g. `chicago`, `new york`)      |
| Y          | int    | Daily outcome (units sold or equivalent KPI)           |
| date       | string | ISO date `YYYY-MM-DD`, range 2021-01-01 to 2021-04-15  |
| day        | int    | 1-indexed day, 1-105                                   |
| treated    | int    | 1 for chicago/portland, 0 for the other 38 cities      |

**Shape**: 4200 rows (40 locations x 105 days), 5 columns. ~127 KB.

**Reproducibility**: To regenerate from the upstream `.rda`:

```python
import rdata
import pandas as pd

# Read .rda directly (pyreadr or rpy2 also work)
parsed = rdata.read_rda("GeoLift_Test.rda")
df = parsed["GeoLift_Test"].copy()

# Add columns matching this fixture
df["location"] = df["location"].astype(str)
df["Y"] = df["Y"].astype(int)
df["date"] = df["date"].astype(str)
df = df.sort_values(["location", "date"]).reset_index(drop=True)
df["day"] = df.groupby("location").cumcount() + 1
df["treated"] = df["location"].isin({"chicago", "portland"}).astype(int)

df.to_csv("docs/tutorials/data/geolift_test.csv", index=False)
```
