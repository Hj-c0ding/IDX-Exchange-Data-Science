# Harshal-Joshi-IDX-Exchange-Data-Science

## Weekly EDA workflow

Create or update the environment, then select its Jupyter kernel:

```bash
conda env update -n idx-exchange-env -f environment.yml
conda activate idx-exchange-env
```

Open `01_exploration.ipynb` and run all cells. The notebook reads `CRMLSSold*.csv` from `Data and Python File for Internship/`, verifies at least six months of `CloseDate` coverage, filters to `PropertyType == "Residential"` and `PropertySubType == "SingleFamilyResidence"`, and explores `ClosePrice`, `LivingArea`, `Bedrooms`, `Bathrooms`, and `LotSize`.

The source columns are mapped as follows:

| Analysis field | Source field |
| --- | --- |
| `Bedrooms` | `BedroomsTotal` |
| `Bathrooms` | `BathroomsTotalInteger` |
| `LotSize` | `LotSizeSquareFeet` |

## Confidential data handling

- Raw CSVs, spreadsheets, Parquet files, archives, generated data, and notebook checkpoints are ignored by Git.
- The notebook loads only the eight fields required for this analysis and does not print or export identifiers, addresses, agent names, coordinates, or emails.
- Never commit raw MLS files, notebook outputs containing raw records, API keys, access tokens, or `.env` files.
- The downloader scripts read `IDXEXCHANGE_AUTH_ENDPOINT` from the local environment and write downloaded files under the ignored `data/` directory. Set that variable locally; do not place its value in source code.
- Before committing, inspect `git status --short` and `git diff --cached`.

## Weekly deliverable progress

- **Week 1 — Exploration:** Completed the initial exploration of sold CRMLS records, including the required property filters and requested variable distributions. See [`01_exploration.ipynb`](./01_exploration.ipynb).
- **Week 2 — Data preparation:** Completed the data preparation workflow used to organize the selected records for modeling.
- **Week 3 — Data preprocessing:** Completed missing-value handling, categorical encoding, numeric scaling, and the time-ordered train/test split. See [`02_preprocessing.ipynb`](./02_preprocessing.ipynb).
- **Week 4 — Baseline model:** Completed a linear regression baseline, evaluated it with test-set R², and recorded the result in the notebook. See [`03_baseline_model.ipynb`](./03_baseline_model.ipynb).