# Soil-profile background and surface-enrichment screening

Core analysis code for paired topsoil, A-horizon and C-horizon geochemistry from USGS Data Series 801. It includes background prediction, controlled enrichment experiments, reference-contamination sensitivity, national residual screening, horizon-depth comparisons and adjusted land-use contrasts.

This repository contains analysis source and compact frozen numerical results. Manuscript files, publication artwork, author information, credentials and large third-party downloads are excluded.

## Environment

The original calculations used Python 3.12. The pinned packages describe the recorded analysis environment; they are not a claim of compatibility with every Python version.

```bash
python -m venv .venv
# Activate the environment using the command appropriate to your operating system.
python -m pip install -r requirements.txt
python smoke_check.py
```

For the optional model benchmark, accelerated-failure-time sensitivity and EPA workbook extraction:

```bash
python -m pip install -r requirements-optional.txt
```

`smoke_check.py` checks core calculations, installed dependency versions, entry-point syntax and frozen-result consistency. `python smoke_check.py --fit-model` additionally fits a small synthetic model, and `--all-imports` explicitly imports the optional analysis modules. Neither command reproduces the complete scientific analysis. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Public inputs

USGS DS801 is available at [the official release](https://doi.org/10.3133/ds801). The downloader obtains the three tab-delimited horizon tables from their official URLs and verifies their SHA-256 values against the input snapshot used here.

```bash
python scripts/download_inputs.py
python run_analysis.py prepare
```

National source-proximity comparisons additionally use [USGS MRDS](https://mrdata.usgs.gov/mrds/), EPA smelter-related SEMS records, Superfund soil/sediment contaminants and TRI metal air releases. Downloading these inputs uses more than 350 MB of local disk space. The downloader's source list contains the exact official endpoints and checksums. EPA FRS coordinates are queried only when preparing missing source coordinates.

```bash
python scripts/download_inputs.py --external
```

`python scripts/download_inputs.py --check-links --external` checks each public endpoint by reading at most 1 KB. Some endpoints do not support HEAD requests. Public source files can change: a checksum mismatch stops the download instead of silently treating a changed input as the original snapshot. Large downloads, parsed input tables and regenerated outputs are git-ignored.

## Recompute stages

Run commands from the repository root. All inputs and outputs are relative to that root.

| Command | Calculation |
|---|---|
| `python run_analysis.py background` | Inheritance spectrum, profile and dual background comparisons |
| `python run_analysis.py injection` | Spatially partitioned semi-synthetic enrichment experiments |
| `python run_analysis.py reference` | Thirty-repetition cleaning comparison and label-noise sensitivity |
| `python run_analysis.py national` | External-input preparation, cross-conformal scores, source rules, national variants and mineral-occurrence comparisons |
| `python run_analysis.py vertical` | Withheld A-horizon comparisons and apparent penetration depth; requires the national stage |
| `python run_analysis.py landuse` | Overlap, permutation and lithogenic-normalised adjusted contrasts |
| `python run_analysis.py benchmark` | Completed four-element regression benchmark: Pb, As, Cu and Zn |
| `python run_analysis.py censoring` | Censored-response model sensitivity |

`python run_analysis.py all` runs these stages sequentially after public inputs have been downloaded. Full experiments and Gaussian-process benchmarks can be expensive; no full rerun is implied by a successful smoke check. The `reference` stage can create reusable score caches that are intentionally excluded from version control.

`config/analysis_settings.json` records the main screening settings and frozen headline counts. Full model settings, fixed random seeds and repetition-specific seed offsets remain explicit in the corresponding source modules. Older supporting modules are retained when imported by final analyses; they need not be run separately.

## Frozen results

`frozen_results/` contains completed numerical summaries and the 33 element–site discoveries at 31 unique sites. These files enable comparison with a rerun. They are not substituted for model fitting, raw horizon inputs or external-source harmonisation. No publication-figure rendering is included in this core-code release.

The release adds portable data directories and an external-preparation entry point that breaks the original ordering dependency between mineral-record preparation and national labels. Descriptive comments distinguish source-rule match prevalence from false-label probability. Leave-one-out ranks now count tied scores conservatively. None of the 96 actual national reference-score sets had ties, and comparison of both cleaning rules found no mask changes; see `config/score_tie_verification.json`. These changes do not alter the frozen national screening results. The portable parser also reproduced all 4,857 rows and 320 columns of the original derived input table exactly; see `config/input_parser_verification.json`.

## Interpretation

The paired-profile analysis separates concentration abundance, enrichment conditional on measured local geochemistry, and adjusted land-use composition. National calibration uses pooled spatial out-of-fold scores and data-adaptive reference trimming. Its error behaviour is evaluated empirically under specified simulations. Mineral proximity is a geological comparison rather than independent contamination truth, and apparent depth is a horizon-composite summary under a mixing model.

## Key sources

- [USGS geochemical and mineralogical data](https://doi.org/10.3133/ds801)
- [Testing for outliers with conformal p-values](https://doi.org/10.1214/22-AOS2244)
- [Robust conformal outlier detection under contaminated reference data](https://proceedings.mlr.press/v267/bashari25a.html)
- [Trimming under calibration contamination](https://doi.org/10.48550/arXiv.2605.06204)
- [Double/debiased machine learning](https://doi.org/10.1111/ectj.12097)

Third-party data and software retain the terms of their respective providers. This repository does not redistribute the source agency datasets.
