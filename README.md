# Contextual time squeeze: the well-being price of an hour in shrinking cities

Replication code for:

> Kato, H. *Contextual time squeeze: Population decline and the well-being
> price of an hour in Japan's shrinking small and medium-sized cities.*
> *Habitat International*, under review.

This repository contains the Python code used to estimate the within-person
"price" of an hour in each of seven daily activities — the change in life
satisfaction associated with reallocating one hour — for residents of
shrinking small and medium-sized cities (SMCs), non-shrinking SMCs and large
cities in Japan, and to test whether that price depends on municipal
population decline. It implements, on the 2020–2024 waves of the Cabinet
Office's Survey on Satisfaction and Quality of Life:

1. a **within-person isotemporal substitution model** (Eq. (1) of the article)
   with person and city-size × year fixed effects, estimated separately for
   the three groups, from which deviation-coded **domain values** (Eq. (3))
   and the 21 **pairwise reallocations** (Eq. (2)) are derived as linear
   combinations of the coefficients (Eq. (4));
2. **specificity checks** that interact the model with the continuous pace of
   decline, with four further municipal indicators (size, density,
   non-metropolitan location, merger), jointly, and in a three-way
   interaction with city size;
3. a **23-specification sensitivity analysis** of the shrinking-SMC estimates;
4. an **eldercare-onset event study** (Eq. (5); static, dynamic and dose
   forms, comparison events, a stacked never-treated specification);
5. the manuscript **figures**.

Every quantity in Tables 1–3, Figures 1–4, Supplementary Tables S1–S5 and
Figure S1 of the article is produced by `run_all.py`.

## Repository structure

```
├── run_all.py                   # Runs the five steps in order (--panel, --outdir, --steps)
├── src/
│   ├── common.py                # Data preparation, sample definition, Eq. (1), contrast vectors, inference
│   ├── s01_sample.py            # Steps 1-3: sample construction -> Table S1, Table 1
│   ├── s02_reallocation.py      # Steps 4-5: reallocation model and specificity -> Table 2, S2, S4; Fig. 1-3, S1 inputs
│   ├── s03_sensitivity.py       # Step 6:   sensitivity S0-S23 -> Table S3
│   ├── s04_eventstudy.py        # Step 7:   eldercare-onset event study -> Table 3, S5; Fig. 4 inputs
│   └── s05_figures.py           # Step 8:   Figures 1-4 and S1 (PNG 300 dpi + PDF), from the tables only
├── data/
│   ├── schema.csv               # Layout of panel.csv: every variable, its coding, source and use (no raw data)
│   ├── large_cities.csv         # The 105 statutory large cities (designated, core, former special)
│   └── municipal_means_2018_2023.csv  # Basic Resident Register 2018-2023 means by municipality (public)
├── .gitignore                   # Prevents accidental data/output commits
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

Running the code creates `output/main`, `output/supplement` and
`output/figures` under `--outdir`, which are not tracked in the repository.

## Data availability

The individual-level panel (`panel.csv`) is **not** redistributed here and
cannot be reconstructed from public sources. Life satisfaction, the time-use
items and the respondent covariates come from the Survey on Satisfaction and
Quality of Life (Cabinet Office of Japan, waves 2020–2025), whose individual
data the Cabinet Office provides on application for approved academic use,
subject to conditions that prohibit re-identification and re-provision to
third parties (<https://www5.cao.go.jp/keizai2/wellbeing/manzoku/index.html>).
Municipal population, in- and out-migration (Basic Resident Register, e-Stat
of Japan), land area and the metropolitan and merger indicators are public.

`data/schema.csv` documents every column of `panel.csv` — type, unit,
coding, source and whether the code uses it — including the conversion of
the survey's time-use bands to hours, so that researchers who obtain the
individual data can assemble an equivalent panel and run the pipeline
without modification. `data/large_cities.csv` holds the statutory list used
to classify city size (matched on the Japanese municipality name), and
`data/municipal_means_2018_2023.csv` the municipal means from which the
size and density indicators are built; an annual file
(`data/municipal_annual.csv`, columns `code5, Year, Population, Inflow,
Outflow`) is used instead if present.

All outputs are **aggregate-level** (coefficients, contrasts, counts,
sample flows) and safe to share; nothing person-level is written to disk.

## Requirements

- Python ≥ 3.10
- `pandas`, `numpy`, `scipy`, `statsmodels`, `matplotlib`, `pyfixest` 0.60.0
  (the article's results were produced with pyfixest 0.60.0, which is
  pinned because its singleton handling and cluster counts enter the
  reported person-years and degrees of freedom)

```
pip install -r requirements.txt
```

## How to run

```
python run_all.py --panel /path/to/panel.csv --outdir output
```

Runs the five steps in order and writes the tables (CSV and Markdown) and
figures (PNG and PDF). **Runtime** is a few minutes on a laptop; no random
numbers are used. `--steps s05` re-draws the figures from the existing
tables without re-estimating; each script can also be run on its own from
`src/` once `TSQ_PANEL` and `TSQ_OUTDIR` are set (or with the defaults
`data/panel.csv` and `output/`).

### Pipeline

| Step | Script | What it does | Article |
|---|---|---|---|
| 1 | `common.load_panel` | Reads the panel; `LS0 = LS − 1`; statutory city-size classification (128 large municipalities: 20 designated, 62 core, 23 former special cities and the 23 Tokyo wards; all others SMC); linkage check (drops persons whose age band falls or jumps across waves or whose sex changes); stayer flag on all observed rows; CAGR 2017–2023 standardised across municipalities; shrinking = CAGR ≤ −0.15 %/yr; municipal indicators for the specificity checks | Section 3.1 |
| 2 | `common.analytic_sample` | 2020–2024 waves; seven-domain total within 10–26 h; persons observed in ≥ 2 waves | Section 3.2, Table S1 |
| 3 | `common.groups` | Adults aged 15–64 split into shrinking SMCs, non-shrinking SMCs and large cities, each re-restricted to ≥ 2 waves within the group | Table 1 |
| 4 | `s02_reallocation.py` | Eq. (1) per group: `LS0 ~ 6 domain hours + total hours + age bands + covariates \| person + city-size × year`, SE clustered by municipality; domain values (Eq. 3) and reallocations (Eq. 2) as contrasts c′β with SE √(c′Σc) (Eq. 4); group differences as differences of independent estimates; Benjamini–Hochberg at FDR 5 % over the seven domain values and, separately, the 21 reallocations of the shrinking-SMC model | Sections 4.1–4.2, Tables 2, S2, Figures 1–2 |
| 5 | `s02_reallocation.py` | Specificity: (i) `(hours + total) × CAGR_z` among all SMC adults (per-SD interactions, implied values at ±1 SD, a grid along the CAGR axis, terciles); (ii) the same interaction for each of five indicators, and decline, density and size jointly; (iii) `(hours + total) × CAGR_z × SMC` among all adults | Section 4.3, Table S4, Figures 3, S1 |
| 6 | `s03_sensitivity.py` | S0 (main) and S1–S23 for the eldercare value, the sleep value and eldercare → sleep in shrinking SMCs: covariates (S1–S3), total-hours window (S4–S10), sample and instrument period (S11–S15, incl. person × period fixed effects), measurement (S16–S18), context and inference (S19–S23) | Section 4.4, Table S3 |
| 7 | `s04_eventstudy.py` | Eq. (5): onset = first observed 0 → >0 transition in eldercare hours; baseline carers excluded, never-starters as controls; static, dynamic (leads ≤ −3, −2; reference −1; 0, +1, ≥ +2; joint Wald test of the leads) and dose (0.5 h vs ≥ 1 h at onset) forms; pooled Post × group contrasts; exit from eldercare and childcare onset as comparison events; robustness for shrinking SMCs; stacked DID with never-treated controls (window −3 … +2, cohort-specific person and year effects) | Section 4.5, Tables 3, S5, Figure 4 |
| 8 | `s05_figures.py` | Figures from the tables: dot-and-whisker domain values, 7 × 7 reallocation matrices, eldercare value along the gradient of decline, event-study dynamics, moderator coefficient plots | Figures 1–4, S1 |

Inference throughout: municipality-clustered standard errors (CRV1) with
p-values and 95 % intervals from t(G − 1), where G is the number of clusters
in each model's own estimation sample (701, 179 and 128 in the three main
models; the smaller count for differences between groups). Every output row
carries its own `clusters`, `crit`, `lo` and `hi`, which the figures read.

## Output ↔ article mapping

### `output/main/`

| Output | Article element |
|---|---|
| `table1_sample.csv` / `.md` | Table 1 — estimation samples by type of municipality |
| `table2_domain_values.csv` / `.md` | Table 2 — value of one additional hour in each domain, three groups and the two differences (the CSV also holds the 2022–2024 replication, Table S3 S12, and the adjusted p-values given in the table note) |
| `table3_eventstudy.csv` / `.md` | Table 3 — change in life satisfaction after eldercare onset, three groups and pooled contrasts (the CSV also holds the exit and childcare events) |

### `output/supplement/`

| Output | Article element |
|---|---|
| `tableS1_sample_construction.md` (`tableS1_flow.csv`, `tableS1b_window_exclusions.csv`) | Table S1 — (a) sample flow, (b) person-years excluded by the total-hours window |
| `tableS2_reallocations.csv` / `.md`; `matrix_full.csv` | Table S2 — the 21 reallocations; Figure 2 inputs (all 42 directed pairs) |
| `tableS3_sensitivity.csv` / `.md` | Table S3 — S0–S23 in five panels |
| `tableS4_eldercare_indicators.md` | Table S4 — (a) eldercare value × municipal indicators, joint and three-way models, (b) eldercare value by CAGR tercile |
| `tableS4_moderators.csv` / `.md`, `tableS4b_joint.csv` / `.md`, `tableS4c_threeway.csv` / `.md`, `spec_terciles.csv` | all seven domains behind Table S4; Figure S1 inputs |
| `spec_gradient.csv`, `me_grid.csv` | Section 4.3 (per-SD interaction, values at ±1 SD); Figure 3 inputs |
| `bh_familywide.csv` | adjusted p over the 21 domain values of all three groups (note to Section 4.1) |
| `tableS5_es_robustness.csv` / `.md`, `tableS5b_es_stacked.csv` | Table S5 — (a) intensity and comparison events, (b) alternative specifications, (c) stacked DID |
| `es_dynamic.csv`, `es_dose.csv`, `es_pretrend.csv` | Figure 4 inputs; dose estimates; joint pre-trend tests |

### `output/figures/`

| Output | Article element |
|---|---|
| `fig1_domain_values.(png\|pdf)` | Figure 1 — domain values, three groups |
| `fig2_reallocation_matrices.(png\|pdf)` | Figure 2 — 7 × 7 reallocation matrices |
| `fig3_gradient.(png\|pdf)` | Figure 3 — eldercare value along the gradient of decline, SMC adults |
| `fig4_eventstudy.(png\|pdf)` | Figure 4 — life satisfaction around eldercare onset |
| `figS1_moderator_grids.(png\|pdf)` | Figure S1 — domain values × municipal indicators |

## Reproducibility notes

- City size follows the Local Autonomy Act, not the survey's own city-type
  code (which departs from the statutory list for eight municipalities and
  is retained only as `scale_survey`). `s01_sample.py` prints the number of
  large municipalities found in the panel and any statutory large city that
  is missing; the article's panel contains all 128.
- The person-years reported in every table are those of the estimation
  sample: persons are explicitly restricted to ≥ 2 waves within each group
  (and, for the municipality × year specification S19, singletons are
  removed iteratively before counting), so that the reported counts equal
  the observations used by `pyfixest`, which drops singletons by default.
- Free time is the omitted domain in Eq. (1); the reported domain values are
  deviation-coded (Eq. 3) and do not depend on that choice. In every
  interacted model the total-hours term is interacted together with the six
  domain terms, so the deviation-coded interactions are likewise invariant
  to the omitted domain.
- The 2022 redesign of the time-use module (reference day and eligible
  respondents) is addressed by the city-size × year fixed effects, the
  2022–2024 replication (S12) and the person × instrument-period
  specification (S14), which uses no within-person change across 2021→2022.
- The total-hours window (10–26 h) is an analytical choice; Table S1(b)
  describes the excluded person-years and S4–S10 trace the estimates across
  alternative windows.
- No random numbers are used, so the same input yields the same estimates.
  Thread ordering in the linear-algebra libraries can move point estimates
  by ~1e-11 and standard errors and p-values by ~1e-7 between machines,
  which can flip the third decimal of a value lying on a rounding boundary;
  no significance judgement is affected.
- Figures are drawn only from the saved tables, so `--steps s05` reproduces
  them exactly from a previous run.

## Releases

- **v1.0.0** (2026-10-30): initial release — Tables 1–3, Supplementary
  Tables S1–S5, Figures 1–4 and S1 of the submitted manuscript.

## Citation

If you use this code, please cite the paper above and, for the exact
released version of this code, the archived DOI (Zenodo; citation
metadata for this repository is provided in `CITATION.cff`).

## License

MIT — see `LICENSE`. The license applies to the code only and does not
extend to the survey data, which remain subject to the Cabinet Office's
conditions of provision.

## Contact

Haruka Kato, Osaka Metropolitan University
