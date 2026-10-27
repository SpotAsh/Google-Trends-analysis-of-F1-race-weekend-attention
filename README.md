# What Drives Search Interest Around F1 Grand Prix Weekends?

An end-to-end data analysis of Google Trends search interest across **213 Formula 1 race weekends (2016–2025)**. Penn State DAAN 545 final project.

The project asks whether a tight championship fight makes the public pay more attention to a race, and whether outside shocks (*Drive to Survive*, the 2020 pandemic season) changed the level of interest. It combines hypothesis testing, frequent pattern mining, clustering, dimensionality reduction and outlier detection in a reproducible Python pipeline.

**Full write-up:** [`reports/What Drives Search Interest Around F1 Grand Prix Weekends Final Report (1).pdf`]

---

## Key findings

- **A measurement flaw almost produced a false result.** My first spike metric (peak ÷ baseline) correlated with title closeness (ρ = −0.18, p = 0.009), but it mostly measures how small the baseline was, and close seasons were early seasons with tiny baselines. The corrected metric (lift in index points) shows no relationship (ρ = +0.04, p = 0.61; pre-specified within-season test r = −0.05, p = 0.52).
- **Title closeness is not the driver.** Every cut of the data agrees, and the confidence intervals rule out effects larger than about one index point out of a typical 21.
- **Attention is a three-day event.** Race day is the peak in 210 of 213 races, and the day before, race day and the day after carry 77% of the volume in a fifteen-day window.
- **There was a step in 2022, not 2019.** Median lift rose by 2.5 points (95% interval 1.6 to 3.7) and baseline interest roughly quadrupled from 2022. This is compatible with a *Drive to Survive* era effect but does not isolate it. The 2020 season shows no clear change.
- **Race weekends form three soft types** (silhouette 0.37) that follow circuit type and season, not the title fight: established year-round interest (e.g. Monaco), high-surge, and event-only.
- **Most outliers are measurement artifacts.** Of 11 consensus outliers, three Las Vegas races are explained by Saturday-night timing (verified by shifting the window), and two São Paulo races look like very low search volume.

<!-- Add one or two figures from reports/figures/ here, e.g.:
![Average search interest around race day](reports/figures/<your_figure>.png)

![Alt text](path/to/image.png)
-->
The metric that almost misled me: old spike ratio vs. corrected lift
![Alt text](f1-economic-anomalies/reports/figures/final_A_old_vs_new_metric.png)

Championship gap vs. race-week lift (no relationship)
![Alt text](f1-economic-anomalies/reports/figures/final_C_gap_vs_lift.png)

Lift by season (the 2022 step)
![Alt text](f1-economic-anomalies/reports/figures/final_E_lift_by_season.png)

Three clustering methods compared
![Alt text](f1-economic-anomalies/reports/figures/mid2_clusters_three_methods.png)

Outlier detector agreement
![Alt text](f1-economic-anomalies/reports/figures/mid2_outlier_agreement.png)

## Data

| Source | What it provides | Coverage |
| --- | --- | --- |
| Wikipedia season pages | Race calendar (round, Grand Prix, circuit, date) | 214 races, 2016–2025 |
| Google Trends (via `pytrends`, worldwide) | Daily search interest for each Grand Prix name, 60 days before to 30 days after the race | 213 of 214 races |
| Jolpica-F1 API | Driver standings before each race (points gap between 1st and 2nd) | 203 races (season openers have no prior standings) |
| Inside Airbnb (supplementary) | One listings snapshot per host city | 36 races, 2022–2025 only |

The project began as a study of local economic impact (hotel and Airbnb prices). Hotel sites block scraping and Airbnb coverage was too thin, so it pivoted to search interest, which exists for nearly every race. The Airbnb layer is kept as a minor supplementary variable and supports no conclusions.

## Methods

- **Outcome metric:** lift in index points = mean interest in race week (days −7 to 0) minus mean interest in the baseline (days −60 to −14), plus a lift z-score and race-week share.
- **Hypothesis tests:** Spearman and within-season rank correlations, season fixed effects with permutation tests, Mann-Whitney comparisons, break-year scan, bootstrap intervals.
- **Frequent pattern mining:** Apriori and FP-Growth at three minimum supports on 23 binary features.
- **Clustering:** K-means, Ward hierarchical and DBSCAN on five standardized curve-shape features.
- **Dimensionality reduction:** PCA and t-SNE.
- **Outlier detection:** z-score, Isolation Forest, Local Outlier Factor, k-NN distance and One-Class SVM, with a 3-of-5 consensus rule.

## Repository structure

```
f1-economic-anomalies/
├── data/
│   ├── raw/          # scraped races, Trends pulls, standings cache, Airbnb files
│   └── processed/    # merged and engineered feature tables
├── src/              # numbered pipeline scripts (see below)
├── notebook/         # exploratory notebooks
├── outputs/          # intermediate outputs
├── reports/          # final report, result tables and figures
├── requirements.txt
└── README.md
```

### Pipeline

| Script | What it does |
| --- | --- |
| `01_scrape_races.py` | Scrapes Wikipedia season calendars for 2016–2025 |
| `02_pull_trends.py` | Pulls daily Google Trends per race with retries, backoff and caching |
| `03_preprocessing.py` | Merges races and trends, adds championship gap (Jolpica-F1, cached) and the 2020 flag |
| `04_anomaly_detection.py` | First-pass anomaly flags (z-score, Isolation Forest, LOF) |
| `05_visualize_anomalies.py` | Exploratory anomaly figures |
| `06_pull_airbnb.py` | Supplementary Airbnb pull (run separately, 2022–2025 only) |
| `07_missing_values_and_integration.py` | Missing-value and match-loss reporting at each integration stage |
| `08_descriptive_stats_and_normalization.py` | Descriptive statistics and log transforms |
| `09_binary_features.py` | Binary feature table for pattern mining |
| `10_frequent_pattern_mining.py` | Apriori, FP-Growth and association rules |
| `11_final_analysis.py` | Lift metric, hypothesis tests, driver analysis and figures |
| `12_clustering_dimred_outliers.py` | Clustering, PCA/t-SNE and five outlier detectors |

## How to run

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline in order
python src/01_scrape_races.py
python src/02_pull_trends.py
python src/03_preprocessing.py
# ... continue through 12_clustering_dimred_outliers.py
```

Downloads are cached in `data/raw/`, so re-running does not hit the web APIs again. All random procedures use seed 42. Google Trends is accessed through the unofficial `pytrends` library and may rate-limit; the scripts retry with backoff.

## Limitations

- **Google Trends is a relative, per-race index.** Each pull is scaled to its own maximum, so lift measures how sharply attention concentrates around race day, not absolute audience size. A fixed reference term in each pull would be needed to compare audiences.
- **Era and treatment are confounded.** Rule changes, media rights, social-media reach and *Drive to Survive* all changed together, so the 2022 step is described, not attributed.
- **Search term ambiguity.** The query is the Grand Prix name, which can be diluted by non-F1 searches or miss local-language queries.
- **Date alignment.** Saturday-night races (Las Vegas) peak the day after race day; the shifted window was verified only for those three races.
- **Exploratory tests.** Only the within-season closeness test was pre-specified; other p-values near 0.05 are suggestive.
- **One race missing.** The 2021 São Paulo Grand Prix returned no Trends data.

### Next steps

- Recompute lift with a shifted window for all Saturday-night races and correct the three understated Las Vegas values.
- Add a fixed reference term to every Trends pull so lifts share one scale.
- Use localized queries for non-English-speaking host countries.
- Search for drivers and the championship, not just the Grand Prix name.

## Author

Penn State, M.S. Data Analytics. [Aditya Mogullapalli] · [[LinkedIn](https://www.linkedin.com/in/aditya-mogullapalli-75879a369/)] · [Email](aditya.mogullapalli@gmail.com)
