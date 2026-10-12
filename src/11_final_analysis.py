"""
11_final_analysis.py

Answers the project question: what drives search-interest spikes around Grand
Prix weekends, and does championship closeness explain them?

Why a new metric: the old spike_factor = peak / baseline is ~100 / baseline
(each Trends pull is scaled so its busiest day = 100), so it mostly measures
how tiny the baseline was. Here the spike is measured as a LIFT in index
points, which is bounded and not inflated by near-zero baselines:

    lift_pts        = race_week_mean - baseline_mean      (0-100 index points)
    lift_z          = lift_pts / max(baseline_std, 1.0)   (floored so a flat
                                                           baseline cannot blow it up)
    race_week_share = race-week volume / whole-window volume (0-1)

ANALYSIS PLAN (fixed BEFORE the 2016-2021 seasons were added, so the extra years
cannot be used to hunt for significance):
    PRIMARY test   : within-season rank correlation of championship gap vs
                     lift_pts (reported once, two-sided, whatever the outcome).
    EXPLORATORY    : everything else below (thresholds, per-season, drivers,
                     era comparison, anomaly overlap). With this many tests,
                     treat p-values near 0.05 as suggestive, not conclusive.

Inputs (all already produced by earlier scripts):
    data/processed/trends_master.csv
    data/processed/f1_final_features.csv
    data/processed/f1_binary_features.csv
Outputs:
    reports/final_race_metrics.csv
    reports/final_results.txt
    reports/figures/final_*.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

PROC = Path("data/processed")
REPORTS = Path("reports")
FIGS = REPORTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

OUT_LINES = []


def say(text=""):
    print(text)
    OUT_LINES.append(str(text))


# ---------------------------------------------------------------- load
trends = pd.read_csv(PROC / "trends_master.csv")
trends["date"] = pd.to_datetime(trends["date"], format="mixed", errors="coerce")
feat = pd.read_csv(PROC / "f1_final_features.csv")
feat["race_date"] = pd.to_datetime(feat["race_date"], errors="coerce")
binf = pd.read_csv(PROC / "f1_binary_features.csv")
if "is_covid_season" not in feat.columns:
    feat["is_covid_season"] = feat["season"].astype(int) == 2020
feat["is_covid_season"] = feat["is_covid_season"].map(
    {True: True, False: False, "True": True, "False": False}
).fillna(False).astype(bool)

assert trends["date"].notna().all(), "unparseable trends dates"
assert feat["race_date"].notna().all(), "unparseable race dates"

t = trends.merge(feat[["race_id", "race_date"]], on="race_id", how="inner")
t["d"] = (t["date"] - t["race_date"]).dt.days

# ---------------------------------------------------------------- new metrics
rows = []
for rid, g in t.groupby("race_id"):
    base = g[g["d"].between(-60, -14)]["search_interest"]
    rw = g[g["d"].between(-7, 0)]["search_interest"]
    if len(base) == 0 or len(rw) == 0:
        continue
    base_mean = float(base.mean())
    base_std = float(base.std(ddof=0))
    rw_mean = float(rw.mean())
    total = float(g["search_interest"].sum())
    rows.append({
        "race_id": rid,
        "baseline_mean": base_mean,
        "baseline_std": base_std,
        "race_week_mean_new": rw_mean,
        "lift_pts": rw_mean - base_mean,
        "lift_z": (rw_mean - base_mean) / max(base_std, 1.0),
        "race_week_share": float(rw.sum()) / total if total > 0 else np.nan,
    })
m = pd.DataFrame(rows)

df = feat.merge(m, on="race_id", how="inner")
df = df.merge(binf[["race_id", "circuit_type_street"]], on="race_id", how="left")
df["is_street"] = df["circuit_type_street"].fillna(False).astype(bool)
df["is_covid_season"] = df["is_covid_season"].fillna(False).astype(bool)
df["is_close"] = df["is_close_championship_race"].map({True: True, False: False, "True": True, "False": False})
df["log_spike_old"] = np.log1p(df["spike_factor"])
df["log_baseline"] = np.log1p(df["baseline_mean"])

assert len(df) == len(feat), f"lost races in merge: {len(df)} vs {len(feat)}"
df.to_csv(REPORTS / "final_race_metrics.csv", index=False)

say("=" * 70)
say(f"NEW SPIKE METRIC: DESCRIPTIVES ({len(df)} races)")
say("=" * 70)
say(df[["baseline_mean", "lift_pts", "lift_z", "race_week_share", "spike_factor"]]
    .describe().round(2).T[["count", "mean", "std", "min", "50%", "max"]].to_string())

# Show that the old metric was just the inverse baseline.
rho_old_base, _ = stats.spearmanr(df["spike_factor"], df["baseline_mean"])
rho_new_base, p_new_base = stats.spearmanr(df["lift_pts"], df["baseline_mean"])
say(f"\nSpearman(old spike_factor, baseline_mean) = {rho_old_base:.2f}  (near -1 => old metric ~ inverse baseline)")
say(f"Spearman(new lift_pts,     baseline_mean) = {rho_new_base:.2f}  (p={p_new_base:.3f})")

# ---------------------------------------------------------------- H1: closeness
say("\n" + "=" * 70)
say("H1: DOES CHAMPIONSHIP CLOSENESS EXPLAIN THE LIFT?")
say("=" * 70)
h = df.dropna(subset=["championship_gap_entering_race"]).copy()
n = len(h)


def boot_ci(x, y, n_boot=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    y = np.asarray(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        r, _ = stats.spearmanr(x[idx], y[idx])
        vals.append(r)
    return np.nanpercentile(vals, [2.5, 97.5])


for col, label in [("lift_pts", "lift_pts"), ("lift_z", "lift_z"),
                   ("race_week_share", "race_week_share"), ("log_spike_old", "old log spike (reference)")]:
    rho, p = stats.spearmanr(h["championship_gap_entering_race"], h[col])
    lo, hi = boot_ci(h["championship_gap_entering_race"], h[col])
    say(f"gap vs {label:<26} Spearman rho = {rho:+.3f}  p = {p:.3f}  95% CI [{lo:+.2f}, {hi:+.2f}]  n = {n}")
say("(negative rho = closer title fight -> bigger lift, the hypothesised direction)")

# Within-season version: removes season-level differences (e.g. 2022 vs 2025).
h["gap_rank_in_season"] = h.groupby("season")["championship_gap_entering_race"].rank(pct=True)
h["lift_rank_in_season"] = h.groupby("season")["lift_pts"].rank(pct=True)
r_w, p_w = stats.pearsonr(h["gap_rank_in_season"], h["lift_rank_in_season"])
say(f"\n*** PRIMARY TEST (pre-specified) *** within-season (ranks inside each season) "
    f"gap vs lift_pts: r = {r_w:+.3f}, p = {p_w:.3f}, n = {len(h)}")

say("\nPer-season Spearman (gap vs lift_pts):")
for s, g in h.groupby("season"):
    if len(g) > 4:
        r, p = stats.spearmanr(g["championship_gap_entering_race"], g["lift_pts"])
        say(f"  {s}: rho = {r:+.2f}, p = {p:.3f}, n = {len(g)}")

close = h[h["is_close"] == True]["lift_pts"]
notc = h[h["is_close"] == False]["lift_pts"]
u, p_u = stats.mannwhitneyu(close, notc, alternative="two-sided")
say(f"\nClose (<25 pts) vs not close, lift_pts: median {close.median():.2f} vs {notc.median():.2f}, "
    f"mean {close.mean():.2f} vs {notc.mean():.2f}, n = {len(close)} vs {len(notc)}")
say(f"Mann-Whitney U = {u:.0f}, p = {p_u:.3f}")

# Threshold sensitivity so the answer does not hinge on the 25-point cutoff.
say("\nThreshold sensitivity (median lift_pts, close vs not close; Mann-Whitney p):")
for thr in [10, 25, 50]:
    a = h[h["championship_gap_entering_race"] < thr]["lift_pts"]
    b = h[h["championship_gap_entering_race"] >= thr]["lift_pts"]
    if len(a) > 2 and len(b) > 2:
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        say(f"  gap < {thr:>2}: {a.median():.2f} vs {b.median():.2f}  (n = {len(a)} vs {len(b)}), p = {p:.3f}")

# ---------------------------------------------------------------- 2022 contrast
say("\n" + "=" * 70)
say("2022 CONTRAST CASE AND SEASON EFFECT")
say("=" * 70)
by_season = df.groupby("season").agg(
    n=("race_id", "count"),
    median_lift=("lift_pts", "median"),
    mean_lift=("lift_pts", "mean"),
    median_gap=("championship_gap_entering_race", "median"),
    median_baseline=("baseline_mean", "median"),
).round(2)
say(by_season.to_string())
groups = [g["lift_pts"].dropna().values for _, g in df.groupby("season")]
groups = [g for g in groups if len(g) > 0]
H, p_k = stats.kruskal(*groups)
say(f"\nKruskal-Wallis across seasons (lift_pts): H = {H:.2f}, p = {p_k:.3f}")
# Tight vs blowout seasons, defined by each season's median gap (lowest third
# of seasons = tight, highest third = blowout). Exploratory.
season_gap = df.groupby("season")["championship_gap_entering_race"].median().sort_values()
n_side = max(2, len(season_gap) // 3)
tight_seasons = list(season_gap.index[:n_side])
blowout_seasons = list(season_gap.index[-n_side:])
a = df[df["season"].isin(tight_seasons)]["lift_pts"]
b = df[df["season"].isin(blowout_seasons)]["lift_pts"]
_, p_tb = stats.mannwhitneyu(a, b, alternative="two-sided")
say(f"\nTight seasons {tight_seasons} (median gap) vs blowout seasons {blowout_seasons}: "
    f"median lift {a.median():.2f} vs {b.median():.2f}, Mann-Whitney p = {p_tb:.3f}")
p_2225 = p_tb  # kept for the figure title below

# Drive to Survive era (first season released March 2019). Exploratory: did the
# overall level of attention change, and did the closeness relationship change?
df["post_dts"] = df["season"] >= 2019
pre = df[~df["post_dts"]]
post = df[df["post_dts"]]
if len(pre) > 4 and len(post) > 4:
    say(f"\nPre-DTS (<=2018, n={len(pre)}) vs post-DTS (>=2019, n={len(post)}):")
    for col in ["baseline_mean", "lift_pts", "race_week_share"]:
        _, p = stats.mannwhitneyu(pre[col], post[col], alternative="two-sided")
        say(f"  {col:<16} median {pre[col].median():.2f} vs {post[col].median():.2f}  (p = {p:.3f})")
    for label, g in [("pre-DTS", pre), ("post-DTS", post)]:
        gg = g.dropna(subset=["championship_gap_entering_race"])
        if len(gg) > 4:
            r, p = stats.spearmanr(gg["championship_gap_entering_race"], gg["lift_pts"])
            say(f"  gap vs lift_pts within {label}: rho = {r:+.3f}, p = {p:.3f}, n = {len(gg)}")

covid = df[df["is_covid_season"]]
non_covid = df[~df["is_covid_season"]]
if len(covid) > 0 and len(non_covid) > 4:
    say(f"\n2020 COVID season (n={len(covid)}) vs other seasons (n={len(non_covid)}):")
    for col in ["baseline_mean", "lift_pts", "race_week_share"]:
        _, p = stats.mannwhitneyu(covid[col], non_covid[col], alternative="two-sided")
        say(f"  {col:<16} median {covid[col].median():.2f} vs {non_covid[col].median():.2f}  (p = {p:.3f})")

# ---------------------------------------------------------------- drivers
say("\n" + "=" * 70)
say("WHAT ELSE DRIVES THE LIFT? (Spearman with lift_pts)")
say("=" * 70)
drivers = {
    "baseline_mean (off-season interest)": "baseline_mean",
    "championship_gap": "championship_gap_entering_race",
    "round_num (position in season)": "round_num",
    "is_street_circuit": "is_street",
    "airbnb_price_snapshot_mean_usd": "airbnb_price_snapshot_mean_usd",
    "airbnb_listing_count": "airbnb_listing_count",
    "is_covid_season": "is_covid_season",
}
drv_rows = []
for label, col in drivers.items():
    sub = df[[col, "lift_pts"]].dropna()
    x = sub[col].astype(float)
    if x.nunique() < 2 or len(sub) < 5:
        continue
    rho, p = stats.spearmanr(x, sub["lift_pts"])
    drv_rows.append({"driver": label, "rho": rho, "p": p, "n": len(sub)})
    say(f"{label:<38} rho = {rho:+.3f}  p = {p:.3f}  n = {len(sub)}")
drv = pd.DataFrame(drv_rows)

# Joint rank regression on races with a gap value (numpy OLS, standardized ranks).
X_cols = ["baseline_mean", "championship_gap_entering_race", "round_num", "is_street"]
if h["is_covid_season"].nunique() >= 2:
    X_cols = X_cols + ["is_covid_season"]
rr = h[X_cols + ["lift_pts"]].copy()
rr["is_street"] = rr["is_street"].astype(float)
if "is_covid_season" in rr.columns:
    rr["is_covid_season"] = rr["is_covid_season"].astype(float)
rk = rr.rank(pct=True)
rk = (rk - rk.mean()) / rk.std(ddof=0)
Xm = np.column_stack([np.ones(len(rk))] + [rk[c].values for c in X_cols])
yv = rk["lift_pts"].values
beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
pred = Xm @ beta
r2 = 1 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
say("\nJoint rank regression (standardized ranks) on lift_pts, n = %d, R^2 = %.2f" % (len(rk), r2))
for c, b_ in zip(X_cols, beta[1:]):
    say(f"  {c:<34} beta = {b_:+.3f}")

# ---------------------------------------------------------------- anomalies on new metric
say("\n" + "=" * 70)
say("ANOMALIES ON THE NEW METRIC")
say("=" * 70)
Xa = df[["lift_pts", "lift_z", "race_week_share"]].fillna(0)
df["z_flag"] = ((df["lift_pts"] - df["lift_pts"].mean()) / df["lift_pts"].std(ddof=0)) > 1.5
df["iso_flag"] = IsolationForest(n_estimators=200, contamination=0.10, random_state=42).fit_predict(Xa) == -1
df["lof_flag"] = LocalOutlierFactor(n_neighbors=10, contamination=0.10).fit_predict(Xa) == -1
df["n_flags"] = df[["z_flag", "iso_flag", "lof_flag"]].sum(axis=1)
df["is_anomaly_2of3"] = df["n_flags"] >= 2
say(f"z>1.5: {df['z_flag'].sum()}, IsolationForest: {df['iso_flag'].sum()}, LOF: {df['lof_flag'].sum()}, "
    f"flagged by all 3: {(df['n_flags'] == 3).sum()}, by >=2 of 3: {df['is_anomaly_2of3'].sum()}")
top = df[df["is_anomaly_2of3"]].sort_values("lift_pts", ascending=False)[
    ["race_id", "grand_prix", "baseline_mean", "lift_pts", "lift_z", "championship_gap_entering_race", "n_flags"]]
say(top.round(2).to_string(index=False))

ct = pd.crosstab(h.merge(df[["race_id", "is_anomaly_2of3"]], on="race_id")["is_anomaly_2of3"],
                 h["is_close"].values)
say("\nAnomaly (>=2 of 3) x close-championship (races with a gap):")
say(ct.to_string())
if ct.shape == (2, 2):
    odds, p_f = stats.fisher_exact(ct.values)
    say(f"Fisher exact: odds ratio = {odds:.2f}, p = {p_f:.3f}")

# ---------------------------------------------------------------- figures
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
_seasons_sorted = sorted(df["season"].unique())
_cmap = plt.get_cmap("viridis")
SEASON_COLORS = {s: _cmap(i / max(1, len(_seasons_sorted) - 1)) for i, s in enumerate(_seasons_sorted)}

# Fig A: old vs new metric against baseline (shows why we changed the metric)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(df["baseline_mean"], df["spike_factor"], s=22, color="#555")
ax[0].set_yscale("log")
ax[0].set_xlabel("Baseline search interest (0-100)")
ax[0].set_ylabel("Old spike factor (log scale)")
ax[0].set_title(f"Old metric tracks 1/baseline\n(Spearman = {rho_old_base:.2f})")
ax[1].scatter(df["baseline_mean"], df["lift_pts"], s=22, color="#1f77b4")
ax[1].set_xlabel("Baseline search interest (0-100)")
ax[1].set_ylabel("New lift (index points)")
ax[1].set_title(f"New metric is bounded\n(Spearman = {rho_new_base:.2f})")
fig.tight_layout()
fig.savefig(FIGS / "final_A_old_vs_new_metric.png")
plt.close(fig)

# Fig B: distribution of new lift
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(df["lift_pts"], bins=20, color="#1f77b4", edgecolor="white")
ax.axvline(df["lift_pts"].median(), color="k", ls="--", lw=1, label=f"median = {df['lift_pts'].median():.1f}")
ax.set_xlabel("Race-week lift over baseline (index points)")
ax.set_ylabel("Races")
ax.set_title(f"Distribution of race-week lift ({len(df)} races)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGS / "final_B_lift_distribution.png")
plt.close(fig)

# Fig C: gap vs lift, coloured by season, with overall rank trend
fig, ax = plt.subplots(figsize=(7, 4.5))
for s, g in h.groupby("season"):
    ax.scatter(g["championship_gap_entering_race"], g["lift_pts"], s=28, label=str(s),
               color=SEASON_COLORS.get(s, "#777"), alpha=0.85)
rho_all, p_all = stats.spearmanr(h["championship_gap_entering_race"], h["lift_pts"])
ax.axvline(25, color="grey", ls=":", lw=1)
ax.text(26, ax.get_ylim()[1] * 0.95, "close (<25)", fontsize=8, color="grey", va="top")
ax.set_xlabel("Championship gap entering race (points)")
ax.set_ylabel("Race-week lift (index points)")
ax.set_title(f"Title-fight closeness vs search lift\nSpearman rho = {rho_all:+.2f}, p = {p_all:.2f}")
ax.legend(title="Season", frameon=False)
fig.tight_layout()
fig.savefig(FIGS / "final_C_gap_vs_lift.png")
plt.close(fig)

# Fig D: close vs not close boxplot
fig, ax = plt.subplots(figsize=(5, 4.2))
ax.boxplot([notc.values, close.values], tick_labels=[f"Not close\n(n={len(notc)})", f"Close <25 pts\n(n={len(close)})"],
           widths=0.5, showfliers=True)
ax.set_ylabel("Race-week lift (index points)")
ax.set_title(f"Close vs not close\nMann-Whitney p = {p_u:.2f}")
fig.tight_layout()
fig.savefig(FIGS / "final_D_close_vs_not.png")
plt.close(fig)

# Fig E: lift by season (2022 contrast)
fig, ax = plt.subplots(figsize=(8.5, 4.2))
seasons = sorted(df["season"].unique())
ax.boxplot([df[df["season"] == s]["lift_pts"].values for s in seasons],
           tick_labels=[str(s) for s in seasons], widths=0.5)
ax.set_ylabel("Race-week lift (index points)")
ax.set_title(f"Lift by season (Kruskal-Wallis p = {p_k:.2f})")
fig.tight_layout()
fig.savefig(FIGS / "final_E_lift_by_season.png")
plt.close(fig)

# Fig F: top 10 races by lift
top10 = df.sort_values("lift_pts", ascending=False).head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 4.5))
labels = [f"{r.grand_prix.replace(' Grand Prix', '')} {int(r.season)}" for r in top10.itertuples()]
colors = ["#d62728" if c is True else "#7f7f7f" for c in top10["is_close"]]
ax.barh(labels, top10["lift_pts"], color=colors)
ax.set_xlabel("Race-week lift (index points)")
ax.set_title("Ten largest race-week lifts\n(red = close title fight, grey = not close / season opener)")
fig.tight_layout()
fig.savefig(FIGS / "final_F_top10_lift.png")
plt.close(fig)

# Fig G: driver strengths
if not drv.empty:
    d2 = drv.sort_values("rho")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(d2["driver"], d2["rho"], color=["#d62728" if p < 0.05 else "#9aa0a6" for p in d2["p"]])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Spearman rho with race-week lift (red = p < 0.05)")
    ax.set_title("What moves race-week lift?")
    fig.tight_layout()
    fig.savefig(FIGS / "final_G_drivers.png")
    plt.close(fig)

say("\nSaved: reports/final_race_metrics.csv, reports/final_results.txt, reports/figures/final_*.png")
(REPORTS / "final_results.txt").write_text("\n".join(OUT_LINES), encoding="utf-8")