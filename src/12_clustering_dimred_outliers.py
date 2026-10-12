"""
12_clustering_dimred_outliers.py   (Mid-Report 2)

Clustering, dimensionality reduction and outlier detection on race-weekend
"attention profiles".

UNIT OF ANALYSIS: one row per race weekend. Each race is described by the SHAPE
of its Google Trends curve (all in 0-100 index points, computed from the daily
series in trends_master.csv):

    log_baseline     log1p(mean interest, days -60..-14)   off-season interest level
    lift_pts         race-week mean - baseline mean         how far race week sits above baseline
    race_week_share  race-week volume / whole-window volume how concentrated the attention is
    buildup_pts      mean(days -13..-8) - baseline mean     anticipation before race week
    drop_pts         race-week mean - mean(days +1..+7)     how fast interest falls after the race

Context variables (championship gap, season, street circuit, round) are NOT
used to build clusters. They are used afterwards to PROFILE the clusters, so
"does closeness explain the clusters?" stays a real test, not a built-in answer.

Inputs : data/processed/trends_master.csv, f1_final_features.csv, f1_binary_features.csv
Outputs: reports/mid2_results.txt, reports/mid2_assignments.csv,
         reports/mid2_cluster_profiles.csv, reports/figures/mid2_*.png
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, cophenet
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROC = Path("data/processed")
REPORTS = Path("reports")
FIGS = REPORTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
SEED = 42
np.random.seed(SEED)

LINES = []


def say(text=""):
    print(text)
    LINES.append(str(text))


plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10})
CLUSTER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def ccol(i):
    return "#bbbbbb" if i < 0 else CLUSTER_COLORS[i % len(CLUSTER_COLORS)]


# =============================================================== 1. FEATURES
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
assert trends["date"].notna().all() and feat["race_date"].notna().all(), "unparseable dates"

t = trends.merge(feat[["race_id", "race_date"]], on="race_id", how="inner")
t["d"] = (t["date"] - t["race_date"]).dt.days

rows = []
for rid, g in t.groupby("race_id"):
    s = g.set_index("d")["search_interest"]
    base = s[(s.index >= -60) & (s.index <= -14)]
    pre = s[(s.index >= -13) & (s.index <= -8)]
    rw = s[(s.index >= -7) & (s.index <= 0)]
    post = s[(s.index >= 1) & (s.index <= 7)]
    if min(len(base), len(pre), len(rw), len(post)) == 0:
        continue
    b = float(base.mean())
    rows.append({
        "race_id": rid,
        "baseline_mean": b,
        "log_baseline": np.log1p(b),
        "lift_pts": float(rw.mean()) - b,
        "race_week_share": float(rw.sum()) / float(s.sum()) if s.sum() > 0 else np.nan,
        "buildup_pts": float(pre.mean()) - b,
        "drop_pts": float(rw.mean()) - float(post.mean()),
    })
F = pd.DataFrame(rows)

ctx_cols = ["race_id", "season", "round_num", "grand_prix", "circuit",
            "championship_gap_entering_race", "is_close_championship_race",
            "is_covid_season"]
df = F.merge(feat[ctx_cols], on="race_id", how="inner").merge(
    binf[["race_id", "circuit_type_street"]], on="race_id", how="left")
df["is_street"] = df["circuit_type_street"].fillna(False).astype(bool)
df["is_covid_season"] = df["is_covid_season"].fillna(False).astype(bool)
df["is_close"] = df["is_close_championship_race"].map(
    {True: True, False: False, "True": True, "False": False})
df = df.dropna(subset=["log_baseline", "lift_pts", "race_week_share", "buildup_pts", "drop_pts"]).reset_index(drop=True)

FEATS = ["log_baseline", "lift_pts", "race_week_share", "buildup_pts", "drop_pts"]
X = df[FEATS].values
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
n = len(df)

say("=" * 72)
say(f"DATA: {n} race weekends, seasons {sorted(df['season'].unique())}, {len(FEATS)} shape features")
say("=" * 72)
say(df[FEATS].describe().round(2).T[["mean", "std", "min", "50%", "max"]].to_string())
say("\nFeatures are standardized (z-scores) before every distance-based method, "
    "so no feature dominates because of its units.")

corr = pd.DataFrame(Xs, columns=FEATS).corr()
say("\nFeature correlation matrix (shows redundancy that dimensionality reduction can compress):")
say(corr.round(2).to_string())

fig, ax = plt.subplots(figsize=(5.2, 4.4))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(FEATS)))
ax.set_yticks(range(len(FEATS)))
ax.set_xticklabels(FEATS, rotation=40, ha="right")
ax.set_yticklabels(FEATS)
for i in range(len(FEATS)):
    for j in range(len(FEATS)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.046)
ax.set_title("Feature correlations (standardized)")
fig.tight_layout()
fig.savefig(FIGS / "mid2_corr_heatmap.png")
plt.close(fig)

# =============================================================== 2. CLUSTERING
say("\n" + "=" * 72)
say("1. CLUSTERING")
say("=" * 72)

# ---- K-means: choose k
ks = list(range(2, 9))
inertias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Xs)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(Xs, km.labels_))
sil_by_k = dict(zip(ks, sils))
# k = 2 almost always wins on silhouette simply because it is the coarsest split,
# so we choose among 3..6 (granular enough to be interpretable) and report k=2.
K = max(range(3, 7), key=lambda k: sil_by_k[k])
say("\nK-means model selection (standardized features, n_init=20):")
for k in ks:
    say(f"  k={k}: inertia = {inertias[ks.index(k)]:.1f}, silhouette = {sil_by_k[k]:.3f}"
        + ("   <-- chosen (best silhouette among k=3..6)" if k == K else ""))
say(f"Chosen K = {K}. (k=2 silhouette = {sil_by_k[2]:.3f}; coarsest split, so not used.)")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].plot(ks, inertias, "o-", color="#1f77b4")
ax[0].set_xlabel("k")
ax[0].set_ylabel("Inertia (within-cluster SS)")
ax[0].set_title("Elbow")
ax[1].plot(ks, sils, "o-", color="#d62728")
ax[1].axvline(K, color="grey", ls=":")
ax[1].set_xlabel("k")
ax[1].set_ylabel("Silhouette")
ax[1].set_title("Silhouette by k")
fig.tight_layout()
fig.savefig(FIGS / "mid2_k_selection.png")
plt.close(fig)

km = KMeans(n_clusters=K, n_init=20, random_state=SEED).fit(Xs)
lab_km = km.labels_

# ---- Hierarchical (Ward)
Z = linkage(Xs, method="ward")
coph, _ = cophenet(Z, pdist(Xs))
lab_h = fcluster(Z, K, criterion="maxclust") - 1
thr = (Z[n - K - 1, 2] + Z[n - K, 2]) / 2
say(f"\nHierarchical clustering: Ward linkage, Euclidean distance on standardized features, cut at K = {K}.")
say(f"Cophenetic correlation = {coph:.3f} (how faithfully the dendrogram preserves pairwise distances).")

fig, ax = plt.subplots(figsize=(10, 4))
dendrogram(Z, truncate_mode="lastp", p=min(30, n), color_threshold=thr, ax=ax,
           leaf_rotation=90, leaf_font_size=8, no_labels=False)
ax.axhline(thr, color="grey", ls="--", lw=1)
ax.set_title(f"Ward dendrogram (truncated to 30 leaves; dashed line = cut into {K} clusters)")
ax.set_ylabel("Ward distance")
fig.tight_layout()
fig.savefig(FIGS / "mid2_dendrogram.png")
plt.close(fig)

# ---- DBSCAN: choose minPts / eps by a small, transparent grid
best = None
for minpts in [4, 5, 6, 8]:
    nn = NearestNeighbors(n_neighbors=minpts).fit(Xs)
    kd = np.sort(nn.kneighbors(Xs)[0][:, minpts - 1])
    for pct in range(60, 98, 3):
        eps = float(np.percentile(kd, pct))
        lab = DBSCAN(eps=eps, min_samples=minpts).fit_predict(Xs)
        ncl = len(set(lab)) - (1 if -1 in lab else 0)
        noise = float((lab == -1).mean())
        if ncl >= 2 and noise <= 0.25:
            mask = lab != -1
            if len(set(lab[mask])) >= 2:
                s = silhouette_score(Xs[mask], lab[mask])
                if best is None or s > best["sil"]:
                    best = {"minpts": minpts, "eps": eps, "sil": s, "lab": lab, "ncl": ncl, "noise": noise}
if best is None:  # fallback: standard heuristic, even if it yields one cluster
    minpts = 5
    nn = NearestNeighbors(n_neighbors=minpts).fit(Xs)
    kd = np.sort(nn.kneighbors(Xs)[0][:, minpts - 1])
    eps = float(np.percentile(kd, 85))
    lab = DBSCAN(eps=eps, min_samples=minpts).fit_predict(Xs)
    best = {"minpts": minpts, "eps": eps, "sil": np.nan, "lab": lab,
            "ncl": len(set(lab)) - (1 if -1 in lab else 0), "noise": float((lab == -1).mean())}
    say("\n[note] No DBSCAN setting gave >= 2 clusters with <= 25% noise; using the 85th-percentile k-distance fallback.")
lab_db = best["lab"]
say(f"\nDBSCAN: minPts = {best['minpts']}, eps = {best['eps']:.2f}  ->  {best['ncl']} clusters, "
    f"{(lab_db == -1).sum()} noise points ({best['noise']:.0%}), silhouette (non-noise) = {best['sil']:.3f}")
say("Chosen by grid search over minPts in {4,5,6,8} and eps at the 60th-97th percentile of the "
    "k-distance curve, keeping settings with >= 2 clusters and <= 25% noise, maximizing silhouette. "
    "minPts is small because the data has only ~100-200 rows in 5 dimensions.")

nn = NearestNeighbors(n_neighbors=best["minpts"]).fit(Xs)
kd_sorted = np.sort(nn.kneighbors(Xs)[0][:, best["minpts"] - 1])
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(kd_sorted, color="#1f77b4")
ax.axhline(best["eps"], color="#d62728", ls="--", label=f"chosen eps = {best['eps']:.2f}")
ax.set_xlabel("Races (sorted by distance)")
ax.set_ylabel(f"Distance to neighbour #{best['minpts'] - 1}")
ax.set_title("DBSCAN k-distance curve")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGS / "mid2_dbscan_kdist.png")
plt.close(fig)

# ---- Compare methods
say("\n--- Comparison of clustering methods ---")
methods = {"KMeans": lab_km, "Ward": lab_h, "DBSCAN": lab_db}
sil_rows = []
for name, lab in methods.items():
    mask = lab != -1
    ok = len(set(lab[mask])) >= 2
    s = silhouette_score(Xs[mask], lab[mask]) if ok else np.nan
    sil_rows.append((name, len(set(lab[mask])), int((~mask).sum()), s))
    say(f"  {name:<7} clusters = {len(set(lab[mask]))}, noise = {int((~mask).sum())}, silhouette = {s:.3f}")
names = list(methods)
say("\nAgreement between methods (ARI: 1 = identical partitions, 0 = chance; NMI similar, 0-1):")
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        ari = adjusted_rand_score(methods[names[i]], methods[names[j]])
        nmi = normalized_mutual_info_score(methods[names[i]], methods[names[j]])
        say(f"  {names[i]} vs {names[j]}: ARI = {ari:.3f}, NMI = {nmi:.3f}")
say("\nK-means vs Ward contingency table:")
say(pd.crosstab(pd.Series(lab_km, name="KMeans"), pd.Series(lab_h, name="Ward")).to_string())
say("\nK-means vs DBSCAN contingency table (-1 = DBSCAN noise):")
say(pd.crosstab(pd.Series(lab_km, name="KMeans"), pd.Series(lab_db, name="DBSCAN")).to_string())

# ---- Profile K-means clusters
df["km"] = lab_km
df["ward"] = lab_h
df["dbscan"] = lab_db
Zdf = pd.DataFrame(Xs, columns=FEATS)
Zdf["km"] = lab_km
zmeans = Zdf.groupby("km")[FEATS].mean()


def describe_cluster(zrow):
    parts = []
    names_ = {"log_baseline": "baseline", "lift_pts": "race-week lift", "race_week_share": "concentration",
              "buildup_pts": "pre-race buildup", "drop_pts": "post-race drop"}
    for f_ in FEATS:
        if zrow[f_] >= 0.5:
            parts.append(f"high {names_[f_]}")
        elif zrow[f_] <= -0.5:
            parts.append(f"low {names_[f_]}")
    return ", ".join(parts) if parts else "near average on every feature"


prof_rows = []
say("\n--- K-means cluster profiles (raw-unit means; z-score descriptors) ---")
for c in range(K):
    sub = df[df["km"] == c]
    gsub = sub.dropna(subset=["championship_gap_entering_race"])
    # races nearest the centroid = most typical members
    d_to_c = np.linalg.norm(Xs[df["km"].values == c] - km.cluster_centers_[c], axis=1)
    typical = sub.iloc[np.argsort(d_to_c)[:3]]
    typ = "; ".join(f"{r.grand_prix.replace(' Grand Prix', '')} {int(r.season)}" for r in typical.itertuples())
    row = {"cluster": c, "n": len(sub), "description": describe_cluster(zmeans.loc[c]),
           "median_gap": gsub["championship_gap_entering_race"].median(),
           "share_close": (gsub["is_close"] == True).mean() if len(gsub) else np.nan,
           "share_street": sub["is_street"].mean(),
           "share_covid": sub["is_covid_season"].mean(),
           "typical_races": typ}
    for f_ in FEATS:
        row[f_ + "_mean"] = sub[f_].mean()
    prof_rows.append(row)
    say(f"\nCluster {c}  (n = {len(sub)}): {row['description']}")
    say("   means: " + ", ".join(f"{f_}={sub[f_].mean():.2f}" for f_ in FEATS))
    say(f"   median championship gap = {row['median_gap']:.0f}, share close (<25 pts) = {row['share_close']:.0%}, "
        f"share street circuits = {row['share_street']:.0%}, share 2020 COVID = {row['share_covid']:.0%}")
    say(f"   most typical races: {typ}")
    say("   seasons: " + ", ".join(f"{int(k)}: {int(v)}" for k, v in sub["season"].value_counts().sort_index().items()))
prof = pd.DataFrame(prof_rows)
prof.to_csv(REPORTS / "mid2_cluster_profiles.csv", index=False)

say("\nDo the clusters line up with context variables NOT used to build them?")
groups = [g["championship_gap_entering_race"].dropna().values for _, g in df.groupby("km")]
groups = [g for g in groups if len(g) > 1]
if len(groups) >= 2:
    H, p = stats.kruskal(*groups)
    say(f"  championship gap across clusters: Kruskal-Wallis H = {H:.2f}, p = {p:.3f}")
ct = pd.crosstab(df["km"], df["is_close"].dropna())
if ct.shape[0] >= 2 and ct.shape[1] == 2:
    chi, p, _, _ = stats.chi2_contingency(ct)
    say(f"  close-championship share across clusters: chi-square = {chi:.2f}, p = {p:.3f}")
ct = pd.crosstab(df["km"], df["season"])
chi, p, _, _ = stats.chi2_contingency(ct)
say(f"  season mix across clusters: chi-square = {chi:.2f}, p = {p:.3f}")
ct = pd.crosstab(df["km"], df["is_street"])
if ct.shape[1] == 2:
    chi, p, _, _ = stats.chi2_contingency(ct)
    say(f"  street-circuit share across clusters: chi-square = {chi:.2f}, p = {p:.3f}")
ct = pd.crosstab(df["km"], df["is_covid_season"])
if ct.shape[1] == 2 and ct.shape[0] >= 2:
    chi, p, _, _ = stats.chi2_contingency(ct)
    say(f"  2020 COVID-season share across clusters: chi-square = {chi:.2f}, p = {p:.3f}")

# ---- Cluster-profile heatmap
fig, ax = plt.subplots(figsize=(6.4, 0.9 + 0.7 * K))
im = ax.imshow(zmeans.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(len(FEATS)))
ax.set_xticklabels(FEATS, rotation=30, ha="right")
ax.set_yticks(range(K))
ax.set_yticklabels([f"Cluster {c} (n={int((lab_km == c).sum())})" for c in range(K)])
for i in range(K):
    for j in range(len(FEATS)):
        ax.text(j, i, f"{zmeans.values[i, j]:+.1f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.046, label="mean z-score")
ax.set_title("K-means cluster profiles")
fig.tight_layout()
fig.savefig(FIGS / "mid2_cluster_profiles.png")
plt.close(fig)

# =============================================================== 3. DIM. REDUCTION
say("\n" + "=" * 72)
say("2. DIMENSIONALITY REDUCTION")
say("=" * 72)

pca = PCA(n_components=len(FEATS), random_state=SEED).fit(Xs)
evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)
P = pca.transform(Xs)
n90 = int(np.searchsorted(cum, 0.90) + 1)
say("\nPCA variance explained:")
for i, (e, c_) in enumerate(zip(evr, cum), start=1):
    say(f"  PC{i}: {e:.1%}  (cumulative {c_:.1%})")
say(f"PC1+PC2 explain {cum[1]:.1%} of the variance; {n90} components are needed for 90%.")
load = pd.DataFrame(pca.components_[:3].T, index=FEATS, columns=["PC1", "PC2", "PC3"])
say("\nPCA loadings (first three components):")
say(load.round(2).to_string())

fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.bar(range(1, len(evr) + 1), evr, color="#1f77b4", label="Per component")
ax.plot(range(1, len(evr) + 1), cum, "o-", color="#d62728", label="Cumulative")
ax.axhline(0.9, color="grey", ls=":")
ax.set_xlabel("Principal component")
ax.set_ylabel("Variance explained")
ax.set_ylim(0, 1.05)
ax.set_title("PCA scree plot")
ax.legend(frameon=False, loc="center right")
fig.tight_layout()
fig.savefig(FIGS / "mid2_pca_scree.png")
plt.close(fig)

perp = int(min(30, max(5, (n - 1) // 4)))
tsne = TSNE(n_components=2, perplexity=perp, init="pca", learning_rate="auto", random_state=SEED)
T = tsne.fit_transform(Xs)
embeds = {"PCA (2 PCs)": P[:, :2], f"t-SNE (perplexity {perp})": T}
try:
    import umap  # optional: pip install umap-learn
    U = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED).fit_transform(Xs)
    embeds["UMAP"] = U
except Exception:
    say("\n(umap-learn is not installed, so UMAP is skipped; PCA and t-SNE satisfy the two-technique requirement.)")

say("\nStructure preservation (how faithful is each 2-D picture of the 5-D data?):")
d_hi = pdist(Xs)
for name, Y in embeds.items():
    tw = trustworthiness(Xs, Y, n_neighbors=10)
    rho, _ = stats.spearmanr(d_hi, pdist(Y))
    say(f"  {name:<22} local: trustworthiness (k=10) = {tw:.3f}   global: Spearman of pairwise distances = {rho:.3f}")
say("(trustworthiness near 1 = neighbours in the picture were neighbours in the data; "
    "distance correlation near 1 = global layout preserved.)")

# Before / after picture
panels = [("Before: 2 of 5 standardized features", Xs[:, [FEATS.index("log_baseline"), FEATS.index("lift_pts")]],
           ("log_baseline (z)", "lift_pts (z)"))]
for name, Y in embeds.items():
    panels.append((f"After: {name}", Y, ("dim 1", "dim 2")))
fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4))
for ax, (title, Y, labs) in zip(np.atleast_1d(axes), panels):
    for c in range(K):
        m_ = lab_km == c
        ax.scatter(Y[m_, 0], Y[m_, 1], s=22, color=ccol(c), alpha=0.8, label=f"Cluster {c}")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(labs[0])
    ax.set_ylabel(labs[1])
np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIGS / "mid2_dr_before_after.png")
plt.close(fig)

# Effect of DR on clustering
say("\nDoes dimensionality reduction change the clustering? (K-means, same K)")
km_full = lab_km
sil_full = silhouette_score(Xs, km_full)
say(f"  full 5-D features         : silhouette = {sil_full:.3f}  (reference)")
for name, Y in [(f"first {n90} PCs (90% variance)", P[:, :n90]), ("first 2 PCs", P[:, :2]),
                ("t-SNE 2-D", T)]:
    kmY = KMeans(n_clusters=K, n_init=20, random_state=SEED).fit(Y)
    s_Y = silhouette_score(Y, kmY.labels_)
    ari = adjusted_rand_score(km_full, kmY.labels_)
    say(f"  {name:<26}: silhouette (in that space) = {s_Y:.3f}, ARI vs full-feature clusters = {ari:.3f}")
say("  (Silhouettes in different spaces are not directly comparable; ARI shows whether the same groups emerge. "
    "Clustering on t-SNE coordinates is shown for completeness only, since t-SNE distances between "
    "clusters are not meaningful.)")

# =============================================================== 4. OUTLIERS
say("\n" + "=" * 72)
say("3. OUTLIER DETECTION")
say("=" * 72)
say("""
DEFINITION: an outlier is a race weekend whose attention profile (5 standardized features)
is unusual compared with the other races: far from its neighbours in feature space, or
extreme on one feature. It can be (a) a rare but valid event, (b) an artifact of Google
Trends (an empty or zero baseline, an ambiguous search term), or (c) a genuinely
different kind of race weekend.""")

CONT = 0.05
flags = pd.DataFrame(index=df.index)
zmax = np.abs(Xs).max(axis=1)
flags["z>3"] = zmax > 3
iso = IsolationForest(n_estimators=300, contamination=CONT, random_state=SEED).fit(Xs)
flags["IsolationForest"] = iso.predict(Xs) == -1
lof = LocalOutlierFactor(n_neighbors=15, contamination=CONT)
flags["LOF"] = lof.fit_predict(Xs) == -1
knn = NearestNeighbors(n_neighbors=6).fit(Xs)
knn_score = knn.kneighbors(Xs)[0][:, 1:].mean(axis=1)  # mean distance to 5 nearest neighbours
flags["kNN distance"] = knn_score > np.percentile(knn_score, 100 * (1 - CONT))
ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=CONT).fit(Xs)
flags["OneClassSVM"] = ocsvm.predict(Xs) == -1

say("\nMethods and settings (contamination / nu / percentile = 5% for the model-based methods):")
say("  z>3            : any feature with |z| > 3 (statistical threshold, univariate)")
say("  IsolationForest: 300 trees, contamination 0.05 (random-partition isolation)")
say("  LOF            : 15 neighbours, contamination 0.05 (local density vs neighbours)")
say("  kNN distance   : mean distance to the 5 nearest neighbours, top 5% flagged")
say("  OneClassSVM    : RBF kernel, gamma='scale', nu 0.05 (boundary around the bulk of the data)")
say("\nRaces flagged per method:")
say(flags.sum().to_string())

jac = pd.DataFrame(index=flags.columns, columns=flags.columns, dtype=float)
for a in flags.columns:
    for b in flags.columns:
        inter = (flags[a] & flags[b]).sum()
        union = (flags[a] | flags[b]).sum()
        jac.loc[a, b] = inter / union if union else np.nan
say("\nAgreement between methods (Jaccard overlap of flagged sets):")
say(jac.round(2).to_string())

df["n_flags"] = flags.sum(axis=1).values
MIN_VOTES = 3
if (df["n_flags"] >= MIN_VOTES).sum() == 0:
    MIN_VOTES = 2
df["is_outlier"] = df["n_flags"] >= MIN_VOTES
say(f"\nConsensus rule: flagged by >= {MIN_VOTES} of {flags.shape[1]} methods -> {int(df['is_outlier'].sum())} outliers "
    f"({df['is_outlier'].mean():.1%} of races).")

zdf = pd.DataFrame(Xs, columns=FEATS)
out_rows = []
for i in df.index[df["is_outlier"]]:
    drv = zdf.loc[i].abs().idxmax()
    zval = zdf.loc[i, drv]
    r = df.loc[i]
    if r["baseline_mean"] <= 0.05:
        cause = "near-empty baseline (Trends rounds tiny volumes to 0): likely data-resolution artifact"
    elif drv in ("lift_pts", "race_week_share", "drop_pts") and zval > 0:
        cause = "unusually event-concentrated weekend: rare but plausibly valid"
    elif drv in ("lift_pts", "race_week_share", "drop_pts") and zval < 0:
        cause = "weak race-week signal: check that the search term is not diluted by non-F1 searches"
    elif drv == "log_baseline" and zval > 0:
        cause = "high year-round interest in the search term (host-country/term effect)"
    else:
        cause = "unusual pre-race buildup or baseline pattern"
    out_rows.append({"race_id": r["race_id"], "grand_prix": r["grand_prix"], "season": int(r["season"]),
                     "votes": int(r["n_flags"]), "driver_feature": drv, "driver_z": round(float(zval), 2),
                     "baseline_mean": round(float(r["baseline_mean"]), 2), "lift_pts": round(float(r["lift_pts"]), 1),
                     "gap": r["championship_gap_entering_race"], "likely_cause": cause})
outs = pd.DataFrame(out_rows).sort_values(["votes", "driver_z"], ascending=[False, False])
say("\nConsensus outliers, what makes them unusual, and the most likely cause (heuristic, verify before claiming):")
if len(outs):
    say(outs.to_string(index=False))
else:
    say("  none")
if len(outs):
    nclose = int((df.loc[df["is_outlier"], "is_close"] == True).sum())
    say(f"\n{nclose} of {len(outs)} consensus outliers occurred in a close-championship race "
        f"(base rate across all races with a gap: {(df['is_close'] == True).sum() / df['is_close'].notna().sum():.0%}).")
    say(f"DBSCAN labelled {int((df.loc[df['is_outlier'], 'dbscan'] == -1).sum())} of the {len(outs)} consensus outliers as noise "
        "(clustering and outlier detection agreeing is reassuring).")

# ---- Impact of removing outliers
say("\n--- Impact of outliers on clustering and PCA ---")
keep = ~df["is_outlier"].values
Xk_raw = X[keep]
sc2 = StandardScaler().fit(Xk_raw)
Xk = sc2.transform(Xk_raw)
km_k = KMeans(n_clusters=K, n_init=20, random_state=SEED).fit(Xk)
sil_k = silhouette_score(Xk, km_k.labels_)
sil_full_keep = silhouette_score(Xs[keep], lab_km[keep])
ari_k = adjusted_rand_score(lab_km[keep], km_k.labels_)
say(f"  K-means (K={K}) with outliers   : silhouette = {sil_full:.3f} (all races), {sil_full_keep:.3f} (non-outlier races only)")
say(f"  K-means (K={K}) without outliers: silhouette = {sil_k:.3f}  (n = {int(keep.sum())})")
say(f"  ARI between the two clusterings on the retained races = {ari_k:.3f} "
    "(1 = removing outliers changed nothing)")
pca_k = PCA(n_components=len(FEATS), random_state=SEED).fit(Xk)
say(f"  PCA PC1+PC2 variance: {cum[1]:.1%} with outliers vs {np.cumsum(pca_k.explained_variance_ratio_)[1]:.1%} without")
gk = df.loc[keep].copy()
gk["km2"] = km_k.labels_
groups = [g["championship_gap_entering_race"].dropna().values for _, g in gk.groupby("km2")]
groups = [g for g in groups if len(g) > 1]
if len(groups) >= 2:
    H2, p2 = stats.kruskal(*groups)
    say(f"  gap-vs-cluster association (Kruskal-Wallis) without outliers: p = {p2:.3f}")

# t-SNE / PCA pictures with outliers marked
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, (title, Y) in zip(axes, [("PCA", P[:, :2]), (f"t-SNE (perplexity {perp})", T)]):
    ax.scatter(Y[~df["is_outlier"].values, 0], Y[~df["is_outlier"].values, 1], s=18, color="#9aa0a6", alpha=0.7,
               label="Not an outlier")
    sel = df["is_outlier"].values
    ax.scatter(Y[sel, 0], Y[sel, 1], s=70 + 25 * df.loc[sel, "n_flags"].values, color="#d62728",
               edgecolor="black", linewidth=1, label=f"Consensus outlier (>= {MIN_VOTES} methods)")
    for i in np.where(sel)[0][:12]:
        ax.annotate(f"{df.loc[i, 'grand_prix'].replace(' Grand Prix', '')} {int(df.loc[i, 'season'])}",
                    (Y[i, 0], Y[i, 1]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIGS / "mid2_outliers_embeddings.png")
plt.close(fig)

# method agreement
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].bar(flags.columns, flags.sum().values, color="#4c72b0")
ax[0].set_ylabel("Races flagged")
ax[0].set_title("Flags per method")
ax[0].tick_params(axis="x", rotation=25)
im = ax[1].imshow(jac.values.astype(float), cmap="Blues", vmin=0, vmax=1)
ax[1].set_xticks(range(len(flags.columns)))
ax[1].set_yticks(range(len(flags.columns)))
ax[1].set_xticklabels(flags.columns, rotation=40, ha="right")
ax[1].set_yticklabels(flags.columns)
for i in range(len(flags.columns)):
    for j in range(len(flags.columns)):
        ax[1].text(j, i, f"{jac.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax[1].set_title("Jaccard agreement")
fig.tight_layout()
fig.savefig(FIGS / "mid2_outlier_agreement.png")
plt.close(fig)

# before/after removal in PCA space
Pk = pca_k.transform(Xk)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for c in range(K):
    m_ = lab_km == c
    axes[0].scatter(P[m_, 0], P[m_, 1], s=22, color=ccol(c), alpha=0.8, label=f"Cluster {c}")
sel = df["is_outlier"].values
axes[0].scatter(P[sel, 0], P[sel, 1], s=90, facecolors="none", edgecolors="black", linewidths=1.4, label="Outlier")
axes[0].set_title("With outliers (K-means on all races)")
for c in range(K):
    m_ = km_k.labels_ == c
    axes[1].scatter(Pk[m_, 0], Pk[m_, 1], s=22, color=ccol(c), alpha=0.8)
axes[1].set_title("Outliers removed (K-means refit)")
for ax in axes:
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIGS / "mid2_outlier_impact.png")
plt.close(fig)

# clusters on PCA, three methods
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (name, lab) in zip(axes, methods.items()):
    for c in sorted(set(lab)):
        m_ = lab == c
        ax.scatter(P[m_, 0], P[m_, 1], s=22, color=ccol(c), alpha=0.85,
                   label="noise" if c == -1 else f"Cluster {c}", marker="x" if c == -1 else "o")
    ax.set_title(name)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False, fontsize=7)
fig.suptitle("Three clustering methods, drawn in the same PCA plane", y=1.02)
fig.tight_layout()
fig.savefig(FIGS / "mid2_clusters_three_methods.png", bbox_inches="tight")
plt.close(fig)

# =============================================================== 5. SAVE
assign = df[["race_id", "season", "grand_prix", "championship_gap_entering_race", "is_close",
             "is_covid_season"] + FEATS +
            ["km", "ward", "dbscan", "n_flags", "is_outlier"]].copy()
assign["pc1"], assign["pc2"] = P[:, 0], P[:, 1]
assign["tsne1"], assign["tsne2"] = T[:, 0], T[:, 1]
assign.to_csv(REPORTS / "mid2_assignments.csv", index=False)
say("\nSaved: reports/mid2_assignments.csv, reports/mid2_cluster_profiles.csv, reports/mid2_results.txt, "
    "reports/figures/mid2_*.png")
(REPORTS / "mid2_results.txt").write_text("\n".join(LINES), encoding="utf-8")