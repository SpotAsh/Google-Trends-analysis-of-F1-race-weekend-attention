"""
10_frequent_pattern_mining.py

Runs apriori and fpgrowth at multiple min_support levels, generates
association rules, and saves results for report discussion.
"""

import time
from pathlib import Path
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

binary_df = pd.read_csv(PROCESSED_DIR / "f1_binary_features.csv")
race_ids = binary_df["race_id"]
X = binary_df.drop(columns=["race_id"]).astype(bool)

print(f"Binary feature matrix: {X.shape[0]} races, {X.shape[1]} binary columns")
print(f"Columns: {list(X.columns)}\n")

# ---------------------------------------------------------------------------
# PART 1: Support sensitivity — apriori vs fpgrowth at multiple thresholds
# ---------------------------------------------------------------------------
support_levels = [0.05, 0.1, 0.2]
sensitivity_rows = []

for support in support_levels:
    for algo_name, algo_fn in [("apriori", apriori), ("fpgrowth", fpgrowth)]:
        start = time.time()
        itemsets = algo_fn(X, min_support=support, use_colnames=True)
        elapsed = time.time() - start
        sensitivity_rows.append({
            "min_support": support,
            "algorithm": algo_name,
            "num_itemsets": len(itemsets),
            "runtime_seconds": round(elapsed, 4),
        })
        print(f"min_support={support}, {algo_name}: {len(itemsets)} itemsets in {elapsed:.4f}s")

sensitivity_df = pd.DataFrame(sensitivity_rows)
sensitivity_out = REPORTS_DIR / "fpm_support_sensitivity.csv"
sensitivity_df.to_csv(sensitivity_out, index=False)
print(f"\nSaved support sensitivity table to: {sensitivity_out.resolve()}")

# ---------------------------------------------------------------------------
# PART 2: Association rules at a working min_support level
# ---------------------------------------------------------------------------
WORKING_SUPPORT = 0.1
print(f"\nGenerating association rules at min_support={WORKING_SUPPORT}...")

frequent_itemsets = fpgrowth(X, min_support=WORKING_SUPPORT, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules.sort_values("lift", ascending=False)


def is_trivial(row):
    """Flags rules where antecedent/consequent are dummy variables of the
    same original categorical column (e.g. season_2022 -> season_2022-type
    tautologies), which carry no real information."""
    ant = set(row["antecedents"])
    con = set(row["consequents"])
    all_items = ant | con
    season_items = {i for i in all_items if str(i).startswith("season_")}
    circuit_items = {i for i in all_items if str(i).startswith("circuit_type_")}
    return len(season_items) > 1 or len(circuit_items) > 1


rules["is_trivial"] = rules.apply(is_trivial, axis=1)
non_trivial_rules = rules[~rules["is_trivial"]].copy()

rules_out = REPORTS_DIR / "fpm_rules_all.csv"
rules.to_csv(rules_out, index=False)
print(f"Saved all {len(rules)} rules to: {rules_out.resolve()}")

top_rules_out = REPORTS_DIR / "fpm_rules_top_nontrivial.csv"
non_trivial_rules.head(15).to_csv(top_rules_out, index=False)
print(f"Saved top 15 non-trivial rules to: {top_rules_out.resolve()}")

print(f"\nTop 10 non-trivial rules by lift:")
display_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
print(non_trivial_rules[display_cols].head(10).to_string(index=False))

print("\n" + "=" * 70)
print("SUMMARY FOR REPORT DISCUSSION")
print("=" * 70)
print(f"Total frequent itemsets at min_support={WORKING_SUPPORT}: {len(frequent_itemsets)}")
print(f"Total rules generated (lift >= 1.0): {len(rules)}")
print(f"Non-trivial rules (excluding same-category tautologies): {len(non_trivial_rules)}")
print("\nSupport sensitivity (see fpm_support_sensitivity.csv for full table):")
print(sensitivity_df.pivot(index="min_support", columns="algorithm", values="num_itemsets"))
print("\nNote for report: apriori and fpgrowth should produce IDENTICAL itemset")
print("counts at the same min_support (they solve the same problem via different")
print("algorithms) — any runtime difference reflects algorithmic efficiency, not")
print("different results. Discuss which was faster and why fpgrowth tends to")
print("scale better on larger itemsets.")