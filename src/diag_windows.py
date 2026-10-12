import pandas as pd

races = pd.read_csv("data/raw/races.csv")
trends = pd.read_csv("data/processed/trends_master.csv")
trends["date"] = pd.to_datetime(trends["date"], errors="coerce")

races = races[~races["Round"].astype(str).str.contains("Source", case=False, na=False)].copy()
races["round_num"] = pd.to_numeric(
    races["Round"].astype(str).str.replace(r"\[.*?\]", "", regex=True), errors="coerce")
races = races.dropna(subset=["round_num"])
races["race_id"] = races["season"].astype(str) + "_R" + races["round_num"].astype(int).astype(str)
ds = (races["Race date"].astype(str).str.replace(r"\[.*?\]", "", regex=True)
      + " " + races["season"].astype(str)).str.replace(r"\s+", " ", regex=True).str.strip()
races["race_date"] = pd.to_datetime(ds, errors="coerce")

w = trends.groupby("race_id")["date"].agg(["min", "max", "count"]).reset_index()
m = races[["race_id", "season", "Grand Prix", "race_date"]].merge(w, on="race_id", how="left")
# A correct window runs race_date-60 .. race_date+30
m["expected_min"] = m["race_date"] - pd.Timedelta(days=60)
m["shift_days"] = (m["min"] - m["expected_min"]).dt.days
m["race_day_in_window"] = (m["race_date"] >= m["min"]) & (m["race_date"] <= m["max"])

pd.set_option("display.width", 200)
print(m.groupby("season")[["shift_days"]].describe())
print("\nRaces whose window is misaligned:")
print(m[m["shift_days"] != 0][["race_id", "Grand Prix", "race_date", "min", "max", "shift_days"]]
      .head(25).to_string(index=False))
print("\nSample raw date strings:")
print(races.groupby("season")["Race date"].first())