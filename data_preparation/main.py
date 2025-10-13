import pandas as pd
import numpy as np

#Feature 1
def Feature_1(results_csv= "results.csv", races_csv= "races.csv", n: int = 5, min_prev: int = 1) -> pd.DataFrame:
    """
    Per race, per driver rolling average grid over the previous n races.
    Excludes grid == 0. Rounds to 2 decimals.

    Returns: DataFrame [driverId, raceId, grid_start_position]
    """
    """
    Per race, per driver rolling average grid over the previous n races.
    Excludes grid == 0. Rounds to 2 decimals.

    Returns: DataFrame [driverId, raceId, circuitId, constructorId, grid_start_position]
    """
    # Load; include constructorId
    results = pd.read_csv(results_csv, usecols=["raceId","driverId","constructorId","grid"])
    races   = pd.read_csv(races_csv,   usecols=["raceId","year","round","circuitId"])

    # Defensive: if duplicates exist per (raceId, driverId), keep best (lowest) grid
    res_dedup = (results.groupby(["raceId","driverId","constructorId"], as_index=False)["grid"].min())

    df = (res_dedup.merge(races, on="raceId", how="left").sort_values(["driverId","year","round","raceId"]).reset_index(drop=True))

    # ignore grid == 0 (e.g., pit lane / unknown)
    df["grid_valid"] = df["grid"].where(df["grid"] > 0)

    # rolling mean of previous n (shift prevents leakage)
    df["grid_start_position"] = (df.groupby("driverId", group_keys=False)["grid_valid"]
          .apply(lambda s: s.shift().rolling(window=n, min_periods=min_prev).mean())
          .round(2)
    )

    return df[["driverId","raceId","circuitId","constructorId","grid_start_position"]]


#Feature 2
def Feature_2(lap_times_csv="lap_times.csv", results_csv="results.csv", races_csv="races.csv", n=5, min_prev=1, fillna_with=0):
    lap_times = pd.read_csv(lap_times_csv, usecols=["raceId","driverId","lap","position"])
    results   = pd.read_csv(results_csv,   usecols=["raceId","driverId","constructorId"])
    races     = pd.read_csv(races_csv,     usecols=["raceId","year","round","circuitId"])

    # Laps led per driver per race
    laps_led_race = (
        lap_times[lap_times["position"] == 1]
        .groupby(["raceId","driverId"], as_index=False)
        .size()
        .rename(columns={"size":"laps_led_in_race"})
    )

    # Include zeros for drivers who led none
    dr_race = (
        results.drop_duplicates()
               .merge(laps_led_race, on=["raceId","driverId"], how="left")
               .fillna({"laps_led_in_race": 0})
    )

    # Chronology + circuit
    df = (
        dr_race.merge(races, on="raceId", how="left")
               .sort_values(["driverId","year","round","raceId"])
               .reset_index(drop=True)
    )

    # Rolling sum over previous n races (no leakage)
    df["driver_standing_laps_led_n"] = (
        df.groupby("driverId", group_keys=False)["laps_led_in_race"]
          .apply(lambda s: s.shift().rolling(window=n, min_periods=min_prev).sum())
    )

    # Select output and avoid chained-assignment warnings
    out = df.loc[:, ["driverId","raceId","circuitId", "constructorId","driver_standing_laps_led_n"]].copy()
    if fillna_with is not None:
        out.loc[:, "driver_standing_laps_led_n"] = out["driver_standing_laps_led_n"].fillna(fillna_with)

    return out


#Feature 3
def Feature_3(results_csv="results.csv", races_csv="races.csv", n=5, min_prev=1, fillna_with=0):

    results= pd.read_csv(results_csv)
    races= pd.read_csv(races_csv)
    # Load data
    df = results.merge(races[['raceId','circuitId','date']], on='raceId', how='left')

    # Order deterministically
    df = df.sort_values(['driverId','date','raceId'])

    # Rolling sum of previous n races (no data leakage)
    g = df.groupby('driverId')['points']
    cs = g.cumsum()

    # sum_{i-n..i-1} = c_{i-1} - c_{i-n-1}
    prev_n = cs.groupby(df['driverId']).shift(n+1)
    prev_1 = cs.groupby(df['driverId']).shift(1)
    df['driver_standing_points_n'] = (prev_1 - prev_n).fillna(0)

    return df[['raceId','driverId','circuitId','constructorId','driver_standing_points_n']]


#Feature 4
def Feature_4(lap_times_csv="lap_times.csv", races_csv="races.csv" ,fillna_with=None):
    laps  = pd.read_csv(lap_times_csv, usecols=["raceId","driverId","lap","milliseconds"])
    races = pd.read_csv(races_csv,     usecols=["raceId","year","round","circuitId"])

    best_lap_race = (
        laps.dropna(subset=["milliseconds"])
            .groupby(["raceId","driverId"], as_index=False)["milliseconds"]
            .min()
            .rename(columns={"milliseconds": "best_lap_ms_in_race"})
    )

    df = (best_lap_race
          .merge(races, on="raceId", how="left")
          .sort_values(["driverId","circuitId","year","round","raceId"])
          .reset_index(drop=True))

    df["driver_circuit_best_lap"] = (
        df.groupby(["driverId","circuitId"], group_keys=False)["best_lap_ms_in_race"]
          .apply(lambda s: s.shift().cummin())
    )

    out = df[["driverId","raceId","circuitId","driver_circuit_best_lap"]].copy()
    if fillna_with is not None:
        out["driver_circuit_best_lap"] = out["driver_circuit_best_lap"].fillna(fillna_with)
    return out


#Feature 5
def Feature_5(results_csv="results.csv", races_csv="races.csv", fillna_with=0):
    res   = pd.read_csv(results_csv, usecols=["raceId","driverId","constructorId","positionOrder"])
    races = pd.read_csv(races_csv,   usecols=["raceId","year","round","circuitId"])

    dr_race = (res.groupby(["raceId","driverId"], as_index=False)["positionOrder"]
                  .min()
                  .rename(columns={"positionOrder":"best_finish"}))
    dr_race["is_win"] = (dr_race["best_finish"] == 1).astype(int)

    cons = res.drop_duplicates(["raceId","driverId"])[["raceId","driverId","constructorId"]]
    df = (dr_race.merge(cons, on=["raceId","driverId"], how="left")
                 .merge(races, on="raceId", how="left")
                 .sort_values(["driverId","circuitId","year","round","raceId"])
                 .reset_index(drop=True))

    df["driver_circuit_wins"] = (
        df.groupby(["driverId","circuitId"], group_keys=False)["is_win"]
          .apply(lambda s: s.shift().cumsum())
    )

    out = df[["driverId","raceId","circuitId","constructorId","driver_circuit_wins"]].copy()
    if fillna_with is not None:
        out["driver_circuit_wins"] = out["driver_circuit_wins"].fillna(fillna_with)
    return out


#Feature 6
def Feature_6(
    results_csv="results.csv",
    races_csv="races.csv",
    fillna_with=0
):
    # Load
    results = pd.read_csv(results_csv, usecols=["raceId","driverId","constructorId","positionOrder"])
    races   = pd.read_csv(races_csv,   usecols=["raceId","circuitId","date"])
    races["date"] = pd.to_datetime(races["date"])

    # Per-race winning constructor
    winners = (
        results.loc[results["positionOrder"] == 1, ["raceId","constructorId"]]
               .drop_duplicates()
               .assign(is_win=1)
    )

    # Driver–race–constructor rows + circuit + chronology
    dr_race = (
        results[["raceId","driverId","constructorId"]].drop_duplicates()
               .merge(races, on="raceId", how="left")
               .sort_values(["constructorId","circuitId","date","raceId"])
               .reset_index(drop=True)
    )

    # Attach win flag
    dr_race = dr_race.merge(winners, on=["raceId","constructorId"], how="left")
    dr_race["is_win"] = dr_race["is_win"].fillna(0).astype(int)

    # Cumulative wins by constructor on this circuit before this race
    dr_race["constructor_circuit_wins"] = (
        dr_race.groupby(["constructorId","circuitId"], group_keys=False)["is_win"]
               .apply(lambda s: s.shift().cumsum())
    )

    out = dr_race[["driverId","raceId","circuitId","constructorId","constructor_circuit_wins"]].copy()
    if fillna_with is not None:
        out["constructor_circuit_wins"] = out["constructor_circuit_wins"].fillna(fillna_with)
    return out


#Feature 7
def Feature_7(qual_csv="qualifying.csv", races_csv="races.csv", results_csv="results.csv",
    n=5,            # lookback window (races)
    min_prev=1,     # min prior races with a qual position
    fillna_with=None
):
    # Load
    qual   = pd.read_csv(qual_csv,  usecols=["raceId","driverId","position"])
    races  = pd.read_csv(races_csv, usecols=["raceId","circuitId","date"])
    races["date"] = pd.to_datetime(races["date"], errors="coerce")
    cons   = (pd.read_csv(results_csv, usecols=["raceId","driverId","constructorId"])
                .drop_duplicates(["raceId","driverId"]))

    # One row per driver-race with best qualifying position
    dr_race_qual = (
        qual.groupby(["raceId","driverId"], as_index=False)["position"]
            .min()
            .rename(columns={"position":"qual_pos"})
    )

    # Merge chronology and constructor
    df = (dr_race_qual
          .merge(races, on="raceId", how="left")
          .merge(cons,  on=["raceId","driverId"], how="left")
          .sort_values(["driverId","date","raceId"])
          .reset_index(drop=True))

    # Rolling mean of previous n qualifying positions (no leakage)
    df["driver_standing_avg_quali_n"] = (
        df.groupby("driverId")["qual_pos"]
          .transform(lambda s: s.shift().rolling(window=n, min_periods=min_prev).mean())
    )

    # Output
    out = df[["driverId","raceId","circuitId","constructorId","driver_standing_avg_quali_n"]].copy()
    if fillna_with is not None:
        out["driver_standing_avg_quali_n"] = out["driver_standing_avg_quali_n"].fillna(fillna_with)
    return out


#Feature 8
def Feature_8(results_csv="results.csv", status_csv="status.csv", races_csv="races.csv", finished_pattern="Finished",
    n=None,           # None = cumulative prior rate; int = rolling window over last n races
    min_prev=1,       # min prior races required for a value
    fillna_with=None  # e.g., 0
):
    # Load
    res    = pd.read_csv(results_csv, usecols=["raceId","driverId","constructorId","statusId"])
    status = pd.read_csv(status_csv,  usecols=["statusId","status"])
    races  = pd.read_csv(races_csv,   usecols=["raceId","circuitId","date"])
    races["date"] = pd.to_datetime(races["date"], errors="coerce")

    # Flag DNF
    df = (res.merge(status, on="statusId", how="left")
             .merge(races,  on="raceId",   how="left")
             .sort_values(["driverId","date","raceId"])
             .reset_index(drop=True))
    dnf = ~df["status"].fillna("").str.contains(finished_pattern, case=False, na=False)
    df["DNF"] = dnf.astype(int)

    # Prior DNF rate (exclude current race via shift)
    if n is None:
        # cumulative prior rate
        df["prior_races"] = df.groupby("driverId").cumcount()
        df["prior_dnfs"]  = df.groupby("driverId")["DNF"].cumsum().shift(fill_value=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = df["prior_dnfs"] / df["prior_races"].replace(0, np.nan)
    else:
        # rolling over last n prior races
        rate = (df.groupby("driverId")["DNF"]
                  .transform(lambda s: s.shift().rolling(n, min_periods=min_prev).mean()))
    df["driver_standing_dnf_rate_n"] = rate

    # Output schema
    out = df[["driverId","raceId","circuitId","constructorId","driver_standing_dnf_rate_n"]].copy()
    if fillna_with is not None:
        out["driver_standing_dnf_rate_n"] = out["driver_standing_dnf_rate_n"].fillna(fillna_with)
    return out


#feature 11
def Feature_11(
    results_csv="results.csv",
    races_csv="races.csv",
):
    # Load
    res   = pd.read_csv(results_csv, usecols=["raceId","driverId","constructorId"]).drop_duplicates(["raceId","driverId"])
    races = pd.read_csv(races_csv,   usecols=["raceId","circuitId","date","year","round"])

    # Prefer date for ordering if available
    if "date" in races.columns:
        races["date"] = pd.to_datetime(races["date"], errors="coerce")
        order_cols = ["driverId","date","raceId"]
    else:
        order_cols = ["driverId","year","round","raceId"]

    # Merge and sort
    df = (res.merge(races, on="raceId", how="left")
            .sort_values(order_cols)
            .reset_index(drop=True))

    # Prior times this driver has raced on this circuit (no leakage)
    df["driver_circuit_race_count"] = (
        df.groupby(["driverId","circuitId"]).cumcount()
    )

    return df[["driverId","circuitId","raceId","constructorId","driver_circuit_race_count"]]


#Feature 12
def Feature_12( results_csv="results.csv", races_csv="races.csv"):
    # load minimal columns
    res   = pd.read_csv(results_csv, usecols=["raceId","driverId","constructorId","laps"])\
              .drop_duplicates(["raceId","driverId"])
    races = pd.read_csv(races_csv,   usecols=["raceId","circuitId"])

    # total laps per race
    total_laps = res.groupby("raceId", as_index=False)["laps"].max().rename(columns={"laps":"total_laps"})
    df = (res.merge(total_laps, on="raceId", how="left")
            .merge(races,      on="raceId", how="left"))

    # reliability ratio per driver-race
    df["driver_circuit_lap_count"] = df["laps"] / df["total_laps"]

    return df[["driverId","raceId","constructorId","circuitId","driver_circuit_lap_count"]]

def Feature_13(
    pitStop_csv="pit_stops.csv",
    races_csv="races.csv",
    results_csv="results.csv"
):
    pit_stops = pd.read_csv(pitStop_csv, usecols=["raceId","driverId"])
    races     = pd.read_csv(races_csv,   usecols=["raceId","year","round","circuitId"])
    results   = pd.read_csv(results_csv, usecols=["raceId","driverId","constructorId"])

    # count pit stops per driver-race
    pit_counts = (pit_stops.groupby(["raceId","driverId"])
                           .size()
                           .reset_index(name="pitstop_count"))

    # one row per driver-race with constructorId, include zero-stop races
    driver_race = (
        results.drop_duplicates(["raceId","driverId"])
               .merge(pit_counts, on=["raceId","driverId"], how="left")
               .fillna({"pitstop_count": 0})
               .merge(races, on="raceId", how="left")
               .sort_values(["driverId","year","round","raceId"])
               .reset_index(drop=True)
    )

    # average pit stops per prior races (no leakage)
    g = driver_race.groupby("driverId", group_keys=False)["pitstop_count"]
    cum_sum = g.cumsum()
    prior_cnt = driver_race.groupby("driverId").cumcount()  # 0,1,2,...
    driver_race["driver_circuit_avg_pit_stops"] = (cum_sum - driver_race["pitstop_count"]) / prior_cnt.replace(0, pd.NA)

    return driver_race[["driverId","raceId","constructorId","circuitId","driver_circuit_avg_pit_stops"]]
    


#Feature 15
def Feature_15(results_csv="results.csv", races_csv="races.csv", n=10, min_prev=1):
    # Load
    res   = pd.read_csv(results_csv, usecols=["raceId","driverId","constructorId","grid","positionOrder"])\
              .drop_duplicates(["raceId","driverId"])
    races = pd.read_csv(races_csv,   usecols=["raceId","circuitId","date","year","round"])

    # Order per driver (prefer date if present)
    if "date" in races.columns:
        races["date"] = pd.to_datetime(races["date"], errors="coerce")
        order_cols = ["driverId","date","raceId"]
    else:
        order_cols = ["driverId","year","round","raceId"]

    df = (res.merge(races, on="raceId", how="left")
            .sort_values(order_cols)
            .reset_index(drop=True))

    # Per-race position change
    df["pos_diff"] = df["grid"] - df["positionOrder"]
    df.loc[df["grid"] == 0, "pos_diff"] = pd.NA  # ignore pit-lane/unknown starts

    # Rolling mean of previous n races (no leakage)
    df["avg_start_to_finish_diff"] = (
        df.groupby("driverId")["pos_diff"]
          .transform(lambda s: s.shift().rolling(window=n, min_periods=min_prev).mean())
    )

    return df[["driverId","raceId","constructorId","circuitId","avg_start_to_finish_diff"]]


#Feature 16
def Feature_16(results_csv="results.csv", races_csv="races.csv"):
    results = pd.read_csv(results_csv, usecols=["raceId","driverId","constructorId","points"])
    races   = pd.read_csv(races_csv,   usecols=["raceId","year","round","circuitId"])

    df = results.merge(races, on="raceId", how="left")

    team_pts_race = (
        df.groupby(["year","round","raceId","constructorId"], as_index=False)["points"]
          .sum()
          .rename(columns={"points":"team_points_race"})
          .sort_values(["year","round","raceId"])
          .reset_index(drop=True)
    )

    team_pts_race["cum_team_points_prev"] = (
        team_pts_race.groupby(["year","constructorId"])["team_points_race"]
                     .transform(lambda s: s.cumsum().shift())
                     .fillna(0)
    )

    team_pts_race["constructor_standing_position"] = (
        team_pts_race.groupby(["year","round"])["cum_team_points_prev"]
                     .rank(method="min", ascending=False)
                     .astype(int)
    )

    out = df.merge(
        team_pts_race[["raceId","constructorId","constructor_standing_position"]],
        on=["raceId","constructorId"], how="left"
    )[["driverId","raceId","constructorId","circuitId","constructor_standing_position"]]

    return out

def Feature_17(results_csv="results.csv", races_csv="races.csv"):
    # base rows: one per driver–race
    res   = pd.read_csv(results_csv, usecols=["raceId","driverId","constructorId","points"])\
              .drop_duplicates(["raceId","driverId"])
    races = pd.read_csv(races_csv,   usecols=["raceId","circuitId","date","year","round"])

    # per driver–race points with season keys
    dr_pts = (res.groupby(["raceId","driverId"], as_index=False)["points"].sum()
                 .rename(columns={"points":"driver_points_race"})
                 .merge(races, on="raceId", how="left"))

    # order within season (prefer date if present)
    if "date" in dr_pts.columns:
        dr_pts["date"] = pd.to_datetime(dr_pts["date"], errors="coerce")
        dr_pts = dr_pts.sort_values(["year","date","raceId"])
    else:
        dr_pts = dr_pts.sort_values(["year","round","raceId"])

    # cumulative season points BEFORE this race
    dr_pts["cum_driver_points_prev"] = (
        dr_pts.groupby(["year","driverId"])["driver_points_race"]
              .transform(lambda s: s.cumsum().shift().fillna(0))
    )

    # season standings BEFORE this race
    dr_pts["driver_standing_position"] = (
        dr_pts.groupby(["year","round"])["cum_driver_points_prev"]
              .rank(method="min", ascending=False)
              .astype(int)
    )

    # attach to base rows and return required columns
    out = (res.merge(races[["raceId","circuitId"]], on="raceId", how="left")
              .merge(dr_pts[["raceId","driverId","driver_standing_position"]],
                     on=["raceId","driverId"], how="left")
              [["driverId","raceId","constructorId","circuitId","driver_standing_position"]])
    return out



##################################################
def Feature_18(
    driver_standings_csv="driver_standings.csv",
    drivers_csv="drivers.csv",          # not required for the feature; kept for parity
    races_csv="races.csv",
    results_csv="results.csv",
    fillna_with=0
):
    # base keys with constructorId and circuitId
    base = (pd.read_csv(results_csv, usecols=["raceId","driverId","constructorId"])
              .drop_duplicates(["raceId","driverId"])
              .merge(pd.read_csv(races_csv, usecols=["raceId","circuitId","date","year","round"]),
                     on="raceId", how="left"))

    # standings used to compute podium history
    ds = pd.read_csv(driver_standings_csv, usecols=["raceId","driverId","position"])
    # coerce position to numeric; non-numeric -> NaN
    ds["position"] = pd.to_numeric(ds["position"], errors="coerce")

    # add chronology
    races = pd.read_csv(races_csv, usecols=["raceId","date","year","round"])
    if "date" in races.columns:
        races["date"] = pd.to_datetime(races["date"], errors="coerce")
        order_cols = ["driverId","date","raceId"]
    else:
        order_cols = ["driverId","year","round","raceId"]

    ds = ds.merge(races, on="raceId", how="left").sort_values(order_cols).reset_index(drop=True)

    # flag podiums and compute cumulative BEFORE current race
    ds["is_podium"] = ds["position"].isin([1,2,3]).astype(int)
    ds["constructor_standing_points"] = (
        ds.groupby("driverId")["is_podium"]
          .transform(lambda s: s.cumsum().shift().fillna(0))
    )

    # attach to base rows; keep all driver–race rows
    out = (base.merge(ds[["raceId","driverId","constructor_standing_points"]], on=["raceId","driverId"], how="left")
               .rename(columns={"constructor_standing_points":"constructor_standing_points"})  # explicit
               [["driverId","raceId","constructorId","circuitId","constructor_standing_points"]])

    if fillna_with is not None:
        out["constructor_standing_points"] = out["constructor_standing_points"].fillna(fillna_with)

    return out



def Feature_19(
    constructor_standings_csv="constructor_standings.csv",
    constructors_csv="constructors.csv",   # not required for the feature
    races_csv="races.csv",
    results_csv="results.csv",
    fillna_with=0
):
    # base: one row per driver–race with constructorId and circuitId
    base = (pd.read_csv(results_csv, usecols=["raceId","driverId","constructorId"])
              .drop_duplicates(["raceId","driverId"])
              .merge(pd.read_csv(races_csv, usecols=["raceId","circuitId","date","year","round"]),
                     on="raceId", how="left"))

    # standings for podium history
    cs = pd.read_csv(constructor_standings_csv, usecols=["raceId","constructorId","position"])
    cs["position"] = pd.to_numeric(cs["position"], errors="coerce")

    # chronology
    races = pd.read_csv(races_csv, usecols=["raceId","date","year","round"])
    if "date" in races.columns:
        races["date"] = pd.to_datetime(races["date"], errors="coerce")
        order = ["constructorId","date","raceId"]
    else:
        order = ["constructorId","year","round","raceId"]

    cs = cs.merge(races, on="raceId", how="left").sort_values(order).reset_index(drop=True)

    # cumulative podiums BEFORE each race for each constructor
    cs["is_podium"] = cs["position"].isin([1,2,3]).astype(int)
    cs["driver_standing_points"] = (
        cs.groupby("constructorId")["is_podium"].transform(lambda s: s.cumsum().shift().fillna(0))
    )

    # attach to driver–race rows
    out = (base.merge(cs[["raceId","constructorId","driver_standing_points"]],
                      on=["raceId","constructorId"], how="left")
              [["driverId","raceId","constructorId","circuitId","driver_standing_points"]])

    if fillna_with is not None:
        out["driver_standing_points"] = out["driver_standing_points"].fillna(fillna_with)

    return out

def Feature_20(
    driver_standings_csv="driver_standings.csv",
    races_csv="races.csv",
    results_csv="results.csv",
    fillna_with=0  # set to None to keep NaN on a driver's first race of the season
):
    # base: one row per driver–race with constructorId and circuitId
    base = (pd.read_csv(results_csv, usecols=["raceId","driverId","constructorId"])
              .drop_duplicates(["raceId","driverId"])
              .merge(pd.read_csv(races_csv, usecols=["raceId","circuitId","date","year","round"]),
                     on="raceId", how="left"))

    # standings with chronology
    ds = pd.read_csv(driver_standings_csv, usecols=["raceId","driverId","position","points"])
    races = pd.read_csv(races_csv, usecols=["raceId","date","year","round"])
    if "date" in races.columns:
        races["date"] = pd.to_datetime(races["date"], errors="coerce")
        order = ["year","date","raceId"]
    else:
        order = ["year","round","raceId"]

    ds = (ds.merge(races, on="raceId", how="left")
            .sort_values(order)
            .reset_index(drop=True))

    # numeric position; compute rank BEFORE race (shift by 1 within season)
    ds["position"] = pd.to_numeric(ds["position"], errors="coerce")
    ds["constructor_standing_podiums"] = (
        ds.groupby(["year","driverId"])["position"].shift(1)
    )

    # attach to base and return required columns
    out = (base.merge(ds[["raceId","driverId","constructor_standing_podiums"]],
                      on=["raceId","driverId"], how="left")
              [["driverId","raceId","constructorId","circuitId","constructor_standing_podiums"]])

    if fillna_with is not None:
        out["constructor_standing_podiums"] = out["constructor_standing_podiums"].fillna(fillna_with)

    return out


def Feature_21(
    constructor_standings_csv="constructor_standings.csv",
    races_csv="races.csv",
    results_csv="results.csv",
    fillna_with=0  # set None to keep NaN on a constructor's first race of the season
):
    # Base: one row per driver–race with constructorId and circuitId
    base = (pd.read_csv(results_csv, usecols=["raceId","driverId","constructorId"])
              .drop_duplicates(["raceId","driverId"])
              .merge(pd.read_csv(races_csv, usecols=["raceId","circuitId","date","year","round"]),
                     on="raceId", how="left"))

    # Constructor standings with chronology
    cs = pd.read_csv(constructor_standings_csv, usecols=["raceId","constructorId","position","points"])
    races = pd.read_csv(races_csv, usecols=["raceId","date","year","round"])
    if "date" in races.columns:
        races["date"] = pd.to_datetime(races["date"], errors="coerce")
        order = ["year","date","raceId"]
    else:
        order = ["year","round","raceId"]

    cs = (cs.merge(races, on="raceId", how="left")
            .sort_values(order)
            .reset_index(drop=True))

    # Rank BEFORE race (shift by 1 within season)
    cs["position"] = pd.to_numeric(cs["position"], errors="coerce")
    cs["driver_standing_podiums"] = (
        cs.groupby(["year","constructorId"])["position"].shift(1)
    )

    # Attach to driver–race base
    out = (base.merge(cs[["raceId","constructorId","driver_standing_podiums"]],
                      on=["raceId","constructorId"], how="left")
              [["driverId","raceId","constructorId","circuitId","driver_standing_podiums"]])

    if fillna_with is not None:
        out["driver_standing_podiums"] = out["driver_standing_podiums"].fillna(fillna_with)

    return out

def Feature_22(
    lap_times_csv="lap_times.csv",
    races_csv="races.csv",
    results_csv="results.csv",
    n_seasons=3,
    min_seasons=1,
    fillna_with=0
):
    # Base: one row per driver–race with constructor and circuit
    base = (pd.read_csv(results_csv, usecols=["raceId","driverId","constructorId"])
              .drop_duplicates(["raceId","driverId"])
              .merge(pd.read_csv(races_csv, usecols=["raceId","circuitId","year","date"]),
                     on="raceId", how="left"))

    # Lap leader per lap with season + circuit
    lt = pd.read_csv(lap_times_csv, usecols=["raceId","driverId","position"])
    races = pd.read_csv(races_csv, usecols=["raceId","year","circuitId"])
    lt = lt.merge(races, on="raceId", how="left")
    lt["position"] = pd.to_numeric(lt["position"], errors="coerce")
    lt["led"] = (lt["position"] == 1).astype(int)

    # Yearly laps led per driver on each circuit
    yearly = (lt.groupby(["driverId","circuitId","year"], as_index=False)["led"]
                .sum()
                .rename(columns={"led":"laps_led_year"}))
    yearly = yearly.sort_values(["driverId","circuitId","year"]).reset_index(drop=True)

    # Rolling sum over previous n seasons on this circuit (no leakage via shift)
    yearly["driver_circuit_laps_led_n"] = (
        yearly.groupby(["driverId","circuitId"])["laps_led_year"]
              .transform(lambda s: s.shift().rolling(n_seasons, min_periods=min_seasons).sum())
    )

    # Attach to each driver–race row using current race's (driver, circuit, year)
    out = (base.merge(
            yearly[["driverId","circuitId","year","driver_circuit_laps_led_n"]],
            on=["driverId","circuitId","year"], how="left")
          [["driverId","raceId","constructorId","circuitId","driver_circuit_laps_led_n"]])

    if fillna_with is not None:
        out["driver_circuit_laps_led_n"] = out["driver_circuit_laps_led_n"].fillna(fillna_with)

    return out


# Feature 23
def Feature_23(results_csv="results.csv", races_csv="races.csv", n=3,
               min_seasons=1,
               fillna_with=0):
    res   = pd.read_csv(results_csv, usecols=["raceId","driverId","constructorId","points"])
    races = pd.read_csv(races_csv,   usecols=["raceId","year","circuitId"])

    df = res.merge(races, on="raceId", how="left")

    dr_race_pts = (df.groupby(["raceId","driverId","year","circuitId"], as_index=False)["points"]
                     .sum()
                     .rename(columns={"points":"race_points"}))

    season_pts = (dr_race_pts.groupby(["driverId","circuitId","year"], as_index=False)["race_points"]
                    .sum()
                    .rename(columns={"race_points":"season_points"})
                    .sort_values(["driverId","circuitId","year"])
                    .reset_index(drop=True))

    # rolling sum over previous n seasons on that circuit (no leakage)
    season_pts["driver_standing_wins"] = (
        season_pts.groupby(["driverId","circuitId"])["season_points"]
                  .transform(lambda s: s.shift().rolling(n, min_periods=min_seasons).sum())
    )

    cons = res.drop_duplicates(["raceId","driverId"])[["raceId","driverId","constructorId"]]

    out = (dr_race_pts.merge(season_pts[["driverId","circuitId","year","driver_standing_wins"]],
                             on=["driverId","circuitId","year"], how="left")
                     .merge(cons, on=["raceId","driverId"], how="left")
                     [["driverId","raceId","constructorId","circuitId","driver_standing_wins"]])

    if fillna_with is not None:
        out["driver_standing_wins"] = out["driver_standing_wins"].fillna(fillna_with)

    return out

def feature_constructor_standing_wins(
    results_csv="results.csv",
    races_csv="races.csv",
    n=3,
    min_seasons=1,
    fillna_with=0
):
    # base rows: one per driver–race
    res   = pd.read_csv(results_csv, usecols=["raceId","driverId","constructorId","positionOrder"])\
              .drop_duplicates(["raceId","driverId"])
    races = pd.read_csv(races_csv,   usecols=["raceId","year","circuitId"])

    df = res.merge(races, on="raceId", how="left")

    # per-season wins on a circuit by constructor
    winners = df.loc[df["positionOrder"] == 1, ["raceId","constructorId","year","circuitId"]]
    cons_circuit_year_wins = (
        winners.groupby(["constructorId","circuitId","year"], as_index=False)
               .size()
               .rename(columns={"size":"wins_in_season_on_circuit"})
               .sort_values(["constructorId","circuitId","year"])
               .reset_index(drop=True)
    )

    # rolling sum over previous n seasons (no leakage)
    cons_circuit_year_wins["constructor_standing_wins"] = (
        cons_circuit_year_wins.groupby(["constructorId","circuitId"])["wins_in_season_on_circuit"]
            .transform(lambda s: s.shift().rolling(window=n, min_periods=min_seasons).sum())
    )

    # attach to driver–race rows
    dr_race = df[["driverId","raceId","constructorId","circuitId","year"]].drop_duplicates()
    out = dr_race.merge(
        cons_circuit_year_wins[["constructorId","circuitId","year","constructor_standing_wins"]],
        on=["constructorId","circuitId","year"], how="left"
    )[["driverId","raceId","constructorId","circuitId","constructor_standing_wins"]]

    if fillna_with is not None:
        out["constructor_standing_wins"] = out["constructor_standing_wins"].fillna(fillna_with)

    return out

# ---- base table
def make_base_extended(
    results_csv="results.csv",
    races_csv="races.csv",
    circuits_csv="circuits.csv",
    drivers_csv="drivers.csv",
    constructors_csv="constructors.csv",
):
    # results: one row per driver–race (keep first resultId per pair)
    res = (pd.read_csv(results_csv, usecols=["resultId","raceId","driverId","constructorId","number","positionOrder"])
             .sort_values(["raceId","driverId","resultId"])
             .drop_duplicates(["raceId","driverId"], keep="first"))

    races = pd.read_csv(races_csv, usecols=["raceId","year","round","circuitId","date"])
    circuits = pd.read_csv(circuits_csv, usecols=["circuitId","name","country"])
    drivers = pd.read_csv(drivers_csv, usecols=["driverId","forename","surname"])
    constructors = pd.read_csv(constructors_csv, usecols=["constructorId","name"])

    base = (res
        .merge(races, on="raceId", how="left")
        .merge(circuits.rename(columns={"name":"circuit_name","country":"circuit_country"}), on="circuitId", how="left")
        .merge(drivers.rename(columns={"forename":"driver_forename","surname":"driver_surname"}), on="driverId", how="left")
        .merge(constructors.rename(columns={"name":"constructor_name"}), on="constructorId", how="left")
    )

    base = base.rename(columns={"number":"car_number", "positionOrder":"final_position"})
    # keep required leading columns + keys for merges
    cols = ["resultId","raceId","year","round","circuitId","circuit_name","circuit_country",
            "driverId","driver_forename","driver_surname","constructorId","constructor_name","car_number",
            "date"]
    return base[cols + ["final_position"]]

# merge helper (unchanged)
def merge_feature(base, feat, feature_cols):
    drop_cols = [c for c in ["constructorId","circuitId","year","round","date"] if c in feat.columns]
    feat = feat.drop(columns=drop_cols, errors="ignore")
    feat = feat[["driverId","raceId"] + feature_cols].drop_duplicates(["driverId","raceId"])
    return base.merge(feat, on=["driverId","raceId"], how="left")

# finalize order (unchanged)
def finalize_order(ds):
    if "date" in ds.columns and ds["date"].notna().any():
        ds = ds.sort_values(["year","date","raceId","driverId"])
    else:
        ds = ds.sort_values(["year","round","raceId","driverId"])
    return ds.reset_index(drop=True)

# ---- build with new leading columns and final_position at end ----
def build_dataset_and_save(output_path="F1_final_dataset.csv"):
    base = make_base_extended()  # includes names and car_number
    # expose merge keys
    base[["driverId","raceId","constructorId","circuitId"]] = base[["driverId","raceId","constructorId","circuitId"]]

    ds = base.copy()

    # add features (call your existing functions)
    ds = merge_feature(ds, Feature_1(),  ["grid_start_position"])
    ds = merge_feature(ds, Feature_2(),  ["driver_standing_laps_led_n"])
    ds = merge_feature(ds, Feature_3(),  ["driver_standing_points_n"])
    ds = merge_feature(ds, Feature_4(),  ["driver_circuit_best_lap"])
    ds = merge_feature(ds, Feature_5(),  ["driver_circuit_wins"])
    ds = merge_feature(ds, Feature_6(),  ["constructor_circuit_wins"])
    ds = merge_feature(ds, Feature_7(),  ["driver_standing_avg_quali_n"])
    ds = merge_feature(ds, Feature_8(fillna_with=0), ["driver_standing_dnf_rate_n"])
    ds = merge_feature(ds, Feature_11(), ["driver_circuit_race_count"])
    ds = merge_feature(ds, Feature_12(), ["driver_circuit_lap_count"])
    ds = merge_feature(ds, Feature_13(), ["driver_circuit_avg_pit_stops"])
    ds = merge_feature(ds, Feature_15(), ["avg_start_to_finish_diff"])
    ds = merge_feature(ds, Feature_16(), ["constructor_standing_position"])
    ds = merge_feature(ds, Feature_17(), ["driver_standing_position"])
    ds = merge_feature(ds, Feature_18(), ["constructor_standing_points"])
    ds = merge_feature(ds, Feature_19(), ["driver_standing_points"])
    ds = merge_feature(ds, Feature_20(), ["constructor_standing_podiums"])
    ds = merge_feature(ds, Feature_21(), ["driver_standing_podiums"])
    ds = merge_feature(ds, Feature_22(), ["driver_circuit_laps_led_n"])
    ds = merge_feature(ds, Feature_23(fillna_with=0), ["driver_standing_wins"])
    ds = merge_feature(ds, feature_constructor_standing_wins(fillna_with=0),
                       ["constructor_standing_wins"])

    # reorder: leading cols, then all features, then final_position
    leading = ["resultId","raceId","year","round","circuitId","circuit_name","circuit_country",
               "driverId","driver_forename","driver_surname","constructorId","constructor_name","car_number"]
    # move final_position to end; it lives in base
    feat_cols = [c for c in ds.columns if c not in (leading + ["date","final_position"])]
    ds = ds[leading + feat_cols + ["final_position"]]

    # sort and save
    ds = finalize_order(ds)
    ds.to_csv(output_path, index=False)
    print(f"Saved {ds.shape} to {output_path}")

build_dataset_and_save("F1_final_dataset.csv")    
