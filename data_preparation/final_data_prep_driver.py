import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

def get_weather(file_path='weather_features_v4.csv'):
    return kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "mariyakostyrya/formula-1-weather-info-1950-2024",
        file_path
    )

def get_circuits(file_path='circuits.csv'):
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "jtrotman/formula-1-race-data",
        file_path
    )

def get_drivers(file_path='drivers.csv'):
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "jtrotman/formula-1-race-data",
        file_path
    )

def get_races(file_path='races.csv'):
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "jtrotman/formula-1-race-data",
        file_path
    )

def get_results(file_path='results.csv'):
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "jtrotman/formula-1-race-data",
        file_path
    )

def get_pit_stops(file_path='pit_stops.csv'):
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "jtrotman/formula-1-race-data",
        file_path
    )

def get_qualifying(file_path='qualifying.csv'):
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "jtrotman/formula-1-race-data",
        file_path
    )

def get_lap_times(file_path='lap_times.csv'):
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "jtrotman/formula-1-race-data",
        file_path
    )

def get_constructors(file_path='constructors.csv'):
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "jtrotman/formula-1-race-data",
        file_path
    )

def get_constructor_results(file_path='constructor_results.csv'):
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "jtrotman/formula-1-race-data",
        file_path
    )

def get_constructor_standings(file_path='constructor_standings.csv'):
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "jtrotman/formula-1-race-data",
        file_path
    )

def get_driver_standings(file_path='driver_standings.csv'):
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "jtrotman/formula-1-race-data",
        file_path
    )

def get_seasons(file_path='seasons.csv'):
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "jtrotman/formula-1-race-data",
        file_path
    )

def get_status(file_path='status.csv'):
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "jtrotman/formula-1-race-data",
        file_path
    )

def filter_year(df, year):
  races = get_races()
  list_raceid = races["raceId"][races["year"]==year]
  return df[df["raceId"].isin(list_raceid)]

def filter_before_year(df, year):
  races = get_races()
  list_raceid = races["raceId"][races["year"]<year]
  return df[df["raceId"].isin(list_raceid)]

def _finish_is_numeric(df):
    src = df["finish_position"]
    def _isnum(x):
        try:
            float(x)
            return True
        except:
            return False
    return src.astype(str).map(_isnum).astype(int)  


def _rolling_last_n(series, n, agg="sum"):
    if series.empty:
        return np.nan
    s = series.astype(float)
    if agg == "sum":
        return s.rolling(n, min_periods=1).sum()
    if agg == "mean":
        return s.rolling(n, min_periods=1).mean()
    if agg == "min":
        return s.rolling(n, min_periods=1).min()
    return s


def prepare_driver_standings(year):
  ds_full = get_driver_standings()
  ds = filter_year(ds_full, year).sort_values("raceId")

  cols_ds = ["raceId", "driverId", "points", "position", "wins"]
  ds = ds[cols_ds]

  ds = ds.rename(columns={'points': 'driver_standing_points',
                          'position': 'driver_standing_position',
                          'wins': 'driver_standing_wins'
                          })
  ds['raceId'] += 1

  max_raceId = max(ds['raceId'])
  ds = ds[ds['raceId'] != max_raceId]

  return ds


def prepare_results_season(year, n_window):
  r_full = get_results()

  r = filter_year(r_full, year)
  cols = ['raceId', 'driverId', 'constructorId', 'grid', 'position', 'points']
  r = r[cols]
  r = r.rename(columns={'grid': 'grid_start_position',
                        'position': 'finish_position'
                        })

  # cols_side =
  r_side = r[['raceId', 'driverId', 'constructorId', 'finish_position', 'points']].copy()

  # driver_standing_podiums
  r_side["is_podium"] = r_side["finish_position"].isin(['1','2','3'])
  r_side["driver_standing_podiums"] = r_side.groupby("driverId")["is_podium"].cumsum()

  # driver_standing_dnf_rate_n
  r_side["finish_numeric"] = _finish_is_numeric(r_side)
  r_side["driver_standing_dnf_rate_n"] = (1 - (r_side
          .groupby("driverId", group_keys=False)["finish_numeric"]
          .apply(lambda s: _rolling_last_n(s, n_window, "mean"))))

  # driver_standing_points_n
  r_side["driver_standing_points_n"] = (r_side
          .groupby("driverId", group_keys=False)["points"]
          .apply(lambda s: _rolling_last_n(s, n_window, "sum")))
  

  # driver_standing_laps_led
  races = get_races()[['raceId', 'round', 'circuitId']]
  race = filter_year(races, year)

  lap_times  = get_lap_times()
  lap_t = filter_year(lap_times, year)
  lap_t


  laps_y = lap_t.merge(race[["raceId","round"]], on="raceId", how="inner")
  laps_led_per_race = (laps_y[laps_y["position"] == 1]
                      .groupby(["raceId","driverId"])["lap"].count()
                      .rename("laps_led_in_race").reset_index())
  laps_led_per_race
  r_side = r_side.merge(laps_led_per_race, on=["raceId","driverId"], how="left")
  r_side["laps_led_in_race"] = r_side["laps_led_in_race"].fillna(0)
  r_side["driver_standing_laps_led"] = r_side.groupby("driverId")["laps_led_in_race"].cumsum()



  r_side['raceId'] +=1


  r_side = r_side.drop(['finish_position', 'points', 'is_podium', 'finish_numeric', "laps_led_in_race", "constructorId"], axis=1)

  max_raceId = max(r_side['raceId'])
  r_side = r_side[r_side['raceId'] != max_raceId]


  return r_side

def prepare_results_circuits(year, n_window):
  r_full = get_results()

  r_full = r_full[['raceId','driverId', 'position']]

  r = filter_before_year(r_full, year)

  # #driver_circuit_wins
  races = get_races()[['raceId', 'round', 'circuitId']]
  race = filter_before_year(races, year)
  r = r.merge(race, on = 'raceId', how = 'left')
  r["is_win"] = (r["position"] == '1')
  r1 = r.groupby(["driverId","circuitId"])["is_win"].sum().reset_index()

  # driver_circuit_race_count
  r2 = r.groupby(["driverId", "circuitId"]).size().reset_index(name="race_count")

  r = r1.merge(r2, how = 'left', on = ["driverId", "circuitId"])


  # driver_circuit_lap_count
  lap_times  = get_lap_times()
  lap_t = filter_before_year(lap_times, year)
  lap_t = lap_t.merge(race, how = 'left', on = 'raceId')
  lap_t = lap_t.groupby(["driverId","circuitId"])["lap"].count().reset_index()

  r = r.merge(lap_t, how = 'left', on = ["driverId", "circuitId"])

  
  #driver_circuit_laps_led
  lap_t2 = filter_before_year(lap_times, year)
  lap_t2 = lap_t2[lap_t2['position'] == 1]
  lap_t2
  lap_t2 = lap_t2.merge(race, how = 'left', on = 'raceId')
  lap_t2 = lap_t2.groupby(["driverId","circuitId"])["lap"].count().reset_index()
  lap_t2 = lap_t2.rename(columns={'lap': 'lap2'})



  final_r = filter_year(r_full, year)
  race_year = filter_year(races[['raceId', 'circuitId']], year)
  final_r = final_r.merge(race_year, how = 'left', on = 'raceId')

  final_r = final_r.merge(r, how = 'left', on = ["driverId","circuitId"])

  final_r = final_r.rename(columns = {'is_win': 'driver_circuit_wins',
                                      'race_count': 'driver_circuit_race_count',
                                      'lap' : 'driver_circuit_lap_count',
                                      'lap2' : 'driver_circuit_laps_led'})

  final_r["driver_circuit_wins"] = final_r["driver_circuit_wins"].fillna(0)
  final_r["driver_circuit_race_count"] = final_r["driver_circuit_race_count"].fillna(0)
  final_r["driver_circuit_lap_count"] = final_r["driver_circuit_lap_count"].fillna(0)

  final_r = final_r.drop(['position'], axis = 1)
  
  return final_r


def prepare_final_result(year):
  w = get_weather()  
  w["datetime"] = pd.to_datetime(w["datetime"])
  w["year"] = w["datetime"].dt.year
  w = w[['temperature', 'precipitation', 'windspeed', 'round', 'year']]

  races = get_races()[['raceId', 'round', 'year']]
  race = filter_year(races, year)

  w = w[w['year'] == year]
  final_w = race.merge(w, how = 'left', on = ['round', 'year'])
  
  final_w = final_w[['raceId', 'temperature', 'precipitation', 'windspeed']]

  result = get_results()[['resultId', 'raceId', 'driverId', 'constructorId', 'grid', 'position']]
  result = filter_year(result, year)

  result = result.rename(columns = {'grid': 'grid_start_position',
                                    'position' : 'finish_position'})
  
  result = result.merge(final_w, how = 'left', on = ['raceId'])

  return result


start = 2025
end = 2026
n_window = 5

final_data = pd.DataFrame()

for year in range(start, end):
  df = prepare_driver_standings(year)

  df_2 = prepare_results_season(year, n_window)
  
  df_3 = prepare_results_circuits(year, n_window)
    
  df_4 = prepare_final_result(year)
  
  df_4 = df_4.merge(df_3, how = 'left', on = ['raceId', 'driverId'])
  df_4 = df_4.merge(df_2, how = 'left', on = ['raceId', 'driverId'])
  df_4 = df_4.merge(df, how = 'left', on = ['raceId', 'driverId'])


  final_data = pd.concat([final_data, df_4], ignore_index=True)

final_data.to_csv('final_driver_data_20_25.csv', index=False)