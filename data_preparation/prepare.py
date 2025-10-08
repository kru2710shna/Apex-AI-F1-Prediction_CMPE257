from data_fetching.fetcher import get_circuits

def prepare_circuits():
    circuits_df = get_circuits()
    
    # Steps to clean and prepare the circuits data

    return circuits_df



def compute_cumulative_podiums(driver_standings, drivers, races):
    """
    Compute cumulative podiums (positions 1–3) per driver before each race.
    
    Args:
        driver_standings (pd.DataFrame): Data from driver_standings.csv
        drivers (pd.DataFrame): Data from drivers.csv
        races (pd.DataFrame): Data from races.csv

    Returns:
        pd.DataFrame: Cumulative podiums before each race
    """

    # Step 1: Merge races with driver_standings to get year and round
    df = driver_standings.merge(races[['raceId', 'year', 'round', 'name']], on='raceId', how='left')

    # Step 2: Sort by driver and chronological race order
    df = df.sort_values(by=['driverId', 'year', 'round']).reset_index(drop=True)

    # Step 3: Identify podiums (positions 1–3)
    df['is_podium'] = df['position'].isin([1, 2, 3]).astype(int)

    # Step 4: Compute cumulative podiums BEFORE each race
    df['Number of podiums (driver)'] = df.groupby('driverId')['is_podium'].cumsum().shift(fill_value=0)

    # Step 5: Add readable driver names
    df = df.merge(drivers[['driverId', 'driverRef', 'forename', 'surname']], on='driverId', how='left')

    # Step 6: Final column ordering
    df = df[['year', 'round', 'raceId', 'name', 'driverId', 'driverRef', 'forename', 'surname', 'Number of podiums (driver)']]

    return df



def compute_cumulative_constructor_podiums(constructor_standings, constructors, races):
    """
    Compute cumulative podiums (positions 1–3) per constructor before each race.

    Args:
        constructor_standings (pd.DataFrame): Data from constructor_standings.csv
        constructors (pd.DataFrame): Data from constructors.csv
        races (pd.DataFrame): Data from races.csv

    Returns:
        pd.DataFrame: cumulative podiums before each race
    """

    # Step 1: Merge races with constructor_standings to get year and round
    df = constructor_standings.merge(
        races[['raceId', 'year', 'round', 'name']],
        on='raceId', how='left'
    )

    # Step 2: Sort chronologically per constructor
    df = df.sort_values(by=['constructorId', 'year', 'round']).reset_index(drop=True)

    # Step 3: Mark podiums (positions 1–3)
    df['is_podium'] = df['position'].isin([1, 2, 3]).astype(int)

    # Step 4: Compute cumulative podiums BEFORE each race
    df['Number of podiums (constructor)'] = df.groupby('constructorId')['is_podium'].cumsum().shift(fill_value=0)

    # Step 5: Add readable constructor names
    df = df.merge(constructors[['constructorId', 'constructorRef', 'name', 'nationality']],
                  on='constructorId', how='left')

    # Step 6: Final column order
    df = df[['year', 'round', 'raceId', 'name_x', 'constructorId', 'constructorRef',
             'name_y', 'nationality', 'Number of podiums (constructor)']]
    df.rename(columns={'name_x': 'race_name', 'name_y': 'constructor_name'}, inplace=True)

    return df


import pandas as pd

def compute_driver_championship_rank_before_race(driver_standings, drivers, races):
    """
    Compute each driver's championship rank before each race.
    'Before race' = their position in the previous race's standings.
    """

    # Merge standings with races for year and round context
    df = driver_standings.merge(races[['raceId', 'year', 'round', 'name']], on='raceId', how='left')

    # Sort chronologically by driver and round
    df = df.sort_values(by=['driverId', 'year', 'round']).reset_index(drop=True)

    # Shift each driver's position by 1 race to get "before race" rank
    df['Rank_before_race'] = df.groupby(['year', 'driverId'])['position'].shift(1)

    # Merge driver names for clarity
    df = df.merge(drivers[['driverId', 'driverRef', 'forename', 'surname']],
                  on='driverId', how='left')

    # Clean and reorder columns
    df = df[['year', 'round', 'raceId', 'name', 'driverId', 'driverRef',
             'forename', 'surname', 'points', 'Rank_before_race']]

    return df



import pandas as pd

def compute_constructor_championship_rank_before_race(constructor_standings, constructors, races):
    """
    Compute each constructor's championship rank before each race.
    'Before race' = their position in the previous race's standings.
    """

    # Step 1: Merge with race info for year and round
    df = constructor_standings.merge(
        races[['raceId', 'year', 'round', 'name']],
        on='raceId', how='left'
    )

    # Step 2: Sort chronologically
    df = df.sort_values(by=['constructorId', 'year', 'round']).reset_index(drop=True)

    # Step 3: Get previous race’s rank
    df['Constructor_Rank_before_race'] = df.groupby(['year', 'constructorId'])['position'].shift(1)

    # Step 4: Merge with constructor info
    df = df.merge(
        constructors[['constructorId', 'name', 'nationality']],
        on='constructorId', how='left'
    )

    # Step 5: Select and clean up columns
    df = df[['year', 'round', 'raceId', 'name_x', 'constructorId', 'name_y', 'nationality',
             'points', 'Constructor_Rank_before_race']]

    # Rename for readability
    df = df.rename(columns={
        'name_x': 'race_name',
        'name_y': 'constructor_name'
    })

    return df


import pandas as pd

def lap_leads_by_track(lap_times, races, circuits, drivers, n_seasons=3):
    """
    Calculate total laps led by each driver on each track in the last n seasons.

    Args:
        lap_times (DataFrame): lap_times.csv
        races (DataFrame): races.csv
        circuits (DataFrame): circuits.csv
        drivers (DataFrame): drivers.csv
        n_seasons (int): how many past seasons to include (default=3)
    """

    # --- Step 1: Join lap_times with race info ---
    df = lap_times.merge(races[['raceId', 'year', 'circuitId']], on='raceId', how='left')

    # --- Step 2: Restrict to last n seasons ---
    recent_years = sorted(df['year'].unique())[-n_seasons:]
    df = df[df['year'].isin(recent_years)]

    # --- Step 3: Count laps led per driver per circuit ---
    laps_led = (
        df[df['position'] == 1]
        .groupby(['driverId', 'circuitId'])
        .size()
        .reset_index(name='laps_led')
    )

    # --- Step 4: Merge with circuit + driver names ---
    laps_led = laps_led.merge(circuits[['circuitId', 'name', 'location', 'country']],
                              on='circuitId', how='left')
    laps_led = laps_led.merge(drivers[['driverId', 'driverRef', 'forename', 'surname']],
                              on='driverId', how='left')

    # --- Step 5: Clean columns ---
    laps_led = laps_led[['forename', 'surname', 'driverRef', 'name', 'country', 'laps_led']]
    laps_led = laps_led.sort_values(['name', 'laps_led'], ascending=[True, False])

    return laps_led

