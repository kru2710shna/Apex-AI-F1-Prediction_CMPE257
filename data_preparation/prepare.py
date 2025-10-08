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


