import kagglehub
from kagglehub import KaggleDatasetAdapter

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
