import os
import pandas as pd

FILE_PATH = os.path.abspath(os.path.dirname(__file__))


def load_data(league: str, season: str) -> pd.DataFrame:
    file_name = "_".join([league, season]) + ".csv"
    file_path = os.path.join(FILE_PATH, "..", "data", file_name)

    return pd.read_csv(file_path)
