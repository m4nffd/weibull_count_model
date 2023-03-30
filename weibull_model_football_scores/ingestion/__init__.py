import datetime
import os
import pandas as pd

FILE_PATH = os.path.abspath(os.path.dirname(__file__))


def load_data(league: str, season: str) -> pd.DataFrame:
    file_name = "_".join([league, season]) + ".csv"
    file_path = os.path.join(FILE_PATH, "..", "data", file_name)

    df = pd.read_csv(file_path)
    df.sort_values(by="Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["fixture"] = df.index // 10
    df = df[["fixture", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].copy()

    return df


def load_prediction_dataset(league: str, season: str) -> pd.DataFrame:
    # Downloaded from https://cdn.bettingexpert.com/assets/Italy-Serie-A-Season-2022-2023-Fixture.csv
    file_name = "_".join([league, season, "fixtures"]) + ".csv"
    file_path = os.path.join(FILE_PATH, "..", "data", file_name)

    df = pd.read_csv(file_path)

    # Filter only on future events

    today = datetime.datetime.now()
    cond = (
        pd.to_datetime(
            df["Year"].astype(str)
            + "-"
            + df["Month"].astype(str)
            + "-"
            + df["Date"].astype(str)
        )
        >= today
    )

    df = df[cond].copy()

    # Remove AS, AC (e.g. AS Roma)
    df["Home"] = df["Home"].apply(lambda x: x.split()[-1])
    df["Away"] = df["Away"].apply(lambda x: x.split()[-1])

    return df.reset_index(drop=True)
