import pandas as pd
from typing import List


def get_all_matches_by_team(df: pd.DataFrame, team: str) -> pd.DataFrame:
    df = df.copy()

    df = df[(df["HomeTeam"] + df["AwayTeam"]).apply(lambda x: team in x)].copy()
    return df


def get_all_teams(df: pd.DataFrame) -> List:
    df = df.copy()

    all_teams = list(set(list(df["HomeTeam"].unique()) + list(df["AwayTeam"].unique())))

    return all_teams
