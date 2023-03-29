import pandas as pd
import numpy as np
import numdifftools as nd
from weibull_model_football_scores.probs.weibull import weibull_likelihood
from weibull_model_football_scores import config
from weibull_model_football_scores.processing.functions import (
    get_all_teams,
    get_all_matches_by_team,
)


class WeibullCountFitter:
    def __init__(self):
        self.gamma = config.GAMMA

    def _get_all_teams(self, df: pd.DataFrame):
        self.teams = get_all_teams(df)
        return self

    def _initialise_coefficients(self):
        self.C = {}

        for team in self.teams:
            self.C[team] = {}
            self.C[team]["alpha"] = np.random.uniform()
            self.C[team]["beta"] = np.random.uniform()
        return self

    def _calculate_lambda(self, a: float, b: float, home: bool):
        _lambda = a + b
        if home:
            _lambda += self.gamma
        return _lambda

    def _get_team_likelihood(self, a: float, b: float, team: str):
        _df = get_all_matches_by_team(self.df, team)

        log_likelihood = 0
        for i, row in _df.iterrows():
            if team == row["HomeTeam"]:
                _a = self.C[row["AwayTeam"]]["alpha"]
                _b = self.C[row["AwayTeam"]]["beta"]
                l1 = self._calculate_lambda(a=a, b=_b, home=True)
                l2 = self._calculate_lambda(a=_a, b=b, home=False)
            else:
                _a = self.C[row["HomeTeam"]]["alpha"]
                _b = self.C[row["HomeTeam"]]["beta"]
                l1 = self._calculate_lambda(a=_a, b=b, home=True)
                l2 = self._calculate_lambda(a=a, b=_b, home=False)

            home_goals = row["FTHG"]
            away_goals = row["FTAG"]

            log_likelihood += np.log(
                weibull_likelihood(home_goals, away_goals, l1=l1, l2=l2)
            )

        return log_likelihood

    # These two functions are needed to well-define the derivatives
    def _f_a(self, a: float, b: float, team: str):
        return self._get_team_likelihood(a, b, team)

    def _f_b(self, b: float, a: float, team: str):
        return self._get_team_likelihood(a, b, team)

    def _d_da(self):
        return nd.Derivative(self._f_a)

    def _d_db(self):
        return nd.Derivative(self._f_b)

    def _get_current_likelihood(self):
        _likelihood = 0
        for i, row in self.df.iterrows():
            home_team = row["HomeTeam"]
            away_team = row["AwayTeam"]
            home_goals = row["FTHG"]
            away_goals = row["FTAG"]

            l1 = self._calculate_lambda(
                self.C[home_team]["alpha"], self.C[away_team]["beta"], home=True
            )
            l2 = self._calculate_lambda(
                self.C[home_team]["beta"], self.C[away_team]["alpha"], home=False
            )

            _likelihood += weibull_likelihood(home_goals, away_goals, l1, l2)

        return _likelihood

    def fit(
        self,
        df: pd.DataFrame,
        n_iter: int = 100,
        learning_rate: float = 0.001,
        verbose=True,
    ):
        self._get_all_teams(df)
        self._initialise_coefficients()

        self.df = df.copy()

        print(f"Starting likelihood: {self._get_current_likelihood()}")
        self.likelihoods = []

        for n in range(n_iter):
            for team in self.teams:
                a, b = self.C[team].values()
                self.C[team]["alpha"] += learning_rate * self._d_da()(a, b, team)
                self.C[team]["beta"] += learning_rate * self._d_db()(b, a, team)

            log_l = self._get_current_likelihood()

            if verbose:
                print(f"Step {n+1}, current likelihood: {log_l}")
