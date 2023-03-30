import os.path

import pandas as pd
import numpy as np
from weibull_model_football_scores.probs.weibull import weibull_likelihood
from weibull_model_football_scores import config
from weibull_model_football_scores.processing.functions import (
    get_all_teams,
)

FILE_ROOT = os.path.abspath(os.path.dirname(__file__))


class WeibullCountFitter:
    def __init__(self):
        self.gamma = config.GAMMA

    def _get_all_teams(self, df: pd.DataFrame):
        self.teams = get_all_teams(df)
        return self

    def _calculate_lambda(self, a: float, b: float, home: bool):
        _lambda = a + b
        if home:
            _lambda += self.gamma
        return _lambda

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

    @property
    def team_mapping(self):
        return {team: n for n, team in enumerate(self.teams)}

    @property
    def inverse_team_mapping(self):
        return {v: k for k, v in self.team_mapping.items()}

    @property
    def C_team(self):
        X = pd.DataFrame(self.C, columns=["alpha", "beta"])
        X.reset_index(inplace=True)
        X.rename(columns={"index": "team"}, inplace=True)
        X["team"] = X["team"].map(self.inverse_team_mapping)
        return X

    def _initialise_coefficients(self):
        self.C = abs(np.random.normal(0, 1, (len(self.teams), 2)))

    def _get_likelihood(self, C: np.array):
        df = self.df.copy()
        log_l = 0

        max_fixture = df["fixture"].max()

        for i, row in df.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            i, j = self.team_mapping[home], self.team_mapping[away]
            l1 = self._calculate_lambda(C[i][0], C[j][1], home=True)
            l2 = self._calculate_lambda(C[j][0], C[i][1], home=False)
            log_l += np.log(
                np.exp(-config.XI * (max_fixture - row["fixture"]))
                * weibull_likelihood(y1=row["FTHG"], y2=row["FTAG"], l1=l1, l2=l2)
            )

        return log_l

    def _get_grads(self, C: np.array):
        C = C.copy()
        eps = 1e-6

        grads = np.zeros_like(C)

        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                C_plus = C.copy()
                C_minus = C.copy()
                C_plus[i, j] += eps
                C_minus[i, j] -= eps
                grads[i, j] = (
                    self._get_likelihood(C_plus) - self._get_likelihood(C_minus)
                ) / (2 * eps)

        return grads

    def fit(
        self,
        df: pd.DataFrame,
        n_iter: int = 100,
        learning_rate: float = 0.001,
        verbose: bool = True,
        save_model: bool = True,
    ):
        self.df = df.copy()
        self._get_all_teams(df)
        self._initialise_coefficients()

        log_l = self._get_likelihood(self.C)
        if verbose:
            print(f"Starting likelihood: {log_l}")

        self.likelihoods = [log_l]

        C = self.C.copy()
        for n in range(n_iter):
            C += learning_rate * self._get_grads(C)
            self.C = C.copy()
            log_l = self._get_likelihood(self.C)
            self.likelihoods.append(log_l)

            if verbose:
                print(f"Step {n + 1}, current likelihood: {log_l}")

            if self.likelihoods[-1] - self.likelihoods[-2] < 10e-4:
                print("Early stop. Algorithm has converged")
                break

        if save_model:
            pd.to_pickle(
                self.C,
                filepath_or_buffer=os.path.join(
                    FILE_ROOT, "..", "assets", f"model.pickle"
                ),
            )

    def predict_single_match(self, home: str, away: str):
        results = []

        C = self.C_team.set_index("team").to_dict(orient="index")
        for i in config.GOAL_RANGE:
            for j in config.GOAL_RANGE:
                l1 = self._calculate_lambda(
                    C[home]["alpha"], C[away]["beta"], home=True
                )
                l2 = self._calculate_lambda(
                    C[home]["beta"], C[away]["alpha"], home=False
                )
                p = weibull_likelihood(i, j, l1, l2)
                results.append([i, j, p])

        X = pd.DataFrame(results, columns=["H", "A", "p"])

        X["over"] = X["H"] + X["A"] > 2.5

        under = X[~X["over"]]["p"].sum()
        over = 1 - under

        return (under, over)

    def predict(
        self, df: pd.DataFrame, how: str, under_over: float = 2.5, score: str = None
    ) -> pd.DataFrame:
        df = df.copy()

        res = df.apply(
            lambda row: self.predict_single_match(
                row["Home"],
                row["Away"],
            ),
            axis=1,
        ).apply(pd.Series)

        res.rename(columns={0: "under", 1: "over"}, inplace=True)

        return res

    def get_results(self):
        return pd.DataFrame(self.C).T
