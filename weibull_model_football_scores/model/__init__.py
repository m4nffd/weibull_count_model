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
        return np.exp(_lambda)

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
        self.C = 0.1 * (np.random.normal(0, 1, (len(self.teams), 2)))

    def _get_likelihood(self, df: pd.DataFrame, C: np.array) -> float:
        log_l = 0

        max_fixture = df["fixture"].max()

        import time

        for i, row in df.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            i, j = self.team_mapping[home], self.team_mapping[away]
            l1 = self._calculate_lambda(C[i][0], C[j][1], home=True)
            l2 = self._calculate_lambda(C[j][0], C[i][1], home=False)
            #   print(l1, l2, row["FTHG"], row["FTAG"])
            #   print(
            #       np.exp(-config.XI * (max_fixture - row["fixture"])),
            #       weibull_likelihood(y1=row["FTHG"], y2=row["FTAG"], l1=l1, l2=l2),
            #   )
            #   print(
            #       np.log(
            #           np.exp(-config.XI * (max_fixture - row["fixture"]))
            #           * weibull_likelihood(y1=row["FTHG"], y2=row["FTAG"], l1=l1, l2=l2)
            #       )
            #   )
            log_l += np.log(
                np.exp(-config.XI * (max_fixture - row["fixture"]))
                * weibull_likelihood(y1=row["FTHG"], y2=row["FTAG"], l1=l1, l2=l2)
            )

        return log_l

    def _get_grads(self, df: pd.DataFrame, C: np.array):
        C = C.copy()
        df = df.copy()
        eps = 1e-6

        grads = np.zeros_like(C)

        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                C_plus = C.copy()
                C_minus = C.copy()
                C_plus[i, j] += eps
                C_minus[i, j] -= eps
                grads[i, j] = (
                    self._get_likelihood(df, C_plus) - self._get_likelihood(df, C_minus)
                ) / (2 * eps)

        return grads

    def fit(
        self,
        train: pd.DataFrame,
        n_iter: int = 100,
        learning_rate: float = 0.001,
        verbose: bool = True,
        save_model: bool = True,
        test: pd.DataFrame = None,
    ):
        train = train.copy()
        self._get_all_teams(train)
        self._initialise_coefficients()

        train_log_l = self._get_likelihood(train, self.C)
        if test is not None:
            test_log_l = self._get_likelihood(test, self.C)

        if verbose:
            print(f"Starting train likelihood: {train_log_l}")
            if test is not None:
                print(f"Starting test likelihood: {test_log_l}")

        self.train_likelihoods = [train_log_l]
        if test is not None:
            self.test_likelihoods = [test_log_l]

        C = self.C.copy()
        for n in range(n_iter):
            # C += learning_rate * self._get_grads(train, C)
            C += learning_rate * self._get_grads(train, C)
            self.C = C.copy()
            train_log_l = self._get_likelihood(train, self.C)
            self.train_likelihoods.append(train_log_l)
            if test is not None:
                test_log_l = self._get_likelihood(test, self.C)
                self.test_likelihoods.append(test_log_l)

            if verbose:
                print(f"Step {n + 1}, current likelihood: {train_log_l}")
                if test is not None:
                    print(f"Step {n + 1}, current test likelihood: {test_log_l}")

            if self.train_likelihoods[-1] - self.train_likelihoods[-2] < 10e-4:
                print("Early stop. Algorithm has converged")
                break

        if save_model:
            pd.to_pickle(
                self.C,
                filepath_or_buffer=os.path.join(
                    FILE_ROOT, "..", "assets", f"model.pickle"
                ),
            )

    def predict_all_scores_single_match(self, home: str, away: str):
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

        return X

    def predict_under_over_single_match(self, home: str, away: str):
        X = self.predict_all_scores_single_match(home, away)

        X["over"] = X["H"] + X["A"] > 2.5

        under = X[~X["over"]]["p"].sum()
        over = 1 - under

        return (under, over)

    def predict_under_over(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        df = df.copy()

        res = df.apply(
            lambda row: self.predict_under_over_single_match(
                row["HomeTeam"],
                row["AwayTeam"],
            ),
            axis=1,
        ).apply(pd.Series)

        res.rename(columns={0: "under", 1: "over"}, inplace=True)

        return res

    def predict_1x2_single_match(self, home: str, away: str):
        X = self.predict_all_scores_single_match(home, away)

        X["pred"] = "0"
        X.loc[X["H"] > X["A"], "pred"] = "1"
        X.loc[X["H"] < X["A"], "pred"] = "2"
        X.loc[X["H"] == X["A"], "pred"] = "X"

        scores = X.groupby("pred")["p"].sum()
        scores /= (
            scores.sum()
        )  # Renormalising to 1 as we are not calculating p for all possible scores

        return scores.T

    def predict_1x2(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        df = df.copy()

        res = df.apply(
            lambda row: self.predict_1x2_single_match(
                row["HomeTeam"],
                row["AwayTeam"],
            ),
            axis=1,
        )

        return res

    def get_results(self):
        return pd.DataFrame(self.C).T
