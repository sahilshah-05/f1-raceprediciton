from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class FullRaceOutcomePredictor:
    recent_races: int = 3
    random_state: int = 42

    def __post_init__(self) -> None:
        self._required_train_columns = {
            "race_id",
            "race_date",
            "driver_id",
            "circuit_id",
            "weather",
            "finish_position",
        }
        self._required_predict_columns = {
            "race_id",
            "race_date",
            "driver_id",
            "circuit_id",
            "weather",
        }
        self._feature_columns = [
            "driver_id",
            "circuit_id",
            "weather",
            "previous_race_position",
            "recent_form_position",
            "circuit_history_position",
        ]

    def _validate_columns(self, frame: pd.DataFrame, required: set[str]) -> None:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def _sorted(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        data["race_date"] = pd.to_datetime(data["race_date"])
        return data.sort_values(["race_date", "race_id", "driver_id"]).reset_index(drop=True)

    def _build_history_features(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        data = self._sorted(historical_data)

        data["previous_race_position"] = data.groupby("driver_id")["finish_position"].shift(1)
        data["recent_form_position"] = data.groupby("driver_id")["finish_position"].transform(
            lambda values: values.shift(1).rolling(window=self.recent_races, min_periods=1).mean()
        )
        data["circuit_history_position"] = data.groupby(["driver_id", "circuit_id"])[
            "finish_position"
        ].transform(lambda values: values.shift(1).expanding().mean())

        global_mean = data["finish_position"].mean()
        data["previous_race_position"] = data["previous_race_position"].fillna(global_mean)
        data["recent_form_position"] = data["recent_form_position"].fillna(global_mean)
        data["circuit_history_position"] = data["circuit_history_position"].fillna(global_mean)
        return data

    def prepare_training_data(self, historical_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        self._validate_columns(historical_data, self._required_train_columns)
        feature_frame = self._build_history_features(historical_data)
        return feature_frame[self._feature_columns], feature_frame["finish_position"]

    def fit(self, historical_data: pd.DataFrame) -> "FullRaceOutcomePredictor":
        self._validate_columns(historical_data, self._required_train_columns)
        engineered = self._build_history_features(historical_data)
        X_train = engineered[self._feature_columns]
        y_train = engineered["finish_position"]

        numeric_features = [
            "previous_race_position",
            "recent_form_position",
            "circuit_history_position",
        ]
        categorical_features = ["driver_id", "circuit_id", "weather"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", numeric_features),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_features,
                ),
            ]
        )

        self.model_ = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        self.model_.fit(X_train, y_train)

        sorted_history = self._sorted(historical_data)
        self.global_mean_position_ = sorted_history["finish_position"].mean()
        self.driver_last_position_: Dict[str, float] = (
            sorted_history.groupby("driver_id")["finish_position"].last().to_dict()
        )
        self.driver_recent_position_: Dict[str, float] = (
            sorted_history.groupby("driver_id")["finish_position"]
            .apply(lambda values: values.tail(self.recent_races).mean())
            .to_dict()
        )
        self.driver_circuit_position_: Dict[Tuple[str, str], float] = (
            sorted_history.groupby(["driver_id", "circuit_id"])["finish_position"].mean().to_dict()
        )
        return self

    def _build_prediction_features(self, upcoming_race: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(upcoming_race, self._required_predict_columns)
        if not hasattr(self, "model_"):
            raise ValueError("Model must be fitted before predicting.")

        prediction_data = self._sorted(upcoming_race)

        prediction_data["previous_race_position"] = prediction_data["driver_id"].map(
            self.driver_last_position_
        )
        prediction_data["recent_form_position"] = prediction_data["driver_id"].map(
            self.driver_recent_position_
        )
        prediction_data["circuit_history_position"] = prediction_data.apply(
            lambda row: self.driver_circuit_position_.get((row["driver_id"], row["circuit_id"])),
            axis=1,
        )

        prediction_data["previous_race_position"] = prediction_data["previous_race_position"].fillna(
            self.global_mean_position_
        )
        prediction_data["recent_form_position"] = prediction_data["recent_form_position"].fillna(
            self.global_mean_position_
        )
        prediction_data["circuit_history_position"] = prediction_data["circuit_history_position"].fillna(
            self.global_mean_position_
        )
        return prediction_data

    def predict_race_outcome(self, upcoming_race: pd.DataFrame) -> pd.DataFrame:
        prediction_data = self._build_prediction_features(upcoming_race)
        scores = self.model_.predict(prediction_data[self._feature_columns])
        prediction_data["predicted_score"] = scores
        prediction_data = prediction_data.sort_values("predicted_score", ascending=True).reset_index(
            drop=True
        )
        prediction_data["predicted_finish_position"] = range(1, len(prediction_data) + 1)
        return prediction_data[
            [
                "race_id",
                "race_date",
                "driver_id",
                "circuit_id",
                "weather",
                "predicted_score",
                "predicted_finish_position",
            ]
        ]
