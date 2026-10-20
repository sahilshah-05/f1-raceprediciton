import unittest

import pandas as pd

from f1_race_prediction.model import FullRaceOutcomePredictor


class FullRaceOutcomePredictorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.historical_data = pd.DataFrame(
            [
                {
                    "race_id": "r1",
                    "race_date": "2024-01-01",
                    "driver_id": "VER",
                    "circuit_id": "monza",
                    "weather": "dry",
                    "finish_position": 1,
                },
                {
                    "race_id": "r1",
                    "race_date": "2024-01-01",
                    "driver_id": "NOR",
                    "circuit_id": "monza",
                    "weather": "dry",
                    "finish_position": 2,
                },
                {
                    "race_id": "r1",
                    "race_date": "2024-01-01",
                    "driver_id": "HAM",
                    "circuit_id": "monza",
                    "weather": "dry",
                    "finish_position": 3,
                },
                {
                    "race_id": "r2",
                    "race_date": "2024-01-08",
                    "driver_id": "VER",
                    "circuit_id": "spa",
                    "weather": "wet",
                    "finish_position": 1,
                },
                {
                    "race_id": "r2",
                    "race_date": "2024-01-08",
                    "driver_id": "NOR",
                    "circuit_id": "spa",
                    "weather": "wet",
                    "finish_position": 2,
                },
                {
                    "race_id": "r2",
                    "race_date": "2024-01-08",
                    "driver_id": "HAM",
                    "circuit_id": "spa",
                    "weather": "wet",
                    "finish_position": 3,
                },
                {
                    "race_id": "r3",
                    "race_date": "2024-01-15",
                    "driver_id": "VER",
                    "circuit_id": "monza",
                    "weather": "dry",
                    "finish_position": 1,
                },
                {
                    "race_id": "r3",
                    "race_date": "2024-01-15",
                    "driver_id": "NOR",
                    "circuit_id": "monza",
                    "weather": "dry",
                    "finish_position": 2,
                },
                {
                    "race_id": "r3",
                    "race_date": "2024-01-15",
                    "driver_id": "HAM",
                    "circuit_id": "monza",
                    "weather": "dry",
                    "finish_position": 3,
                },
            ]
        )

    def test_prepare_training_data_includes_history_and_weather_features(self) -> None:
        predictor = FullRaceOutcomePredictor(recent_races=2)
        X_train, y_train = predictor.prepare_training_data(self.historical_data)

        self.assertEqual(len(X_train), len(y_train))
        self.assertTrue(
            {"previous_race_position", "recent_form_position", "circuit_history_position"}.issubset(
                X_train.columns
            )
        )
        self.assertFalse(X_train["previous_race_position"].isna().any())
        self.assertFalse(X_train["recent_form_position"].isna().any())
        self.assertFalse(X_train["circuit_history_position"].isna().any())
        self.assertIn("weather", X_train.columns)

    def test_predict_race_outcome_ranks_full_grid(self) -> None:
        predictor = FullRaceOutcomePredictor(recent_races=2, random_state=1)
        predictor.fit(self.historical_data)

        upcoming_race = pd.DataFrame(
            [
                {
                    "race_id": "r4",
                    "race_date": "2024-01-22",
                    "driver_id": "VER",
                    "circuit_id": "monza",
                    "weather": "dry",
                },
                {
                    "race_id": "r4",
                    "race_date": "2024-01-22",
                    "driver_id": "NOR",
                    "circuit_id": "monza",
                    "weather": "dry",
                },
                {
                    "race_id": "r4",
                    "race_date": "2024-01-22",
                    "driver_id": "HAM",
                    "circuit_id": "monza",
                    "weather": "dry",
                },
            ]
        )

        prediction = predictor.predict_race_outcome(upcoming_race)
        self.assertEqual(list(prediction["predicted_finish_position"]), [1, 2, 3])
        self.assertEqual(prediction.iloc[0]["driver_id"], "VER")
        self.assertEqual(set(prediction["driver_id"]), {"VER", "NOR", "HAM"})


if __name__ == "__main__":
    unittest.main()
