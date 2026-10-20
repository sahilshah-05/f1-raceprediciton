# f1-raceprediciton

This repository now includes a baseline machine learning workflow to predict the
full race outcome for a Formula 1 race.

## What the model uses

The predictor is trained at the **driver-race** level and ranks all drivers in
an upcoming race using:

- previous race position for each driver
- recent form (rolling average of recent race positions)
- historical performance for each driver at the same circuit
- weather condition

## Files

- `/home/runner/work/f1-raceprediciton/f1-raceprediciton/f1_race_prediction/model.py`:
  model implementation
- `/home/runner/work/f1-raceprediciton/f1-raceprediciton/tests/test_model.py`:
  focused tests for feature generation and full-race ranking

## Quick usage

```python
import pandas as pd
from f1_race_prediction.model import FullRaceOutcomePredictor

historical = pd.DataFrame(
    [
        {"race_id": "r1", "race_date": "2024-03-01", "driver_id": "VER", "circuit_id": "monza", "weather": "dry", "finish_position": 1},
        {"race_id": "r1", "race_date": "2024-03-01", "driver_id": "NOR", "circuit_id": "monza", "weather": "dry", "finish_position": 2},
    ]
)

upcoming = pd.DataFrame(
    [
        {"race_id": "r2", "race_date": "2024-03-08", "driver_id": "VER", "circuit_id": "monza", "weather": "wet"},
        {"race_id": "r2", "race_date": "2024-03-08", "driver_id": "NOR", "circuit_id": "monza", "weather": "wet"},
    ]
)

model = FullRaceOutcomePredictor()
model.fit(historical)
prediction = model.predict_race_outcome(upcoming)
print(prediction)
```