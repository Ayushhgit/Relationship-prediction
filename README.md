# Bodycount Prediction — Synthetic Dataset & Model

This repository contains code and a trained model for predicting a synthetic "bodycount" target. It includes dataset generation, a model notebook, and a lightweight prediction script.

## Contents
- `generate_dataset.py` — script to (re)generate the synthetic dataset (`synthetic_bodycount_dataset.csv`).
- `model.ipynb` — notebook used for feature engineering, training, and evaluation.
- `prediction.py` — small script example showing how to load the trained pipeline and run predictions.
- `synthetic_bodycount_dataset.csv` — example dataset used in the notebook.
- `models/` — folder containing trained artifacts (example: `lgbm_bodycount_pipeline.joblib`).

## Requirements
Install the typical data science packages used in the notebook and inference script. Example:

```bash
pip install pandas scikit-learn lightgbm joblib numpy
```

Adjust versions to match those used when training in `model.ipynb` if reproducing results.

## Quickstart

1. (Optional) Regenerate dataset:

```bash
python generate_dataset.py
```

2. Run the notebook `model.ipynb` to retrain or inspect the model.

3. Run inference using the saved pipeline in `models/lgbm_bodycount_pipeline.joblib`.

Example: predict using `prediction.py` (assumes the `models/` file exists):

```bash
python prediction.py --input synthetic_bodycount_dataset.csv --model models/lgbm_bodycount_pipeline.joblib
```

Or use a short Python snippet to load the pipeline and predict:

```python
import joblib
import pandas as pd

# load saved pipeline
pipe = joblib.load('models/lgbm_bodycount_pipeline.joblib')

# load some data (adjust columns to match training features)
df = pd.read_csv('synthetic_bodycount_dataset.csv')

# run predictions
preds = pipe.predict(df)
print(preds[:10])
```

## Notes
- The `lgbm_bodycount_pipeline.joblib` is a scikit-learn pipeline wrapping preprocessing + LightGBM.
- Ensure feature columns passed to the pipeline match what was used during training.

