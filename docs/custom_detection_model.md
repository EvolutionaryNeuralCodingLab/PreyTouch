## Adding a detection model to the system

1. Create a python class for your model with a “predict” method that gets a frame (numpy array) as its first argument and returns a pandas dataframe with the predictions.
This class should also inherit the Predictor class found in:
Arena/analysis/predictors/base.py
2. Register your model in Arena/config.py under the dictionary arena_modules[‘predictors’]
3. Add configuration to your model in predict_config.yaml (check below in Configurations)
4. Restart the arena
