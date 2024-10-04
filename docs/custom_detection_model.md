## Adding a detection model to the system

The system supports integration with new visual models for detection. Example of detection models:
- model that detects an action made by the animal before striking the screen
- custom pose detection model
- classification of multiple animals

For adding such model to the system, so you can use in real-time or for offline analysis:

1. Create a python class for your model with a “predict” method that gets a frame (numpy array) as its first argument and returns a pandas dataframe with the predictions.
This class should also inherit the Predictor class found in:
Arena/analysis/predictors/base.py
2. Register your model in Arena/config.py under the dictionary arena_modules[‘predictors’]
3. Add configuration to your model in predict_config.yaml (check below in Configurations)
4. Restart the arena

Now the model is integrated with the system.

For using the model in real-time pre-strike:
1. Open the file "Arena/image_handlers/predictor_handlers"
2. Add a new class that inherits the PredictHandler class
3. Check the TongueOutHandler for how the logic goes. basically you should write the init(), predict_frame() and on_stop() methods for the new class. You can also use the analyze_prediction() and before_predict() methods if needed.
4. Open the "Arena/config.py" file 
5. Under *arena_modules.image_handlers* add a new image handler with its name as key and the value should be a tuple of ('image_handlers.predictor_handlers', \<the name of the class in the predictor_handlers file\>)
6. For connecting this new predictor with a certain camera stream, open the cam_config.json configuration file and under the "predictors" of the designated camera add this new predictor. Check [cam_config](configurations.md#cam_config) for more information.
