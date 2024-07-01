# New Deeplabcut Model

### Model Preparation
1. To add a new DeepLabCut model to PreyTouch one must first export a trained model.
https://github.com/DeepLabCut/DeepLabCut/blob/main/docs/HelperFunctions.md#model-export-function
2. The exported model directory should have files like these:
   - pose_cfg.yaml
   - snapshot-500000.data-00000-of-00001
   - snapshot-500000.index
   - snapshot-500000.meta
   - snapshot-500000.pb
   - snapshot-500000.pbtxt

### Adding the New Model
1. In PreyTouch UI go to settings (⚙ icon on top-left) > Predict > Add New Model
2. In the opened form: 
   - The name you want this model to be called in PreyTouch
   - Path of the exported model directory
   - Model body parts separated by "," and in the same order as in deeplabcut's config.yaml
   - The threshold to use for each body part.
3. Press "Add" and you should see the new predict model in the Predict json.