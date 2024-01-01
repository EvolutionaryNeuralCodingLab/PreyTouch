# Adding New Camera

PreyTouch supports FLIR and AlliedVision cameras. In case you want to use a different camera, you have to build a custom camera class inside PreyTouch. Check below for how to add custom camera.

Open the cam_config.yaml file

```console
nano Arena/configurations/cam_config.yaml
```

and insert a new camera section:

```yaml
front: # camera name
    id: 19506475  # serial number of the camera
    module: flir  # currently supported modules: flir, allied_vision
    exposure: 7000  # exposure in microseconds
    image_size: [ 1080, 1440 ]  # image size in pixels
    output_dir:  # leave blank. This is used by the video writing module.
    always_on: true  # if true, the scheduler tries to start the camera if off.
    fps: 60  # If set, the camera takes frames according to this value. 
    writing_fps: 60  # FPS for video writing, can be lower than camera's fps
    is_color: false  # set true, if the camera's frames are with 3 channels (color)
    predictors: {}
```

**Notice!**

- if it's an alliedVision camera, change to "module: allied_vision"
- If you want the camera to work with an external trigger; remove the "fps: 60" and instead set: "trigger_source: Line3", and replace "Line3" with the trigger source name.

Finally, restart PreyTouch and you should see the new camera.

## Install camera SDK

Although PreyTouch supports FLIR and alliedVision cameras intergration, you still need to install the cameras SDK.
In order to check whether the full SDK is installed, you need to be able to import the following in your python interpreter:

- For FLIR cameras:

```python
import PySpin
```

If any errors occur, please visit <https://www.flir.eu/products/spinnaker-sdk> and install both the SDK and python package that are suitable for your system.

- For allied vision cameras:

```python
import Vimba
```

If any errors occur, please visit <https://www.alliedvision.com/en/products/vimba-sdk> and install vimba.

## Custom Camera Module

In case you have cameras which are not FLIR or AlliedVision, then you have to create a custom module for them.

All camera modules are located in "Arena/cameras". Create a new python file there for your custom camera. Inside this new module put the following:

```python
from arena import Camera

class CustomCamera(Camera):
    def _run(self):
        """Camera class must have _run method. This is the main function 
        to run inside the camera process"""
        
        # camera init

        try:
            while not self.stop_signal.is_set():
                # here goes the frames acquisition process
                pass
        except Exception:
            self.logger.exception('Error in camera:')
```
