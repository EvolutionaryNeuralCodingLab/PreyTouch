# Getting Started
## Access the User Interface
In order to open PreyTouch user interface, open any internet browser (we recommend using Mozilla Firefox) and go to:
```commandline
http://localhost:5084
```
In case you want to access PreyTouch from another computer on the same network as the server running PreyTouch, simply change "localhost" to the hostname of the server.

Changing the port number can be done using the environment variable: "MANAGEMENT_PORT"

## System Configurations
To get into system configurations click on the cog icon (⚙) which is found on the top-left side of the screen. 
### Cameras
Click the "cameras" button, and you'll see the cam_config.json file. 
This json can be manually edited online and saved by click "save". 
However, you can simplify the process for adding a new camera by clicking the button "Add New Camera". 
Notice that the system currently supports FLIR and Allied Vision cameras only. 
Moreover, to add a new camera you'll need its serial number. This can easily be found by clicking the "Cam Scan" button.
For more information on
[adding a new camera to the system](docs/new_camera.md) and on the [cam_config](docs/configurations.md##cam_config): 

### Periphery
Here you can configure all the peripheral devices, such as: Feeders, lights, temperature sensors or any switch 
that you want the system to manage.
The control over peripheral devices is done through an Arduino. To scan the arduinos that are connected to the server, you can click the "Serial Scan" button.
For an example of how the periphery config file should, check: 
## Animal Configurations