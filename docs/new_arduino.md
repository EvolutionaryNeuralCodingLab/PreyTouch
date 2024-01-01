# Adding Arduino

## Install arduino-cli and packages

In case you alreay did it, you can skip this step.
This manual was taken from the reptilearn project <https://github.com/neural-electrophysiology-tool-team/reptilearn>

1. install arduino-cli (activate your python interpreter before running)

```console
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | BINDIR=$(dirname $(which python)) sh
```

2. Once installed, run these commands to install the necessary Arduino libraries:

```console
arduino-cli core update-index
arduino-cli lib install AccelStepper ArduinoJson OneWire DallasTemperature
```

3. Finally, install the software for your specific Arduino board model(s). For example for an Arduino Nano Every or UNO WiFi Rev 2 run:

```console
arduino-cli core install arduino:megaavr
```

For other models, the following command will list all available board IDs:

```console
arduino-cli core list --all 
```

## Add new arduino to periphery_config.json

1. First, you need the arduino serial number

```console
cd periphery
python main.py --list-ports
```

2. Get the serial number from the output and create a new item in Arena/configurations/periphery_config.json:

```json
# example of adding an arduino for camera trigger of 30Hz

"camera trigger": {
    "allow_get": true,
    "fqbn": "arduino:megaavr:nona4809",
    "interfaces": [
        {
            "name": "Camera Trigger",
            "pin": 12,
            "pulse_len": 33,
            "pulse_width": 0.7,
            "serial_trigger": false,
            "type": "trigger",
            "ui": "camera_trigger"
        }
    ],
    "serial_number": "<arduino_serial_number>"
}
```

```json
# example of adding an arduino for feeder, day lights, IR lights and temperature sensors

"arena": {
    "allow_get": true,
    "fqbn": "arduino:megaavr:nona4809",
    "interfaces": [
        {
            "command": "dispense",
            "icon": "gift",
            "name": "Feeder 1",
            "pins": [
                10,
                9,
                8,
                7
            ],
            "type": "feeder",
            "ui": "action",
            "order": 1
        },
        {
            "name": "day_lights",
            "pin": 2,
            "type": "line",
            "ui": "toggle"
        },
        {
            "name": "IR_lights",
            "pin": 14,
            "type": "line",
            "ui": "toggle"
        },
        {
            "icon": "thermometer-half",
            "name": "Temp",
            "pin": 5,
            "type": "dallas_temperature",
            "ui": "sensor",
            "unit": "\u00b0"
        }
    ],
    "serial_number": "0E4273DD51534C5036202020FF072035"
}
```

3. Restart the periphery container

```console
cd docker/
docker-compose restart periphery
```

4. Restart PreyTouch
