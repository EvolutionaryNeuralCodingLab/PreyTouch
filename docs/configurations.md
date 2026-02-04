# Configurations

All system configurations can be shown and edited from the "Arena Configurations" panel.
To open this panel, you can click the cog icon (⚙) on the top-left side of PreyTouch UI.
![Image](/docs/images/arena_configurations.png)

# Configuration Files
There are 4 JSON configuration files in the system, that can be edited 
in PreyTocuh UI:
1. **cameras** - Specify all the cameras in the system. From this window
you can also scan the cameras in the system (only FLIR and Allied-Vision),
and also add a new camera using a simple form.
2. **periphery** - Specify all the peripheral devices 
3. **predict**
4. **agent**

## cam_config
- *camera name*
    - **id** - serial number of the camera
    - **module** - currently supported modules: flir, allied_vision
    - **exposure** - exposure in microseconds
    - **image_size** - image size in pixels
    - **output_dir** - leave blank. This is used by the video writing module.
    - **always_on** - if true, the scheduler tries to start the camera if off.
    - **fps** - If set, the camera takes frames according to this value.
    - **trigger_source** - if set, camera takes frames according to trigger
    - **writing_fps** - FPS for video writing, can be lower than camera's fps
    - **is_color** - set true, if the camera's frames are with 3 channels
    - **predictors** - specify real-time predictors
      - *predictor name* (must be configured in predict_config)
        - **image_size** - image size provided to the predictor
        - **mode** - can be "experiment" (run only during experiments), "no_experiment" (run only when there's no experiment running) 
                or "always" (run always).
        - **movement_type** - specify the specific movement types in which the predictor is allowed to run.
```json
{
    "front": {
        "id": 19506475,
        "module": "flir",
        "exposure": 7000,
        "image_size": [
            1080,
            1440
        ],
        "output_dir": null,
        "always_on": true,
        "fps": 60,
        "writing_fps": 60,
        "is_color": false,
        "predictors": {
            "tongue_out": {
                "image_size": [
                    1080,
                    1440
                ],
                "mode": "experiment",
                "movement_type": [
                    "jump_up",
                    "accelerate",
                    "circle_accelerate"
                ]
            }
        }
    },
    "top": {
        "id": 19506455,
        "module": "flir",
        "exposure": 4000,
        "image_size": [
            1080,
            1440
        ],
        "output_dir": null,
        "always_on": true,
        "fps": 10,
        "writing_fps": 10,
        "is_color": false,
        "mode": "tracking",
        "predictors": {}
    }
}
```
## periphery_config
- *arduino name*
  - **allow_get** - (bool) allow get status 
  - **fqbn** - Fully Qualified Board Name
  - **interfaces** - list of devices
    - **name** - device name
    - **pin** - single arduino pin 
    - **pins** - list of pins
    - **type** - can be line, feeder, dallas_temperature, trigger
  - **serial_number** - Arduino serial number
```json
{
    "arena": {
        "allow_get": true,
        "fqbn": "arduino:megaavr:nona4809",
        "interfaces": [
            {
                "command": "dispense",
                "name": "Feeder 1",
                "pins": [
                    10,
                    9,
                    8,
                    7
                ],
                "type": "feeder",
                "order": 1
            },
	        {
		        "command": "dispense",
                "name": "Feeder 2",
                "pins": [
                    14,
                    15,
                    16,
         	        17
                ],
                "type": "feeder",
                "order": 2
	        },
            {
                "name": "day_lights",
                "pin": 2,
                "type": "line"
            },
            {
                "name": "IR_lights",
                "pin": 13,
                "type": "line"
            },
            {
                "name": "Temp",
                "pin": 5,
                "type": "dallas_temperature"
            }
        ],
        "serial_number": "0E4273DD51534C5036202020FF072035"
    },
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
                "type": "trigger"
            }
        ],
        "serial_number": "91CEA40751534C5036202020FF07444F"
    }
}

```
## predict_config
- _predictor name_
  - **predictor name** - predictor class name
  - **model_path** - path to model
  - **threshold** - predictions thresholds
  - all the rest are kwargs for the predictor class
```json
{
    "deeplabcut": {
        "predictor_name": "DLCPose",
        "model_path": "/data/Pogona_Pursuit/output/models/deeplabcut/front_head_only_resnet_152",
        "bodyparts": [
            "nose",
            "right_ear",
            "left_ear"
        ],
        "threshold": 0.5
    },
    "pogona_head": {
        "model_path": "",
        "predictor_name": "PogonaHead",
        "threshold": 0.5
    },
    "tongue_out": {
        "model_path": "/data/Pogona_Pursuit/output/models/tongue_out/20230518_145847",
        "dataset_path": "/data/Pogona_Pursuit/output/datasets/pogona_tongue/dataset",
        "save_predicted_path": "/data/Pogona_Pursuit/output/datasets/pogona_tongue/predicted/tongues",
        "predictor_name": "TongueOutAnalyzer",
        "threshold": 0.7,
        "image_size": [
            550,
            1000
        ],
        "prediction_stack_duration": 0.25,
        "tongue_action_timeout": 0.5,
        "num_tongues_in_stack": 6
    }
}
```

## agent_config
- **default_struct** (default experiment parameters)
  - **time_between_blocks** - time in seconds between consecutive blocks.
  - **extra_time_recording** - time in seconds before and after the experiment for extra recording. The cameras record, but no trials are starting.
  - **num_blocks** - Number of blocks in each experiment.
  - **is_identical_blocks** - in case there are multiple blocks, make them all identical.
  - **is_test** - test experiment, app is started on the configured TEST_SCREEN and no rewards are given.
  - **reward_bugs** - specify the bugs which trigger reward. If null, all bugs are being rewarded.
  - **background_color** - app background color in hex.
  - **exit_hole**: can be "left", "right" or "random"
  - **reward_any_touch_prob** - probability to get reward even if missed.
  - **cameras** - specify all the cameras as keys and put any needed camera argument below (see example)
  - **blocks** - default block parameters:
    - **num_trials** - number of trials in each blocks
    - **trial_duration** - default trial duration in seconds
    - **iti** - inter trial interval in seconds
    - **block_type** - bugs or media 
    - **notes** - notes to be saved for the block
    - **bug_speed** - default bug speed
    - **is_default_bug_size** - use the default bug size, that specified in app the config
    - **bug_size** - bug size in pixels
- **times**
  - **start_time** - Time for the agent to start setting experiments (format: "HH:MM").
  - **end_time** - End time for the agent (format: "HH:MM").
  - **time_between_experiments** - Time in minutes between scheduled experiments.
- **success_announce** (optional)
  - **enabled** - Enable daily success announcements.
  - **min_trials_with_strikes** - Minimum number of trials with strikes required for evaluation.
  - **success_threshold** - Success ratio threshold (0-1) for announcing.
- **trials**
  - **repeat** (optional) - int repeat count or `"fill"` to fill the remaining daily slots.
  - **name_template** (optional) - supports `{time}` (HHMM) and `{index}` for generated trial names.
  - **evaluate_success** (optional) - whether to include this block in daily success evaluation.

#### Example
```json
{
    "default_struct": {
        "time_between_blocks": 180,
        "extra_time_recording": 30,
        "num_blocks": 1,
        "is_identical_blocks": false,
        "is_test": false,
        "reward_bugs": null,
        "background_color": "#e8eaf6",
        "exit_hole": "random",
        "reward_any_touch_prob": 0.1,
        "cameras": {
            "back": {
                "is_use_predictions": true
            },
            "front": {
                "is_use_predictions": true
            }
        },
        "blocks": [
            {
                "num_trials": 10,
                "trial_duration": 30,
                "iti": 20,
                "block_type": "bugs",
                "notes": "created by agent",
                "bug_speed": 5,
                "movement_type": null,
                "is_default_bug_size": true
            }
        ]
    },
    "times": {
        "start_time": "09:00",
        "end_time": "18:15",
        "time_between_blocks": 60
    },
    "trials": {
        "random_low_horizontal": {
            "count": {
                "key": "strikes",
                "amount": 40,
                "per": {
                    "bug_speed": [
                        2,
                        4,
                        6,
                        8
                    ]
                }
            },
            "exit_hole": "random",
            "bug_speed": "per_random",
            "movement_type": "random_low_horizontal"
        },
        "circle": {
            "count": {
                "key": "strikes",
                "amount": 40,
                "per": {
                    "bug_speed": [
                        2,
                        4,
                        6,
                        8
                    ]
                }
            },
            "exit_hole": "random",
            "bug_speed": "per_random",
            "reward_any_touch_prob": 0.1,
            "movement_type": "circle"
        },
        "circle_accelerate": {
            "count": {
                "key": "strikes",
                "amount": 100
            },
            "exit_hole": "random",
            "reward_any_touch_prob": 0.1,
            "movement_type": "circle_accelerate",
            "bug_speed": 4
        },
        "low_horizontal": {
            "count": {
                "key": "trials",
                "amount": 200,
                "per": {
                    "exit_hole": [
                        "bottomLeft",
                        "bottomRight"
                    ]
                }
            },
            "exit_hole": "per_ordered",
            "bug_speed": 6,
            "movement_type": "low_horizontal",
            "reward_any_touch_prob": 0,
            "num_trials": 5
        }
    }
}
```
