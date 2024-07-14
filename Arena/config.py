import yaml
import socket
import redis
import json
from environs import Env
from pathlib import Path
from functools import wraps
from config_utils import Conf, load_configuration, configurations

env = Conf()

# General
version = '2.2'
ARENA_NAME = env('ARENA_NAME', socket.gethostname(), group='General', desc='The name of the arena to be saved in the database')
IS_ANALYSIS_ONLY = env.bool('IS_ANALYSIS_ONLY', False, group='General', desc='Mode for development. If set, the arena-manager and periphery are not initialized')
LOGGING_LEVEL = env('LOGGING_LEVEL', 'DEBUG', group='General', desc='The logging level of the system log (DEBUG,INFO,WARNING,etc.)')
UI_LOGGING_LEVEL = env('UI_LOGGING_LEVEL', 'INFO', group='General', desc='The logging level of the management console')
OUTPUT_DIR = env('OUTPUT_DIR', (Path(__file__).parent.parent.resolve() / 'output').as_posix(), group='General', desc='The output directory')

# API
STATIC_FILES_DIR = env('STATIC_FILES_DIR', 'static', group='General', desc='Path to media files directory', is_map=False)
MANAGEMENT_PORT = env.int('MANAGEMENT_PORT', 5084, group='General', desc='Flask API port')
MANAGEMENT_HOST = env('MANAGEMENT_HOST', 'localhost', group='General', desc='Flask API host')
MANAGEMENT_URL = f'http://{MANAGEMENT_HOST}:{MANAGEMENT_PORT}'
POGONA_HUNTER_PORT = env.int('POGONA_HUNTER_PORT', 8080, group='General', desc='Port of the application')
DISABLE_ARENA_SCREEN = env.bool('DISABLE_ARENA_SCREEN', False, group='General', desc='Work without application screen')
api_max_blocks_to_show = 20
IS_GPU = env.bool('IS_GPU', True, group='General', desc='Whether the system has a CUDA GPU')

# Output Folders
CAPTURE_IMAGES_DIR = env('CAPTURE_IMAGES_DIR', f'{OUTPUT_DIR}/captures', is_map=False)
RECORDINGS_OUTPUT_DIR = env('RECORDINGS_OUTPUT_DIR', f'{OUTPUT_DIR}/recordings', is_map=False)
CLIBRATION_DIR = env('CLIBRATION_DIR', f'{OUTPUT_DIR}/calibrations', is_map=False)
EXPERIMENTS_DIR = env('EXPERIMENTS_DIR', f"{OUTPUT_DIR}/experiments", is_map=False)

# Application
TOUCH_SCREEN_NAME = env('TOUCH_SCREEN_NAME', 'Elo', group='Application', desc='Name of the touch screen to be searched in xinput')
DISABLE_APP_SCREEN = env.bool('DISABLE_APP_SCREEN', False, group='Application', desc='whether to disable the application screen')
IS_SCREEN_INVERTED = env.bool('IS_SCREEN_INVERTED', False, group='Application', desc='whether to invert the application screen')
IS_CHECK_SCREEN_MAPPING = env.bool('IS_CHECK_SCREEN_MAPPING', True, group='Application', desc='whether to check the mapping of the touch screen before each experiment')
APP_SCREEN = env('APP_SCREEN', ':0', group='Application', desc='Application screen address')
TEST_SCREEN = env('TEST_SCREEN', ':1.0', group='Application', desc='Screen address to be used for running test experiments')
SCREEN_RESOLUTION = env('SCREEN_RESOLUTION', '1920,1080', group='Application', desc='Resolution of the screen. Write with comma and no spaces.')  # must be written with comma
SCREEN_DISPLACEMENT = env('SCREEN_DISPLACEMENT', '000', group='Application', desc='Screen displacement. Used if more than one screen in connected to the server')  # used for displacing the screen contents in multi screen setup

# Cache (Redis)
IS_USE_REDIS = env.bool('IS_USE_REDIS', True, group='Cache', desc='whether to use redis as cache')
REDIS_HOST = env('REDIS_HOST', 'localhost', group='Cache', desc='Host of the redis server')
REDIS_PORT = env.int('REDIS_PORT', 6379, group='Cache', desc='Port of the redis server')
WEBSOCKET_URL = env('WEBSOCKET_URL', 'ws://localhost:6380', group='Cache', desc='URL of the websocket server that connects the management UI with the BackEnd')
ui_console_channel = "cmd/visual_app/console"
# listeners that should listen only during an experiment
experiment_metrics = {
    'touch': {
        'is_write_csv': True,
        'is_write_db': True,
        'csv_file': 'screen_touches.csv',
        'is_overall_experiment': False
    },
    'trial_data': {
        'is_write_csv': True,
        'is_write_db': True,
        'csv_file': {'bug_trajectory': 'bug_trajectory.csv', 
                     'video_frames': 'video_frames.csv',
                     'app_events': 'app_events.csv',
                     'trials_data': 'trials_data.csv'},
        'is_overall_experiment': False
    }
}
# listeners that should listen as long as the arena_mgr is alive
commands_topics = {
    'reward': 'cmd/arena/reward',
    'led_light': 'cmd/arena/led_light',
    'heat_light': 'cmd/arena/heat_light',

    'arena_shutdown': 'cmd/management/arena_shutdown',
    'end_experiment': 'cmd/management/end_experiment',

    'init_bugs': 'cmd/visual_app/init_bugs',
    'init_media': 'cmd/visual_app/init_media',
    'hide_bugs': 'cmd/visual_app/hide_bugs',
    'hide_media': 'cmd/visual_app/hide_media',
    'reload_app': 'cmd/visual_app/reload_app',
    'app_healthcheck': 'cmd/visual_app/healthcheck',
    'strike_predicted': 'cmd/visual_app/strike_predicted'
}
subscription_topics = {
    'arena_operations': 'cmd/arena/*',
    'metrics_logger': 'log/metric/*',
    'temperature': 'log/metric/temperature'
}
metric_channel_prefix = 'log/metric'
subscription_topics.update({k: f'{metric_channel_prefix}/{k}' for k in experiment_metrics.keys()})
subscription_topics.update(commands_topics)

# Arena
arena_modules = {
    'cameras': {
        'allied_vision': ('cameras.allied_vision', 'AlliedVisionCamera'),
        'flir': ('cameras.flir', 'FLIRCamera')
    },
    'image_handlers': {
        'pogona_head': ('image_handlers.predictor_handlers', 'PogonaHeadHandler'),
        'tongue_out': ('image_handlers.predictor_handlers', 'TongueOutHandler')
    },
    'predictors': {
        'deeplabcut': ('analysis.predictors.deeplabcut', 'DLCPose'),
        'tongue_out': ('analysis.predictors.tongue_out', 'TongueOutAnalyzer'),
        'pogona_head': ('analysis.predictors.pogona_head', 'PogonaHead')
    }
}
predictors_map = {
    'DLCPose': 'analysis.predictors.deeplabcut',
    'TongueOutAnalyzer': 'analysis.predictors.tongue_out',
    'PogonaHead': 'analysis.predictors.pogona_head'
}

# Cameras
DISABLE_CAMERAS_CHECK = env.bool('DISABLE_CAMERAS_CHECK', False, group='Cameras', desc='Disable check of cameras by the scheduler. Manual mode - scheduler does not turn on/off cameras')
output_dir_key = 'output_dir'  # used for cam_config
QUEUE_WAIT_TIME = env.int('QUEUE_WAIT_TIME', 2, group='Cameras', desc='time in seconds to wait for frame in camera queue before exception is raised')
SINK_QUEUE_TIMEOUT = env.int('SINK_QUEUE_TIMEOUT', 2, group='Cameras', desc='time in seconds to wait for frame in camera sink')
VIDEO_WRITER_FORMAT = env('VIDEO_WRITER_FORMAT', 'MJPG', group='Cameras', desc='format of ouput videos')
DEFAULT_EXPOSURE = env.int('DEFAULT_EXPOSURE', 5000, group='Cameras', desc='Default exposure for cameras')
IS_TRACKING_CAMERAS_ALLOWED = env.bool('IS_TRACKING_CAMERAS_ALLOWED', False, group='Cameras', desc='Allow cameras that are configured for tracking to create tracking videos')
MAX_VIDEO_TIME_SEC = env.int('MAX_VIDEO_TIME_SEC', 60 * 10, group='Cameras', desc='Maximum duration for output videos in seconds')
CAMERA_ON_MIN_DURATION = env.float('CAMERA_ON_MIN_DURATION', 10*60, group='Cameras', desc="used by the scheduler to set the duration (seconds) in which 'manual' \
                                   mode cameras would stay on, or for any type of camera this would set the duration it stays on outside of camera_on time.")
FRAMES_TIMESTAMPS_DIR = env('FRAMES_TIMESTAMPS_DIR', 'frames_timestamps', group='Cameras', is_map=False, desc='Name of the folder to be created in the experiment output directory to store all frames timestamps files')
ARRAY_QUEUE_SIZE_MB = env.int('ARRAY_QUEUE_SIZE_MB', 5 * 20, group='Cameras', desc='Queue size in MB for cameras')  # I assume that one image is roughly 5Mb
COUNT_TIMESTAMPS_FOR_FPS_CALC = env.int('COUNT_TIMESTAMPS_FOR_FPS_CALC', 200, group='Cameras', desc='how many timestamps to gather for calculating FPS')
WRITING_VIDEO_QUEUE_MAXSIZE = env.int('WRITING_VIDEO_QUEUE_MAXSIZE', 100, group='Cameras', desc='Max frames in the writing video queue')
shm_buffer_dtype = 'uint8'

# Periphery
DISABLE_PERIPHERY = env.bool('DISABLE_PERIPHERY', False, group='Periphery', desc='Disable all periphery integration')
mqtt = {
    'host': env('MQTT_HOST', 'localhost', group='Periphery', desc='Host of the MQTT server'),
    'port': env.int('MQTT_PORT', 1883, group='Periphery', desc='Port of the MQTT server'),
    'publish_topic': 'arena_command',
    'temperature_sensors': env.list('TEMPERATURE_SENSORS', ['Temp'], group='Periphery', desc='Names of the temperature sensors')
}
IR_LIGHT_NAME = env('IR_LIGHT_NAME', '', group='Periphery', desc='Name of infrared light in periphery config')
DAY_LIGHT_NAME = env('DAY_LIGHT_NAME', '', group='Periphery', desc='Name of LED lights in periphery config')
CAM_TRIGGER_ARDUINO_NAME = env('CAM_TRIGGER_ARDUINO_NAME', 'camera trigger', group='Periphery', desc='name of the camera trigger arduino in the periphery config')
ARENA_ARDUINO_NAME = env('ARENA_ARDUINO_NAME', 'arena', group='Periphery', desc='name of the arena arduino in the periphery config')

# Calibration
MIN_CALIBRATION_IMAGES = env.int('MIN_CALIBRATION_IMAGES', 7, group='Calibration', desc='Nuber of minimum calibration images per camera')
CHESSBOARD_DIM = env.list('CHESSBOARD_DIM', [9, 6], group='Calibration', desc='Dimensions of calibration chessboard', subcast=int)
ARUCO_MARKER_SIZE = env.float('ARUCO_MARKER_SIZE', 2.25, group='Calibration', desc='Size of single aruco marker in centimeters')
NUM_ARUCO_MARKERS = env.int('NUM_ARUCO_MARKERS', 150, group='Calibration', desc='Number of Charuco markers on the board')

# Scheduler
DISABLE_SCHEDULER = env.bool('DISABLE_SCHEDULER', False, group='Scheduler', desc='Disable the scheduler')
SCHEDULER_DATE_FORMAT = env('SCHEDULER_DATE_FORMAT', "%d/%m/%Y %H:%M", group='Scheduler', desc='Scheduler date format or how dates are shown in the schedules panel')
MAX_COMPRESSION_THREADS = env.int('MAX_COMPRESSION_THREADS', 2, group='Scheduler', desc='Number of threads to be used for video compression')
SCHEDULE_EXPERIMENTS_END_TIME = env('SCHEDULE_EXPERIMENTS_END_TIME', '19:00', group='Scheduler', desc='last hour of a day to schedule experiments')
IS_AGENT_ENABLED = env.bool('IS_AGENT_ENABLED', 0, group='Scheduler', desc='Enable the agent to schedule experiments automatically')
AGENT_MIN_DURATION_BETWEEN_PUBLISH = env.int('AGENT_MIN_DURATION_BETWEEN_PUBLISH', 2 * 60 * 60, group='Scheduler', desc='Minimum duration in seconds between publish meassages by the agent')
CAMERAS_ON_TIME = env('CAMERAS_ON_TIME', '07:00', group='Scheduler', desc='Time to start cameras by the scheduler', validator='hour_validator')
CAMERAS_OFF_TIME = env('CAMERAS_OFF_TIME', '19:00', group='Scheduler', desc='Time to stop cameras by the scheduler', validator='hour_validator')
POSE_ON_TIME = env('POSE_ON_TIME', '19:30', group='Scheduler', desc='Time to start nighly pose by the scheduler', validator='hour_validator')
POSE_OFF_TIME = env('POSE_OFF_TIME', '03:00', group='Scheduler', desc='Time to stop nighly pose on the day after by the scheduler', validator='hour_validator')
TRACKING_POSE_ON_TIME = env('TRACKING_POSE_ON_TIME', '03:00', group='Scheduler', desc='Time to start nighly pose analysis on tracking videos by the scheduler', validator='hour_validator')
TRACKING_POSE_OFF_TIME = env('TRACKING_POSE_OFF_TIME', '06:00', group='Scheduler', desc='Time to stop nighly pose analysis on the day after on tracking videos by the scheduler', validator='hour_validator')
LIGHTS_SUNRISE = env('LIGHTS_SUNRISE', '07:00', group='Scheduler', desc='Time to turn on LED lights by the scheduler', validator='hour_validator')
LIGHTS_SUNSET = env('LIGHTS_SUNSET', '19:00', group='Scheduler', desc='Time to turn off LED lights by the scheduler', validator='hour_validator')
DWH_COMMIT_TIME = env('DWH_COMMIT_TIME', '07:00', group='Scheduler', desc='Time of the day to run the commit to data warehouse', validator='hour_validator')
STRIKE_ANALYSIS_TIME = env('STRIKE_ANALYSIS_TIME', '06:30', group='Scheduler', desc='Time of the day to run the strike analysis', validator='hour_validator')
DAILY_SUMMARY_TIME = env('DAILY_SUMMARY_TIME', '20:00', group='Scheduler', desc='Time of the day to send the daily summary in telegram', validator='hour_validator')

# Experiments
CAM_TRIGGER_DELAY_AROUND_BLOCK = env.int('CAM_TRIGGER_DELAY_AROUND_BLOCK', 8, group='Experiments', desc='The trigger delay in seconds before and after a block in an experiment. If 0, no delay is used')
IR_TOGGLE_DELAY_AROUND_BLOCK = env.int('IR_TOGGLE_DELAY_AROUND_BLOCK', 1, group='Experiments', desc='The IR toggle delay in seconds before and after a block in an experiment. If 0, no delay is used')
IS_RECORD_SCREEN_IN_EXPERIMENT = env.bool('IS_RECORD_SCREEN_IN_EXPERIMENT', False, group='Experiments', desc='Whether to record the screen in the experiment. Notice it has high CPU usage!')
EXTRA_TIME_RECORDING = env.int('EXTRA_TIME_RECORDING', 30, group='Experiments', desc='Extra time in seconds before and after the experiment in which no trials are on and only the cameras record')
TIME_BETWEEN_BLOCKS = env.int('TIME_BETWEEN_BLOCKS', 300, group='Experiments', desc='Time in seconds between blocks in an experiment')
EXPERIMENTS_TIMEOUT = env.int('EXPERIMENTS_TIMEOUT', 60 * 60, group='Experiments', desc='Timeout in seconds used by the cache for maximum experiment time')
REWARD_TIMEOUT = env.int('REWARD_TIMEOUT', 10, group='Experiments', desc='Time in seconds to wait between rewards')
MAX_DAILY_REWARD = env.int('MAX_DAILY_REWARD', 40, group='Experiments', desc='Max number of rewards per day')
MAX_DURATION_CONT_BLANK = env.int('MAX_DURATION_CONT_BLANK', 48*3600, group='Experiments', desc='Max duration in seconds of a blank continuous experiment')
CHECK_ENGAGEMENT_HOURS = env.int('CHECK_ENGAGEMENT_SPAN', 0, group='Experiments', desc='Hours before to check engagement or whether there were any strikes. If there are no strikes in this time span, give reward. Setting 0 will disable this check.')
RANDOM_LOW_HORIZONTAL_MAX_STRIKES = env.int('RANDOM_LOW_HORIZONTAL_MAX_STRIKES', 30, group='Experiments', desc='Max number of strikes per bug speed in random_low_horizontal movement')
CACHED_EXPERIMENTS_DIR = env('CACHED_EXPERIMENTS_DIR', 'cached_experiments', group='Experiments', desc='Folder name in the main Arena folder to store saved experiments')
experiment_types = {
    'bugs': ['reward_type', 'bug_types', 'reward_bugs', 'bug_speed', 'movement_type', 'time_between_bugs',
             'is_anticlockwise' 'target_drift', 'background_color', 'exit_hole_position'],
    'media': ['media_url'],
    'blank': ['blank_rec_type'],
    'psycho': ['psycho_file']
}
blank_rec_types = [
    'trials',
    'continuous'
]
reward_types = [
    'always',
    'end_trial'
]

# Database
DISABLE_DB = env.bool('DISABLE_DB', False, group='Database', desc='Disable usage of database by the system')
db_name = env('DB_NAME', 'arena', group='Database', desc='Database name')
db_host = env('DB_HOST', 'localhost', group='Database', desc='Database host')
db_port = env.int('DB_PORT', 5432, group='Database', desc='Database port')
db_engine = env('DB_ENGINE', 'postgresql+psycopg2', group='Database', desc='Database engine')
db_user = env('DB_USER', 'postgres', group='Database', desc='Database user')
db_password = env('DB_PASSWORD', 'password', group='Database', desc='Database password')
sqlalchemy_url = f'{db_engine}://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
DWH_N_TRIES = env.int('DWH_N_TRIES', 5, group='Database', desc='Number of tries to commit to data warehouse')
DWH_HOST = env('DWH_HOST', None, group='Database', desc='Data-Warehouse host. Must be configured for DWH operations to work')
DWH_URL = f'{db_engine}://{db_user}:{db_password}@{DWH_HOST}:{db_port}/{db_name}'

# Publishers
TELEGRAM_CHAT_ID = env('TELEGRAM_CHAT_ID', '725002866', group='Publishers', desc='Telegram Chat ID')
TELEGRAM_TOKEN = env('TELEGRAM_TOKEN', None, group='Publishers', desc='Token to use for telegram communication')
SENTRY_DSN = env('SENTRY_DSN', '', group='Publishers', desc='Data source name for the sentry service')

# Pose Estimation
NIGHT_POSE_CAMERA = env('NIGHT_POSE_CAMERA', '', group='Pose-Estimation', validator='cam_exist', desc='Which camera videos should the nighlty pose estimation take. Must set this and NIGHT_POSE_PREDICTOR to activate nightly pose estimation.')
NIGHT_POSE_PREDICTOR = env('NIGHT_POSE_PREDICTOR', '', group='Pose-Estimation', validator='predict_model_exist', desc='Name of Deeplabcut predictor from predict_config to use for nighly pose estimation')
NIGHT_POSE_RUN_ONLY_BUG_SESSIONS = env.bool('NIGHT_POSE_RUN_ONLY_BUG_SESSIONS', False, group='Pose-Estimation', desc='Whether to run nightly pose estimation only on bugs experiments')
IS_RUN_NIGHTLY_POSE_ESTIMATION = bool(NIGHT_POSE_CAMERA) and bool(NIGHT_POSE_PREDICTOR)
SCREEN_PIX_CM = env.float('SCREEN_PIX_CM', None, group='Pose-Estimation', desc='Number to multiply with the pixels values to convert to cm units')
SCREEN_START_X_CM = env.float('SCREEN_START_X_CM', None, group='Pose-Estimation', desc='Start X position in cm of the screen inside the arena')
SCREEN_Y_CM = env.float('SCREEN_Y_CM', None, group='Pose-Estimation', desc='Location of the screen within the arena, along the Y axis [cm]')
IS_SCREEN_CONFIGURED_FOR_POSE = (SCREEN_PIX_CM is not None) and (SCREEN_START_X_CM is not None)

# PsychoPy
PSYCHO_FOLDER = env('PSYCHO_FOLDER', None, group='PsychoPy', desc='Path to folder with psychopy files')
PSYCHO_PYTHON_INTERPRETER = env('PSYCHO_PYTHON_INTERPRETER', None, group='PsychoPy', desc='Full path to the python interpreter that will be used to run psychopy')

cameras = load_configuration('cameras')