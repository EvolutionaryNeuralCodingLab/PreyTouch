import io
import time
import sys
import cv2
import json
import warnings
import base64
import psutil
import logging
import pytest
import importlib
import tempfile
from pathlib import Path
from PIL import Image
from datetime import datetime
import numpy as np
import pandas as pd
from io import BytesIO
import torch.multiprocessing as mp
from flask import Flask, render_template, Response, request, send_from_directory, jsonify, send_file
import sentry_sdk
import config
import utils
from cache import RedisCache, CacheColumns as cc
from utils import titlize, turn_display_on, turn_display_off, get_sys_metrics, get_psycho_files
from experiment import ExperimentCache
from arena import ArenaManager
from loggers import init_logger_config, create_arena_handler
from calibration import CharucoEstimator, Calibrator
from periphery_integration import PeripheryIntegrator
from agent import Agent
from analysis.pose import run_predict
import matplotlib
matplotlib.use('Agg')

app = Flask('ArenaAPI')
cache: RedisCache = None
arena_mgr: ArenaManager = None
periphery_mgr: PeripheryIntegrator = None
queue_app: mp.Queue = None


@app.route('/')
def index():
    """Video streaming ."""
    cached_experiments = sorted([c.stem for c in Path(config.CACHED_EXPERIMENTS_DIR).glob('*.json')])
    with open('../pogona_hunter/src/config.json', 'r') as f:
        app_config = json.load(f)
    if arena_mgr is None:
        cameras = list(config.cameras.keys())
    else:
        cameras = list(arena_mgr.units.keys())

    if config.IS_ANALYSIS_ONLY:
        toggels, feeders, cameras = [], [], []
    else:
        toggels, feeders = periphery_mgr.toggles, periphery_mgr.feeders

    confs = {k: json.dumps(config.load_configuration(k)) for k in config.configurations.keys()}
    predictors = list(config.load_configuration('predict').keys())
    return render_template('index.html', cameras=cameras, exposure=config.DEFAULT_EXPOSURE, arena_name=config.ARENA_NAME,
                           config=app_config, log_channel=config.ui_console_channel, reward_types=config.reward_types,
                           experiment_types=config.experiment_types, media_files=list_media(), min_calib_images=config.MIN_CALIBRATION_IMAGES,
                           blank_rec_types=config.blank_rec_types, config_envs=config.env.get_all_from_cache(), predictors=predictors,
                           max_blocks=config.api_max_blocks_to_show, toggels=toggels, psycho_files=get_psycho_files(),
                           extra_time_recording=config.EXTRA_TIME_RECORDING, feeders=feeders, configurations=confs,
                           acquire_stop={'num_frames': 'Num Frames', 'rec_time': 'Record Time [sec]'})


@app.route('/check', methods=['GET'])
def check():
    # periphery_mgr.publish_cam_trigger_state()
    res = dict()
    res['experiment_name'] = cache.get_current_experiment()
    res['block_id'] = cache.get(cc.EXPERIMENT_BLOCK_ID)
    res['open_app_host'] = cache.get(cc.OPEN_APP_HOST)
    res['temperature'] = json.loads(cache.get(cc.TEMPERATURE) or "{}")
    res['cached_experiments'] = sorted([c.stem for c in Path(config.CACHED_EXPERIMENTS_DIR).glob('*.json')])
    res['cam_trigger_state'] = cache.get(cc.CAM_TRIGGER_STATE)

    if config.IS_ANALYSIS_ONLY:
        res.update({'reward_left': 0, 'schedules': {}})
    else:
        res['reward_left'] = periphery_mgr.get_feeders_counts()
        res['streaming_camera'] = arena_mgr.get_streaming_camera()
        res['schedules'] = arena_mgr.schedules
        for cam_name, cu in arena_mgr.units.copy().items():
            res.setdefault('cam_units_status', {})[cam_name] = cu.is_on()
            res.setdefault('cam_units_fps', {})[cam_name] = {k: cu.mp_metadata.get(k).value for k in ['cam_fps', 'sink_fps', 'pred_fps', 'pred_delay']}
            res.setdefault('cam_units_predictors', {})[cam_name] = ','.join(cu.get_alive_predictors()) or '-'
            proc_cpus = {}
            for p in cu.processes.copy().values():
                try:
                    proc_cpus[p.name] = round(psutil.Process(p.pid).cpu_percent(0.1))
                except:
                    continue
            res.setdefault('processes_cpu', {})[cam_name] = proc_cpus

    res.update(get_sys_metrics())
    return jsonify(res)


@app.route('/record', methods=['POST'])
def record_video():
    """Record video"""
    if request.method == 'POST':
        data = request.json
        return Response(arena_mgr.record(**data))


@app.route('/start_experiment', methods=['POST'])
def start_experiment():
    """Set Experiment Name"""
    data = request.json
    print(data)
    e = arena_mgr.start_experiment(**data)
    return Response(e)


@app.route('/save_experiment', methods=['POST'])
def save_experiment():
    """Set Experiment Name"""
    data = request.json
    ExperimentCache().save(data)
    return Response('ok')


@app.route('/load_experiment/<name>')
def load_experiment(name):
    """Load Cached Experiment"""
    data = ExperimentCache().load(name)
    return jsonify(data)


@app.route('/get_experiment')
def get_experiment():
    return Response(cache.get_current_experiment())


@app.route('/stop_experiment')
def stop_experiment():
    experiment_name = cache.get_current_experiment()
    if experiment_name:
        cache.stop_experiment()
        return Response(f'ending experiment {experiment_name}...')
    return Response('No available experiment')


@app.route('/commit_schedule', methods=['POST'])
def commit_schedule():
    data = dict(request.form)
    if not data.get('start_date'):
        arena_mgr.logger.error('please enter start_date for schedule')
    else:
        data['start_date'] = datetime.strptime(data['start_date'], '%d/%m/%Y %H:%M')
        if data.get('end_date'):
            data['end_date'] = datetime.strptime(data['end_date'], '%d/%m/%Y %H:%M')
        data['every'] = int(data['every'])
        arena_mgr.orm.commit_multiple_schedules(**data)
        arena_mgr.update_upcoming_schedules()
    return Response('ok')


@app.route('/delete_schedule', methods=['POST'])
def delete_schedule():
    arena_mgr.orm.delete_schedule(request.form['schedule_id'])
    arena_mgr.update_upcoming_schedules()
    return Response('ok')


@app.route('/update_reward_count', methods=['POST'])
def update_reward_count():
    data = request.json
    feeder_name = data.get('name')
    reward_count = int(data.get('reward_count', 0))
    arena_mgr.logger.info(f'Update {feeder_name} to {reward_count}')
    periphery_mgr.update_reward_count(feeder_name, reward_count)
    return Response('ok')


@app.route('/update_animal_id', methods=['POST'])
def update_animal_id():
    data = request.json
    animal_id = data['animal_id']
    current_animal_id = cache.get(cc.CURRENT_ANIMAL_ID)
    if animal_id != current_animal_id:
        if current_animal_id:
            arena_mgr.orm.update_animal_id(end_time=datetime.now())
        if animal_id:
            arena_mgr.orm.commit_animal_id(**data)
            arena_mgr.logger.info(f'Animal ID was updated to {animal_id} ({data["sex"]})')
    else:
        arena_mgr.orm.update_animal_id(**data)
    return Response('ok')


@app.route('/get_current_animal', methods=['GET'])
def get_current_animal():
    if config.DISABLE_DB or config.IS_ANALYSIS_ONLY:
        return jsonify({})
    animal_id = cache.get(cc.CURRENT_ANIMAL_ID)
    if not animal_id:
        arena_mgr.logger.warning('No animal ID is set')
        return jsonify({})
    animal_dict = arena_mgr.orm.get_animal_settings(animal_id)
    return jsonify(animal_dict)


@app.route('/animal_today_summary', methods=['GET'])
def animal_today_summary():
    if config.DISABLE_DB or config.IS_ANALYSIS_ONLY:
        return Response('Unable to load animal summary since DB is disabled')
    try:
        animal_id = cache.get(cc.CURRENT_ANIMAL_ID)
        strike_df = arena_mgr.orm.get_today_strikes()
        rewards_counts = arena_mgr.orm.get_today_rewards()
        text = f'Animal ID: {animal_id}\n'
        text += f'Total Strikes Today: {len(strike_df)}\n'
        text += f'Today Rewards: {rewards_counts["auto"]} (manual: {rewards_counts["manual"]})\n\n'
        if not strike_df.empty:
            text += f'Today Strikes:\n\n{strike_df.to_string(index=False)}\n\n'
        if config.IS_AGENT_ENABLED:
            ag = Agent()
            ag.update()
            text += f'Agent Summary:\n{ag.get_animal_history()}'
        return Response(text)
    except Exception as e:
        arena_mgr.logger.exception(e)
        return Response('Error loading animal summary', status=500)


@app.route('/start_camera_unit', methods=['POST'])
def start_camera_unit():
    cam_name = request.form['camera']
    if cam_name not in arena_mgr.units:
        app.logger.error(f'cannot start camera unit {cam_name} - unknown')
        return Response('')

    arena_mgr.units[cam_name].start()
    return Response('ok')


@app.route('/stop_camera_unit', methods=['POST'])
def stop_camera_unit():
    cam_name = request.form['camera']
    if cam_name not in arena_mgr.units:
        app.logger.error(f'cannot start camera unit {cam_name} - unknown')
        return Response('')

    arena_mgr.units[cam_name].stop()
    if cam_name == arena_mgr.get_streaming_camera():
        arena_mgr.stop_stream()
    return Response('ok')


@app.route('/set_cam_trigger', methods=['POST'])
def set_cam_trigger():
    if cache.get(cc.CAM_TRIGGER_DISABLE):
        # during experiments the trigger gui is disabled
        return Response('ok')
    state = int(request.form['state'])
    periphery_mgr.cam_trigger(state)
    return Response('ok')


@app.route('/update_trigger_fps', methods=['POST'])
def update_trigger_fps():
    data = request.json
    periphery_mgr.change_trigger_fps(data['fps'])
    return Response('ok')


@app.route('/update_arena_config', methods=['POST'])
def update_arena_config():
    data = request.json
    try:
        config.env.update_from_api(data['key'], data['value'])
    except Exception as exc:
        error_msg = f'Error in update_arena_config; {exc}'
        arena_mgr.logger.error(error_msg)
        return Response(error_msg, status=400)
    
    return Response(f'updated {data["key"]} to {data["value"]}', status=200)


@app.route('/save_config/<name>', methods=['POST'])
def save_config(name):
    data = request.json

    try:
        conf_p, (test_module, test_class) = config.configurations[name]
        test_module = importlib.import_module(test_module)
        test_class = getattr(test_module, test_class)

        test_class().run_all(data)

        with Path(conf_p).open('w') as f:
            json.dump(data, f, indent=4)

    except Exception as exc:
        error_msg = f'Error in save_config; {exc}'
        arena_mgr.logger.error(error_msg)
        return Response(error_msg, status=400)

    return Response(f'Saved {name} config')


@app.route('/capture', methods=['POST'])
def capture():
    cam = request.form['camera']
    folder_prefix = request.form.get('folder_prefix')
    img = arena_mgr.get_frame(cam)
    dir_path = config.CAPTURE_IMAGES_DIR
    Path(dir_path).mkdir(exist_ok=True, parents=True)
    if folder_prefix:
        dir_path = Path(dir_path) / folder_prefix
        dir_path.mkdir(exist_ok=True, parents=True)
    img_path = f'{dir_path}/{utils.datetime_string()}_{cam}.png'
    cv2.imwrite(img_path, img)
    arena_mgr.logger.info(f'Image from {cam} was saved to: {img_path}')
    return Response('ok')


@app.route('/run_calibration/<mode>', methods=['POST'])
def run_calibration(mode):
    assert mode in ['undistort', 'realworld'], f'mode must be "undistort" or "realworld", received: {mode}'
    data = request.json
    calib = Calibrator(data['cam_name']) if mode == 'undistort' else CharucoEstimator(data['cam_name'])
    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            for img_name, img in data['images'].items():
                imgdata = img.split(',')[1]
                img = base64.b64decode(imgdata)
                img = np.array(Image.open(BytesIO(img)))
                cv2.imwrite(f'{tmpdirname}/{img_name}', img)
            
            calib_date = datetime.strptime(data['date'], '%Y-%m-%d')
            if mode == 'undistort':
                err_text = calib.calibrate_camera(img_dir=tmpdirname, calib_date=calib_date)
                img = cv2.imread(calib.calib_results_image_path)
            else:
                img, err_text = calib.find_aruco_markers(f'{tmpdirname}/{list(data["images"].keys())[0]}')
            
            img = Image.fromarray(img)
            rawBytes = io.BytesIO()
            img.save(rawBytes, "JPEG")
            rawBytes.seek(0)
            img_base64 = base64.b64encode(rawBytes.read())
            return jsonify({'res': str(img_base64), 'err_text': err_text})

    except Exception as exc:
        return Response(f'Error in calibration; {exc}', status=500)


@app.route('/run_predictions', methods=['POST'])
def run_predictions():
    data = request.json
    try:
        assert data['pred_name'] in config.load_configuration('predict'), f'Unknown predictor: {data["pred_name"]}'
        image_date = data.get('image_date')
        if image_date:
            image_date = datetime.strptime(image_date, '%Y-%m-%d')
        res = run_predict(data['pred_name'], list(data['images'].values()), data.get('cam_name', ''), image_date, is_base64=True)
        if 'images' not in res or len(res['images']) == 0:
            return Response(f'Prediction returned nothing', status=500)
        
        img, pdf = res['images'][0]
        img = Image.fromarray(img)
        rawBytes = io.BytesIO()
        img.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        img_base64 = base64.b64encode(rawBytes.read())
        return jsonify({'image': str(img_base64), 'result': pdf.to_string() if isinstance(pdf, (pd.Series, pd.DataFrame)) is not None else 'No prediction'})
    except ImportError as exc:
        return Response(f'Error in prediction; {exc}', status=500)
    

@app.route('/reward')
def reward():
    """Activate Feeder"""
    # cache.publish_command('reward')
    periphery_mgr.feed(is_manual=True)
    return Response('ok')


@app.route('/arena_switch/<name>/<state>')
def arena_switch(name, state):
    state = int(state)
    assert state in [0, 1], f'state must be 0 or 1; received {state}'
    arena_mgr.logger.debug(f'Turn {name} {"on" if state == 1 else "off"}')
    periphery_mgr.switch(name, state)
    return Response('ok')


@app.route('/display/<state>')
def display(state):
    if state == 'off':
        stdout = turn_display_off(logger=arena_mgr.logger)
    else:
        stdout = turn_display_on(logger=arena_mgr.logger)
    return Response(stdout)


@app.route('/cameras_info')
def cameras_info():
    """Get cameras info"""
    return Response(arena_mgr.display_info(return_string=True))


@app.route('/check_cameras')
def check_cameras():
    """Check all cameras are connected"""
    if config.DISABLE_CAMERAS_CHECK:
        return Response(json.dumps([]))
    info_df = arena_mgr.display_info()
    missing_cameras = []
    for cam_name, cam_config in config.cameras.items():
        if cam_name not in info_df.index:
            missing_cameras.append(cam_name)

    return Response(json.dumps(missing_cameras))


@app.route('/get_camera_settings/<name>')
def get_camera_settings(name):
    return jsonify(arena_mgr.units[name].cam_config)


@app.route('/update_camera/<name>', methods=['POST'])
def update_camera_settings(name):
    data = request.json
    arena_mgr.update_camera_unit(name, data)
    return Response('ok')


@app.route('/manual_record_stop')
def manual_record_stop():
    arena_mgr.stop_recording()
    return Response('Record stopped')


@app.route('/reload_app')
def reload_app():
    cache.publish_command('reload_app')


@app.route('/init_bugs', methods=['POST'])
def init_bugs():
    if request.method == 'POST':
        cache.publish_command('init_bugs', request.data.decode())
    return Response('ok')


@app.route('/hide_bugs')
def hide_bugs():
    cache.publish_command('hide_bugs', '')
    return Response('ok')


@app.route('/start_media', methods=['POST'])
def start_media():
    if request.method == 'POST':
        data = request.json
        if not data or not data.get('media_url'):
            return Response('Unable to find media url')
        payload = json.dumps({'url': f'{config.MANAGEMENT_URL}/media/{data["media_url"]}'})
        print(payload)
        cache.publish_command('init_media', payload)
    return Response('ok')


@app.route('/stop_media')
def stop_media():
    cache.publish_command('hide_media')
    return Response('ok')


def list_media():
    media_files = []
    for f in Path(config.STATIC_FILES_DIR).glob('*'):
        if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.avi', '.mp4', '.mpg', '.mov']:
            media_files.append(f.name)
    return media_files


@app.route('/media/<filename>')
def send_media(filename):
    return send_from_directory(config.STATIC_FILES_DIR, filename)


@app.route('/set_stream_camera', methods=['POST'])
def set_stream_camera():
    if request.method == 'POST':
        arena_mgr.set_streaming_camera(request.form['camera'])
        return Response(request.form['camera'])


@app.route('/stop_stream_camera', methods=['POST'])
def stop_stream_camera():
    if request.method == 'POST':
        arena_mgr.stop_stream()
        return Response('ok')


@app.route('/cam_scan', methods=['GET'])
def cam_scan():
    report = ''
    try:
        report += 'Allied Vision Cameras Scan:\n'
        from cameras.allied_vision import scan_cameras
        df = scan_cameras()
        res = df.reset_index()[['index','Camera ID']].values.tolist()
        for cam_name, cam_id in res:
            report += f'- {cam_name}: {cam_id}\n'
    except Exception as exc:
        report += f'Error scanning allied_vision: {exc}\n'
    report += '\n'
    try:
        report += 'FLIR Cameras Scan:\n'
        from cameras.flir import scan_cameras
        df = scan_cameras()
        res = df.reset_index()[['index', 'DeviceID']].values.tolist()
        for cam_name, cam_id in res:
            report += f'- {cam_name}: {cam_id}\n'
    except Exception as exc:
        report += f'Error scanning flir: {exc}\n'
    return Response(report)


@app.route('/periphery_scan', methods=['GET'])
def periphery_scan():
    try:
        from serial.tools import list_ports
        ports = list_ports.comports()
        res = 'Serial Scan:\n'
        for p in ports:
            if p.serial_number is not None:
                res += f'- Device: {p.device}, SN: {p.serial_number} ({p.description})\n'
        return Response(res)
    except Exception as exc:
        return Response(f'Error scanning periphery: {exc}')


@app.route('/reboot_service/<service>', methods=['GET'])
def reboot_service(service):
    rc = os.system(f'cd ../docker && docker-compose restart {service}')
    if rc != 0:
        return Response(f'Error restarting service {service}', status=500)
    return Response('ok')


@app.route('/load_example_config/<conf_name>', methods=['GET'])
def load_example_config(conf_name):
    if conf_name not in config.configurations:
        return Response(f'Configuration {conf_name} not found', status=404)
    
    p = Path(f'configurations/examples/{conf_name}_example.json')
    if not p.exists():
        return Response(f'Example configuration file {p} not found', status=404)
    
    with p.open('r') as f:
        data = json.load(f)
    return Response(json.dumps(data), status=200)

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(arena_mgr.stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/strike_analysis/<strike_id>')
def get_strike_analysis(strike_id):
    from analysis.strikes import StrikeAnalyzer, Loader
    ld = Loader(strike_id, 'front', is_debug=False)
    sa = StrikeAnalyzer(ld)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = sa.plot_strike_analysis(only_return=True)
    img = Image.fromarray(img.astype('uint8'))
    # create file-object in memory
    file_object = io.BytesIO()
    # write PNG in file-object
    img.save(file_object, 'PNG')
    # move to beginning of file so `send_file()` it will read from start
    file_object.seek(0)
    return send_file(file_object, mimetype='image/PNG')


def initialize():
    logger = logging.getLogger(app.name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("""%(levelname)s in %(module)s [%(pathname)s:%(lineno)d]:\n%(message)s""")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


import os
import re


def get_chunk(filename, byte1=None, byte2=None):
    filesize = os.path.getsize(filename)
    yielded = 0
    yield_size = 1024 * 1024

    if byte1 is not None:
        if not byte2:
            byte2 = filesize
        yielded = byte1
        filesize = byte2

    with open(filename, 'rb') as f:
        content = f.read()

    while True:
        remaining = filesize - yielded
        if yielded == filesize:
            break
        if remaining >= yield_size:
            yield content[yielded:yielded+yield_size]
            yielded += yield_size
        else:
            yield content[yielded:yielded+remaining]
            yielded += remaining


@app.route('/play_video11')
def get_file():
    filename = '/data/Pogona_Pursuit/Arena/static/back_20221106T093511.mp4'
    filesize = os.path.getsize(filename)
    range_header = request.headers.get('Range', None)

    if range_header:
        byte1, byte2 = None, None
        match = re.search(r'(\d+)-(\d*)', range_header)
        groups = match.groups()

        if groups[0]:
            byte1 = int(groups[0])
        if groups[1]:
            byte2 = int(groups[1])

        if not byte2:
            byte2 = byte1 + 1024 * 1024
            if byte2 > filesize:
                byte2 = filesize

        length = byte2 + 1 - byte1

        resp = Response(
            get_chunk(filename, byte1, byte2),
            status=206, mimetype='video/mp4',
            content_type='video/mp4',
            direct_passthrough=True
        )

        resp.headers.add('Content-Range',
                         'bytes {0}-{1}/{2}'
                         .format(byte1,
                                 length,
                                 filesize))
        return resp

    return Response(
        get_chunk(filename),
        status=200, mimetype='video/mp4'
    )


@app.after_request
def after_request(response):
    response.headers.add('Accept-Ranges', 'bytes')
    return response


@app.route('/play_video')
def play():
    return render_template('management/play_video.html')


def restart_cmd():
    import signal
    import loggers
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # pid = os.fork()
    # if pid > 0:
    loggers.init_logger_config()
    logger = loggers.get_logger('restart')
    logger.info(f'starting restart in process')
    rc = os.system('nohup supervisorctl restart prey_touch')
    logger.info(f'finished restart with rc={rc}')


@app.route('/restart')
def restart():
    # from subprocess import Popen, PIPE

    arena_mgr.logger.info(f'start arena restart')
    # returncode = os.system('supervisorctl restart prey_touch &')
    # err = os.system('supervisorctl restart prey_touch')
    # CMD = ['nohup', 'supervisorctl', 'restart', 'prey_touch']
    # process = Popen(CMD, preexec_fn=os.setpgrp)
    # stdout, stderr = process.communicate()
    # os.spawnvpe(os.P_NOWAIT, CMD[0], CMD, os.environ)

    # p = mp.Process(target=restart_cmd, name='RESTART_ARENA')
    # p.daemon = True
    # p.start()

    arena_mgr.arena_shutdown()
    queue_app.put('restart')
    return Response('ok')


def start_app(queue):
    global cache, arena_mgr, periphery_mgr, queue_app
    queue_app = queue
    if pytest.main(['-x', 'tests', '-s', '--tb=line', '--color=yes']) != 0:
        queue_app.put('stop')
        return

    if config.IS_GPU:
        import torch
        torch.cuda.set_device(0)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    init_logger_config()
    arena_handler = create_arena_handler('API')
    app.logger.addHandler(arena_handler)
    app.logger.setLevel(logging.INFO)

    cache = RedisCache()
    if not config.IS_ANALYSIS_ONLY:
        arena_mgr = ArenaManager()
        periphery_mgr = PeripheryIntegrator()
        utils.turn_display_off(logger=arena_mgr.logger)
        if arena_mgr.is_cam_trigger_setup() and not config.DISABLE_PERIPHERY:
            periphery_mgr.cam_trigger(1)

    app.run(host='0.0.0.0', port=config.MANAGEMENT_PORT, debug=False)


if __name__ == "__main__":

    # app.logger.removeHandler(flask_logging.default_handler)
    # h = logging.StreamHandler(sys.stdout)
    # h.setLevel(logging.WARNING)
    # h.setFormatter(CustomFormatter())
    # werklogger = logging.getLogger('werkzeug')
    # werklogger.addHandler(h)
    # app.debug = False
    # logger = logging.getLogger(app.name)
    # h = logging.StreamHandler()
    # h.setFormatter(CustomFormatter())
    # logger.addHandler(h)
    if not config.IS_ANALYSIS_ONLY and config.SENTRY_DSN:
        sentry_sdk.init(
            dsn=config.SENTRY_DSN,
            # Set traces_sample_rate to 1.0 to capture 100%
            # of transactions for performance monitoring.
            # We recommend adjusting this value in production.
            traces_sample_rate=1.0
        )

    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    queue_app = mp.Queue()

    while True:
        p = mp.Process(target=start_app, args=(queue_app,), name='MAIN')
        p.start()
        while True:
            if queue_app.empty():
                time.sleep(1)
            else:
                x = queue_app.get()
                if x == 'stop':
                    sys.exit(1)
                break
        app.logger.warning('Restarting Arena!')
        p.terminate()

    app.run(host='0.0.0.0', port=config.MANAGEMENT_PORT, debug=False)

