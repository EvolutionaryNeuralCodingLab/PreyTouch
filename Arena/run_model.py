import argparse
import config
import sys
from pathlib import Path
import importlib
import subprocess
import torch
from datetime import datetime, timedelta, timezone

PROCESSING_TIMEOUT = timedelta(minutes=30)
DONE_TIMEOUT = timedelta(hours=24)


def print_models(pconf):
    print(f'Configured models:')
    for model_name in pconf.keys():
        print(f' - {model_name}')
def scan_path_for_videos(dir_path, cam_name, *, video_suffixes=None):
    if video_suffixes is None:
        video_suffixes = [".mp4"]
    allowed_suffixes = {s.lower() for s in video_suffixes}
    find_cmd = [
        'find', str(dir_path),
        '(', '-path', '*/predictions', '-o', '-path', '*/frames_timestamps', '-o', '-path', '*/trials_images', ')', '-prune',
        '-o', '-type', 'f', '(',
    ]
    for suffix in allowed_suffixes:
        find_cmd.extend(['-name', f'{cam_name}*{suffix}', '-o'])
    find_cmd.pop()
    find_cmd.extend([')', '-print'])
    try:
        res = subprocess.run(find_cmd, capture_output=True, text=True, check=False)
    except Exception:
        return []
    return [Path(x) for x in res.stdout.splitlines() if x]


def load_video_list(video_list_file, cam_name, *, video_suffixes=None):
    if video_suffixes is None:
        video_suffixes = [".mp4"]
    allowed_suffixes = {s.lower() for s in video_suffixes}
    video_paths = []
    for line in Path(video_list_file).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        video_path = Path(line)
        if video_path.name.startswith(cam_name) and video_path.suffix.lower() in allowed_suffixes:
            video_paths.append(video_path)
    return video_paths

def print_videos_found(vid_paths):
    print(f'Videos found:')
    for vid_path in vid_paths:
        print(f' - {vid_path}')
    print(f'\nTotal of {len(vid_paths)} videos.')


def load_predictor(pconf, model_name, cam_name):
    pred_item = pconf[model_name]
    prd_class = pred_item['predictor_name']
    if prd_class == 'DLCPose':
        from analysis.pose import DLCArenaPose
        return DLCArenaPose(cam_name, model_path=pred_item['model_path'],
                            is_use_db=False, is_raise_no_caliber=False)

    prd_module = config.predictors_map[prd_class]
    prd_module = importlib.import_module(prd_module)
    return getattr(prd_module, prd_class)(cam_name, pred_item['model_path'])

import traceback
def predict_video(prd, video_path):
    try:
        prd.predict_video(video_path=video_path)
    except Exception as e:
        print("An error occurred while predicting the video:")
        traceback.print_exc()
    finally:
        torch.cuda.empty_cache()




def is_recent_timestamp_file(path: Path, max_age: timedelta) -> bool:
    if not path.exists():
        return False
    try:
        ts_text = path.read_text().strip()
        ts = datetime.fromisoformat(ts_text)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - ts.astimezone(timezone.utc)
        return age < max_age
    except Exception:
        return False


def is_recent_mtime_file(path: Path, max_age: timedelta) -> bool:
    if not path.exists():
        return False
    try:
        age = datetime.now(timezone.utc) - datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        return age < max_age
    except Exception:
        return False


def should_skip_processing(cache_path: Path, *, skip_existing: bool = True) -> bool:
    flag_path = cache_path.with_suffix(".processing")
    done_path = cache_path.with_suffix(".done")
    resume_path = cache_path.with_suffix(".resume")
    resume_tmp_path = resume_path.parent / f'{resume_path.name}.tmp'

    if is_recent_timestamp_file(flag_path, PROCESSING_TIMEOUT):
        print(f"Skipping (processing <30m): {flag_path}")
        return True

    for path in [resume_tmp_path, resume_path]:
        if is_recent_mtime_file(path, PROCESSING_TIMEOUT):
            print(f"Skipping (resume <30m): {path}")
            return True

    if is_recent_timestamp_file(done_path, DONE_TIMEOUT):
        print(f"Skipping (done <24h): {done_path}")
        return True

    if is_recent_mtime_file(cache_path, DONE_TIMEOUT):
        print(f"Skipping (parquet <24h): {cache_path}")
        return True

    if resume_path.exists() or resume_tmp_path.exists():
        return False

    if skip_existing and cache_path.exists():
        print(f"Skipping (exists): {cache_path}")
        return True

    return False


def mark_processing(cache_path: Path) -> None:
    flag_path = cache_path.with_suffix(".processing")
    flag_path.write_text(datetime.now(timezone.utc).isoformat() + "\n")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='Manual tool for running model predictions')
    arg_parser.add_argument('-l', '--list_models', action='store_true', help='list available models')
    arg_parser.add_argument('-m', '--model', help='specify model to run')
    arg_parser.add_argument('--model_path_pref', default=None, help='optional prefix path for the folder') # folder optional prefix path
    arg_parser.add_argument('-p', '--path', help='specify path to scan for videos')
    arg_parser.add_argument('--video_list_file', default=None, help='path to newline-separated video paths')
    arg_parser.add_argument('-c', '--cam_name', default='top', help='specify camera name. Default=top')
    arg_parser.add_argument(
        '--video_suffixes',
        default='.mp4',
        help="Comma-separated list of video suffixes to include (e.g. '.mp4')"
    )
    arg_parser.add_argument('-y', action='store_true', help='skip asking for confirmation before running predictions')
    arg_parser.set_defaults(skip_existing=True)
    arg_parser.add_argument('--skip_existing', dest='skip_existing', action='store_true',
                            help='skip videos that already have a predictions parquet for this model')
    arg_parser.add_argument('--no_skip_existing', dest='skip_existing', action='store_false',
                            help='do not skip videos that already have a predictions parquet for this model')
    arg_parser.add_argument(
        '--model_path_override', '--model_override',
        dest='model_path_override',
        default=None,
        help='Override the model_path for this run (does not modify predict_config.json)'
    )
    arg_parser.add_argument('--calib_dir', default=config.CALIBRATION_DIR, help='specify the calibration directory')
    arg_parser.add_argument('--start_x', default=None, help='specify X start position of the screen')
    arg_parser.add_argument('--pix_cm', default=None, help='specify ratio of pixels to centimeters for the screen')
    arg_parser.add_argument('--screen_y',  default=None, help='location of screen along Y axis in cm')

    args = arg_parser.parse_args()
    pred_conf = config.load_configuration('predict')

    if args.start_x is not None:
        config.SCREEN_START_X_CM = float(args.start_x)
    if args.pix_cm is not None:
        config.SCREEN_PIX_CM = float(args.pix_cm)
    if args.screen_y is not None:
        config.SCREEN_Y_CM = float(args.screen_y)
    config.IS_SCREEN_CONFIGURED_FOR_POSE = (config.SCREEN_START_X_CM is not None) and (config.SCREEN_PIX_CM is not None)
    config.CALIBRATION_DIR = args.calib_dir

    if args.list_models:
        print_models(pred_conf)
        sys.exit(0)

    if not args.model or args.model not in pred_conf.keys():
        print('You must specify one of the configured models to run, using -m/--model.')
        print_models(pred_conf)
        sys.exit(1)

    if not args.path and not args.video_list_file:
        print('You must specify path to scan for videos using -p/--path or --video_list_file')
        sys.exit(1)

    video_suffixes = [s.strip() for s in args.video_suffixes.split(",") if s.strip()]
    video_suffixes = [s if s.startswith(".") else f".{s}" for s in video_suffixes]
    video_suffixes = {s.lower() for s in video_suffixes}
    if args.video_list_file:
        video_paths = load_video_list(args.video_list_file, args.cam_name, video_suffixes=video_suffixes)
    else:
        video_paths = scan_path_for_videos(args.path, args.cam_name, video_suffixes=video_suffixes)
    print_videos_found(video_paths)

    if not args.y:
        res = None
        while not res:
            res = input('Do you want to continue? [y]\n>> ')
        if res.lower() not in ['y', 'yes']:
            print('Aborting...')
            sys.exit(1)
    
    # if -model_path_pref is given, prepend it to the model path in pred_conf
    if args.model_path_pref is not None:
        for m in pred_conf.keys():
            pred_conf[m]['model_path'] = str(Path(args.model_path_pref) / Path(pred_conf[m]['model_path']).relative_to('/'))
            print(f'Updated model path for {m} to {pred_conf[m]["model_path"]}')
    # if -model_path_override is given, override the model path in pred_conf
    if args.model_path_override:
        override_path = Path(args.model_path_override).expanduser()
        if not override_path.exists():
            print(f'[ERROR] model_path_override does not exist: {override_path}', file=sys.stderr)
            sys.exit(1)
        pred_conf[args.model] = dict(pred_conf[args.model])
        pred_conf[args.model]['model_path'] = override_path.resolve().as_posix()

    predictor = load_predictor(pred_conf, args.model, args.cam_name)
    for video_path in video_paths:
        cache_path = None
        if hasattr(predictor, "get_predicted_cache_path"):
            try:
                cache_path = predictor.get_predicted_cache_path(video_path)
            except Exception:
                cache_path = None

        if cache_path is not None:
            if should_skip_processing(cache_path, skip_existing=args.skip_existing):
                continue
            mark_processing(cache_path)

        predict_video(predictor, video_path)
