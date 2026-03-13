import argparse
import config
import sys
from pathlib import Path
import importlib
import torch
from datetime import datetime, timedelta, timezone


def print_models(pconf):
    print(f'Configured models:')
    for model_name in pconf.keys():
        print(f' - {model_name}')


def scan_path_for_videos(dir_path, cam_name, *, video_suffixes=None):
    if video_suffixes is None:
        video_suffixes = [".mp4", ".avi"]
    allowed_suffixes = {s.lower() for s in video_suffixes}
    return [x for x in Path(dir_path).rglob(f'{cam_name}*') if x.suffix.lower() in allowed_suffixes]


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




def should_skip_or_mark_processing(cache_path: Path, *, skip_existing: bool) -> bool:
    if skip_existing and cache_path.exists():
        print(f"Skipping (exists): {cache_path}")
        return True

    flag_path = cache_path.with_suffix(".processing")
    if (not cache_path.exists()) and flag_path.exists():
        try:
            ts_text = flag_path.read_text().strip()
            ts = datetime.fromisoformat(ts_text)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - ts.astimezone(timezone.utc)
            if age < timedelta(hours=24):
                print(f"Skipping (processing <24h): {flag_path}")
                return True
        except Exception:
            pass

    flag_path.write_text(datetime.now(timezone.utc).isoformat() + "\n")
    return False


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='Manual tool for running model predictions')
    arg_parser.add_argument('-l', '--list_models', action='store_true', help='list available models')
    arg_parser.add_argument('-m', '--model', help='specify model to run')
    arg_parser.add_argument('-p', '--path', help='specify path to scan for videos')
    arg_parser.add_argument('-c', '--cam_name', default='top', help='specify camera name. Default=top')
    arg_parser.add_argument(
        '--video_suffixes',
        default='.mp4,.avi',
        help="Comma-separated list of video suffixes to include (e.g. '.mp4' or '.mp4,.avi')"
    )
    arg_parser.add_argument('-y', action='store_true', help='skip asking for confirmation before running predictions')
    arg_parser.add_argument('--skip_existing', action='store_true', help='skip videos that already have a predictions parquet for this model')
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

    if args.model_path_override:
        override_path = Path(args.model_path_override).expanduser()
        if not override_path.exists():
            print(f'[ERROR] model_path_override does not exist: {override_path}', file=sys.stderr)
            sys.exit(1)
        pred_conf[args.model] = dict(pred_conf[args.model])
        pred_conf[args.model]['model_path'] = override_path.resolve().as_posix()

    if not args.path:
        print('You must specify path to scan for videos using -p/--path')
        sys.exit(1)

    video_suffixes = [s.strip() for s in args.video_suffixes.split(",") if s.strip()]
    video_suffixes = [s if s.startswith(".") else f".{s}" for s in video_suffixes]
    video_suffixes = {s.lower() for s in video_suffixes}
    video_paths = scan_path_for_videos(args.path, args.cam_name, video_suffixes=video_suffixes)
    print_videos_found(video_paths)

    if not args.y:
        res = None
        while not res:
            res = input('Do you want to continue? [y]\n>> ')
        if res.lower() not in ['y', 'yes']:
            print('Aborting...')
            sys.exit(1)

    predictor = load_predictor(pred_conf, args.model, args.cam_name)
    for video_path in video_paths:
        cache_path = None
        if hasattr(predictor, "get_predicted_cache_path"):
            try:
                cache_path = predictor.get_predicted_cache_path(video_path)
            except Exception:
                cache_path = None

        if cache_path is not None:
            if should_skip_or_mark_processing(cache_path, skip_existing=args.skip_existing):
                continue

        predict_video(predictor, video_path)
