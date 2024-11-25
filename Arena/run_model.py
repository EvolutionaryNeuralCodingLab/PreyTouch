import argparse
import config
import sys
from pathlib import Path
import importlib
import torch


def print_models(pconf):
    print(f'Configured models:')
    for model_name in pconf.keys():
        print(f' - {model_name}')


def scan_path_for_videos(dir_path, cam_name):
    return [x for x in Path(dir_path).rglob(f'{cam_name}*') if x.suffix in ['.mp4', '.avi']]


def print_videos_found(vid_paths):
    print(f'Videos found:')
    for vid_path in vid_paths:
        print(f' - {vid_path}')
    print(f'\nTotal of {len(vid_paths)} videos.')


def load_predictor(pconf, model_name, cam_name):
    pconf = pconf[model_name]
    if pconf['predictor_name'] == 'DLCPose':
        from analysis.pose import DLCArenaPose
        return DLCArenaPose(cam_name, is_use_db=False, is_raise_no_caliber=False)

    prd_module, prd_class = config.arena_modules['predictors'][model_name]
    prd_module = importlib.import_module(prd_module)
    return getattr(prd_module, prd_class)(cam_name, pconf['model_path'])


def predict_video(prd, video_path):
    try:
        prd.predict_video(video_path=video_path)
    except Exception as e:
        print(e)
    finally:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='Manual tool for running model predictions')
    arg_parser.add_argument('-l', '--list_models', action='store_true', help='list available models')
    arg_parser.add_argument('-m', '--model', help='specify model to run')
    arg_parser.add_argument('-p', '--path', help='specify path to scan for videos')
    arg_parser.add_argument('-c', '--cam_name', default='top', help='specify camera name. Default=top')
    args = arg_parser.parse_args()
    pred_conf = config.load_configuration('predict')

    if args.list_models:
        print_models(pred_conf)
        sys.exit(0)

    if not args.model or args.model not in pred_conf.keys():
        print('You must specify one of the configured models to run, using -m/--model.')
        print_models(pred_conf)
        sys.exit(1)

    if not args.path:
        print('You must specify path to scan for videos using -p/--path')
        sys.exit(1)

    video_paths = scan_path_for_videos(args.path, args.cam_name)
    print_videos_found(video_paths)

    res = None
    while not res:
        res = input('Do you want to continue? [y]\n>> ')
    if res.lower() not in ['y', 'yes']:
        print('Aborting...')
        sys.exit(1)

    predictor = load_predictor(pred_conf, args.model, args.cam_name)
    for video_path in video_paths:
        predict_video(predictor, video_path)