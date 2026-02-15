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

def has_existing_prediction(video_path, cam_name, output_dir=None):
    """Check if video already has prediction parquet file for this camera in the same directory as video"""
    video_path = Path(video_path)
    
    # Look for predictions folder in the same directory as the video
    prediction_dir = video_path.parent / "predictions"
    if prediction_dir.exists():
        # Look for any parquet file in the predictions folder for this camera
        parquet_files = list(prediction_dir.glob(f"*{cam_name}*.parquet"))
        return len(parquet_files) > 0
    
    return False

def scan_path_for_videos(dir_path, cam_name="top", skip_existing=True):

    all_videos = [x for x in Path(dir_path).rglob(f'{cam_name}*') if x.suffix in ['.mp4', '.avi']]
    # Filter out videos in folders containing any of the skip_folder names
    filtered_videos = []
    skipped_videos = []
    skipped_existing = []
    
    for video_path in all_videos:
        should_skip = False
        skip_reason = ""
        
        # Check existing predictions
        if not should_skip and skip_existing and has_existing_prediction(video_path, cam_name):
            should_skip = True
            print(f"predictions already exist for video {video_path}, skipping")
            skipped_existing.append(video_path)
        
    
    filtered_videos = [v for v in all_videos if v not in skipped_existing]
    
    # if skipped_existing:
    #     print("Skipped %s videos with existing predictions", len(skipped_existing)) 
    #     for i, skipped in enumerate(skipped_existing):
    #         print("  - %s", skipped)
    
       
    return filtered_videos

def print_videos_found(vid_paths):
    print(f'Videos found:')
    for vid_path in vid_paths:
        print(f' - {vid_path}')
    print(f'\nTotal of {len(vid_paths)} videos.')


def load_predictor(pconf, model_name, cam_name):
    pconf = pconf[model_name]
    if pconf['predictor_name'] == 'DLCPose':
        from analysis.pose import DLCArenaPose
        return DLCArenaPose(cam_name, model_path=pconf['model_path'], is_use_db=False, is_raise_no_caliber=False)

    prd_module, prd_class = config.arena_modules['predictors'][model_name]
    prd_module = importlib.import_module(prd_module)
    return getattr(prd_module, prd_class)(cam_name, pconf['model_path'])

import traceback
def predict_video(prd, video_path):
    try:
        prd.predict_video(video_path=video_path)
    except Exception as e:
        print("An error occurred while predicting the video:")
        traceback.print_exc()
    finally:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='Manual tool for running model predictions')
    arg_parser.add_argument('-l', '--list_models', action='store_true', help='list available models')
    arg_parser.add_argument('-m', '--model', help='specify model to run')
    arg_parser.add_argument('--model_path_pref', default=None, help='optional prefix path for the folder') # folder optional prefix path
    arg_parser.add_argument('--model_path_override', default=None, help='optional model path to override config') # folder optional prefix path
    arg_parser.add_argument('-p', '--path', help='specify path to scan for videos')
    arg_parser.add_argument('-c', '--cam_name', default='top', help='specify camera name. Default=top')
    arg_parser.add_argument('-y', action='store_true', help='skip asking for confirmation before running predictions')
    arg_parser.add_argument('--calib_dir', default=config.CALIBRATION_DIR, help='specify the calibration directory')
    arg_parser.add_argument('--start_x', default=None, help='specify X start position of the screen')
    arg_parser.add_argument('--pix_cm', default=None, help='specify ratio of pixels to centimeters for the screen')
    arg_parser.add_argument('--screen_y',  default=None, help='location of screen along Y axis in cm')
    arg_parser.add_argument('--skip_existing',  default=True, action='store_true', help='skip videos that already have prediction files')


    args = arg_parser.parse_args()
    pred_conf = config.load_configuration('predict')

    config.SCREEN_START_X_CM = float(args.start_x) if args.start_x is not None else None
    config.SCREEN_PIX_CM = float(args.pix_cm) if args.pix_cm is not None else None
    config.SCREEN_Y_CM = float(args.screen_y) if args.screen_y is not None else None
    config.IS_SCREEN_CONFIGURED_FOR_POSE = (args.start_x is not None) and (args.pix_cm is not None)
    config.CALIBRATION_DIR = args.calib_dir

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

    video_paths = scan_path_for_videos(args.path, args.cam_name, args.skip_existing)
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
    if args.model_path_override is not None:
        pred_conf[args.model]['model_path'] = args.model_path_override
        print(f'Overridden model path for {args.model} to {pred_conf[args.model]["model_path"]}')

    predictor = load_predictor(pred_conf, args.model, args.cam_name)
    for video_path in video_paths:
        predict_video(predictor, video_path)
