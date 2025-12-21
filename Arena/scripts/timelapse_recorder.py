import argparse
import hashlib
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config

import requests
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # Pillow optional; overlay disabled if missing
    Image = ImageDraw = ImageFont = None
_OVERLAY_TIMESTAMP_WARNING_EMITTED = False


ENV_OVERRIDES = {
    'enable': ('TIMELAPSE_ENABLE', 'bool'),
    'camera_names': ('TIMELAPSE_CAMERA_NAMES', 'list'),
    'base_dir': ('TIMELAPSE_BASE_DIR', 'str'),
    'captures_dir': ('TIMELAPSE_CAPTURES_DIR', 'str'),
    'hourly_dir': ('TIMELAPSE_HOURLY_DIR', 'str'),
    'daily_dir': ('TIMELAPSE_DAILY_DIR', 'str'),
    'frame_interval_seconds': ('TIMELAPSE_FRAME_INTERVAL_SECONDS', 'int'),
    'hourly_framerate': ('TIMELAPSE_HOURLY_FRAMERATE', 'int'),
    'daily_framerate': ('TIMELAPSE_DAILY_FRAMERATE', 'int'),
    'delete_images_after_hourly': ('TIMELAPSE_DELETE_IMAGES_AFTER_HOURLY', 'bool'),
    'delete_hourlies_after_daily': ('TIMELAPSE_DELETE_HOURLY_AFTER_DAILY', 'bool'),
    'arena_url': ('TIMELAPSE_ARENA_URL', 'str'),
    'arena_timeout': ('TIMELAPSE_ARENA_TIMEOUT', 'float'),
    'overlay_timestamp': ('TIMELAPSE_OVERLAY_TIMESTAMP', 'bool'),
}


@dataclass
class TimelapseConfig:
    enable: bool
    camera_names: List[str]
    base_dir: Path
    captures_dir: Path
    hourly_dir: Path
    daily_dir: Path
    frame_interval_seconds: int
    hourly_framerate: int
    daily_framerate: int
    delete_images_after_hourly: bool
    delete_hourlies_after_daily: bool
    arena_url: str
    arena_timeout: float
    overlay_timestamp: bool

    @property
    def log_dir(self) -> Path:
        return self.base_dir / 'logs'


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    cfg = _build_config(args)
    _ensure_directories(cfg)
    logger = _build_logger(cfg)
    if not cfg.enable:
        logger.info('Timelapse is disabled in config; exiting.')
        return 0

    try:
        return args.func(args, cfg, logger)
    except ValueError as exc:
        logger.error(str(exc))
        return 1
    except subprocess.CalledProcessError:
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Timelapse capture and rendering utility')
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--enable', dest='enable', action='store_true', help='Force enable timelapse')
    common.add_argument('--disable', dest='enable', action='store_false', help='Force disable timelapse')
    common.set_defaults(enable=None)
    common.add_argument('--camera', dest='cameras', action='append',
                        help='Process only these cameras (can be passed multiple times)')
    common.add_argument('--base-dir', help='Override timelapse base directory')
    common.add_argument('--captures-dir', help='Override directory for raw captures')
    common.add_argument('--hourly-dir', help='Override directory for hourly clips')
    common.add_argument('--daily-dir', help='Override directory for daily clips')
    common.add_argument('--frame-interval', type=int, help='Override capture interval in seconds')
    common.add_argument('--hourly-framerate', type=int, help='Override hourly clip framerate')
    common.add_argument('--daily-framerate', type=int, help='Override daily clip framerate')
    common.add_argument('--arena-url', help='Override Arena API base URL')
    common.add_argument('--arena-timeout', type=float, help='Override Arena API timeout in seconds')
    common.add_argument('--delete-images-after-hourly', dest='delete_images_after_hourly', action='store_true')
    common.add_argument('--keep-images-after-hourly', dest='delete_images_after_hourly', action='store_false')
    common.set_defaults(delete_images_after_hourly=None)
    common.add_argument('--delete-hourlies-after-daily', dest='delete_hourlies_after_daily', action='store_true')
    common.add_argument('--keep-hourlies-after-daily', dest='delete_hourlies_after_daily', action='store_false')
    common.set_defaults(delete_hourlies_after_daily=None)
    common.add_argument('--overlay-timestamp', dest='overlay_timestamp', action='store_true',
                        help='Draw capture timestamp on saved frames')
    common.add_argument('--no-overlay-timestamp', dest='overlay_timestamp', action='store_false',
                        help='Disable timestamp overlay on frames')
    common.set_defaults(overlay_timestamp=None)

    subparsers = parser.add_subparsers(dest='command', required=True)

    cap_parser = subparsers.add_parser('capture-loop', parents=[common], help='Continuously capture frames')
    cap_parser.set_defaults(func=_handle_capture_loop)

    hourly_parser = subparsers.add_parser('build-hourly', parents=[common], help='Build hourly clip for date/hour')
    hourly_parser.add_argument('--date', required=True, help='Date in YYYYMMDD format')
    hourly_parser.add_argument('--hour', required=True, help='Hour in HH format (00-23)')
    hourly_parser.set_defaults(func=_handle_build_hourly)

    daily_parser = subparsers.add_parser('build-daily', parents=[common], help='Build daily timelapse for date')
    daily_parser.add_argument('--date', required=True, help='Date in YYYYMMDD format')
    daily_parser.set_defaults(func=_handle_build_daily)

    hourly_today = subparsers.add_parser('build-hourly-for-today', parents=[common],
                                         help='Build the most recently finished hour')
    hourly_today.set_defaults(func=_handle_build_hourly_today)

    daily_today = subparsers.add_parser('build-daily-for-today', parents=[common],
                                        help='Build timelapse for today')
    daily_today.set_defaults(func=_handle_build_daily_today)

    validate_parser = subparsers.add_parser('validate-all-hours', parents=[common],
                                            help='Scan capture folders and backfill missing hourly clips')
    validate_parser.add_argument('--date', help='Only validate a specific date (YYYYMMDD)')
    validate_parser.add_argument('--include-latest', action='store_true',
                                 help='Also attempt to encode the most recent hour even if it may still be capturing')
    validate_parser.set_defaults(func=_handle_validate_all_hours)

    return parser


def _handle_build_hourly(args: argparse.Namespace, cfg: TimelapseConfig, logger: logging.Logger) -> int:
    date = _validate_date(args.date)
    hour = _validate_hour(args.hour)
    return _run_per_camera(cfg, logger, lambda camera: _build_hourly_clip(date, hour, cfg, logger, camera))


def _handle_build_daily(args: argparse.Namespace, cfg: TimelapseConfig, logger: logging.Logger) -> int:
    date = _validate_date(args.date)
    logger.info('Validating hourlies for %s before daily stitch', date)
    def runner(camera: str) -> int:
        rc = _build_missing_hourlies(cfg, logger, camera, date_filter=date, include_latest=True)
        rc |= _build_daily_clip(date, cfg, logger, camera)
        return rc
    return _run_per_camera(cfg, logger, runner)


def _handle_build_hourly_today(_: argparse.Namespace, cfg: TimelapseConfig, logger: logging.Logger) -> int:
    target = datetime.now() - timedelta(hours=1)
    date = target.strftime('%Y%m%d')
    hour = target.strftime('%H')

    def builder(camera: str) -> int:
        if _hour_has_frames(cfg, camera, date, hour):
            return _build_hourly_clip(date, hour, cfg, logger, camera)
        fallback = _find_latest_hour_with_frames(cfg, camera, before=datetime.now())
        if fallback:
            fb_date, fb_hour = fallback
            logger.info('No frames found for %s %s (%s); using most recent available %s %s',
                        date, hour, camera, fb_date, fb_hour)
            return _build_hourly_clip(fb_date, fb_hour, cfg, logger, camera)
        logger.info('No captured frames available yet for %s; nothing to build.', camera)
        return 0

    return _run_per_camera(cfg, logger, builder)


def _handle_build_daily_today(_: argparse.Namespace, cfg: TimelapseConfig, logger: logging.Logger) -> int:
    target = datetime.now() - timedelta(days=1)
    date = target.strftime('%Y%m%d')
    logger.info('Validating hourlies for %s before daily stitch', date)
    def runner(camera: str) -> int:
        rc = _build_missing_hourlies(cfg, logger, camera, date_filter=date, include_latest=True)
        rc |= _build_daily_clip(date, cfg, logger, camera)
        return rc
    return _run_per_camera(cfg, logger, runner)


def _handle_validate_all_hours(args: argparse.Namespace, cfg: TimelapseConfig, logger: logging.Logger) -> int:
    date = _validate_date(args.date) if args.date else None
    return _run_per_camera(
        cfg, logger,
        lambda camera: _build_missing_hourlies(cfg, logger, camera, date_filter=date, include_latest=args.include_latest)
    )


def _hour_has_frames(cfg: TimelapseConfig, camera: str, date: str, hour: str) -> bool:
    frames_dir = cfg.captures_dir / camera / date / hour
    if not frames_dir.exists():
        return False
    return any(frames_dir.glob('*.jpg'))


def _find_latest_hour_with_frames(cfg: TimelapseConfig, camera: str,
                                  before: Optional[datetime] = None) -> Optional[tuple]:
    comparator = before or datetime.max
    camera_root = cfg.captures_dir / camera
    for date_dir in sorted(camera_root.glob('*'), reverse=True):
        if not date_dir.is_dir():
            continue
        date = date_dir.name
        try:
            datetime.strptime(date, '%Y%m%d')
        except ValueError:
            continue
        for hour_dir in sorted(date_dir.glob('[0-2][0-9]'), reverse=True):
            if not hour_dir.is_dir():
                continue
            hour = hour_dir.name
            try:
                hour_start = datetime.strptime(f'{date}{hour}', '%Y%m%d%H')
            except ValueError:
                continue
            if hour_start >= comparator:
                continue
            if any(hour_dir.glob('*.jpg')):
                return date, hour
    return None


def _run_per_camera(cfg: TimelapseConfig, logger: logging.Logger, fn) -> int:
    if not cfg.camera_names:
        logger.error('No cameras configured; nothing to process.')
        return 1
    exit_code = 0
    for camera in cfg.camera_names:
        exit_code |= fn(camera)
    return exit_code


def _build_missing_hourlies(cfg: TimelapseConfig, logger: logging.Logger, camera: str,
                            *, date_filter: Optional[str] = None,
                            include_latest: bool = False) -> int:
    """
    Scan capture directories for the given camera and create hourly clips for any finished hours that do not yet
    have a corresponding MP4. We only process hours that started at least one hour ago so
    the capture loop has time to finish writing frames.
    """
    now = datetime.now()
    cutoff = now - timedelta(hours=1)
    exit_code = 0
    camera_root = cfg.captures_dir / camera
    for date_dir in sorted(camera_root.glob('*')):
        if not date_dir.is_dir():
            continue
        date = date_dir.name
        try:
            datetime.strptime(date, '%Y%m%d')
        except ValueError:
            continue
        if date_filter and date != date_filter:
            continue
        for hour_dir in sorted(date_dir.glob('[0-2][0-9]')):
            if not hour_dir.is_dir():
                continue
            hour = hour_dir.name
            try:
                hour_start = datetime.strptime(f'{date}{hour}', '%Y%m%d%H')
            except ValueError:
                continue
            if not include_latest and hour_start > cutoff:
                continue
            output_path = cfg.hourly_dir / camera / f'{date}_{hour}.mp4'
            if output_path.exists():
                continue
            exit_code |= _build_hourly_clip(date, hour, cfg, logger, camera)
    return exit_code


def _handle_capture_loop(_: argparse.Namespace, cfg: TimelapseConfig, logger: logging.Logger) -> int:
    if not cfg.camera_names:
        logger.error('At least one camera must be configured to start capture loop')
        return 1
    logger.info('Starting capture loop for cameras %s every %s seconds', ', '.join(cfg.camera_names),
                cfg.frame_interval_seconds)
    next_capture = time.monotonic()
    last_frame_hash: Dict[str, str] = {}
    try:
        while True:
            for camera in cfg.camera_names:
                start = time.time()
                try:
                    frame_bytes = _fetch_frame_from_arena(cfg, camera)
                    digest = hashlib.sha1(frame_bytes).hexdigest()
                    if digest == last_frame_hash.get(camera):
                        logger.warning('Skipping stale frame for %s (no changes detected)', camera)
                        continue
                    _save_frame_bytes(frame_bytes, cfg, logger, start, camera)
                    last_frame_hash[camera] = digest
                except Exception as exc:
                    logger.error('Failed to capture frame from %s: %s', camera, exc)
            next_capture += cfg.frame_interval_seconds
            sleep_for = next_capture - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_capture = time.monotonic()
    except KeyboardInterrupt:
        logger.info('Capture loop interrupted by user')
        return 0


def _build_hourly_clip(date: str, hour: str, cfg: TimelapseConfig, logger: logging.Logger, camera: str) -> int:
    frames_dir = cfg.captures_dir / camera / date / hour
    if not frames_dir.exists():
        logger.info('No frames directory %s for %s %s (%s)', frames_dir, date, hour, camera)
        return 0
    images = sorted(frames_dir.glob('*.jpg'))
    if not images:
        logger.info('No frames to encode for %s %s (%s)', date, hour, camera)
        return 0

    output_path = cfg.hourly_dir / camera / f'{date}_{hour}.mp4'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(frames_dir / '*.jpg')
    logger.info('Building hourly clip %s from %s with %s frames (camera %s)', output_path, frames_dir,
                len(images), camera)
    cmd = [
        '-y',
        '-framerate', str(cfg.hourly_framerate),
        '-pattern_type', 'glob',
        '-i', pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        str(output_path)
    ]
    _run_ffmpeg(cmd, logger)
    logger.info('Created hourly clip %s', output_path)
    if cfg.delete_images_after_hourly:
        _delete_files(images, logger)
        _delete_directory_if_empty(frames_dir)
        _delete_directory_if_empty(frames_dir.parent)
    return 0


def _build_daily_clip(date: str, cfg: TimelapseConfig, logger: logging.Logger, camera: str) -> int:
    hourly_root = cfg.hourly_dir / camera
    hourly_files = sorted(hourly_root.glob(f'{date}_*.mp4'))
    hourly_files = [f for f in hourly_files if f.is_file()]
    if not hourly_files:
        logger.info('No hourly clips found for %s (%s)', date, camera)
        return 0

    concat_file = cfg.base_dir / f'hourly_list_{camera}_{date}.txt'
    with concat_file.open('w') as fh:
        for clip in hourly_files:
            fh.write(f"file '{_escape_ffmpeg_path(clip)}'\n")

    output_path = cfg.daily_dir / camera / f'{date}_timelapse.mp4'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info('Building daily clip %s from %s hourly files (camera %s)', output_path, len(hourly_files), camera)
    cmd = [
        '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(concat_file),
        '-c', 'copy',
        str(output_path)
    ]
    try:
        _run_ffmpeg(cmd, logger)
        logger.info('Created daily clip %s', output_path)
    finally:
        concat_file.unlink(missing_ok=True)

    if cfg.delete_hourlies_after_daily:
        _delete_files(hourly_files, logger)
    return 0


def _run_ffmpeg(args: List[str], logger: logging.Logger) -> None:
    cmd = _build_priority_command(['ffmpeg', *args])
    logger.info('Running ffmpeg command: %s', ' '.join(shlex.quote(part) for part in cmd))
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        logger.error('ffmpeg failed with exit code %s', exc.returncode)
        sys.stderr.write(exc.stderr.decode(errors='ignore'))
        raise


def _build_priority_command(cmd: List[str]) -> List[str]:
    wrapped = []
    ionice = shutil.which('ionice')
    nice = shutil.which('nice')
    if ionice:
        wrapped.extend([ionice, '-c', '3'])
    if nice:
        wrapped.extend([nice, '-n', '10'])
    wrapped.extend(cmd)
    return wrapped


def _delete_files(paths: List[Path], logger: logging.Logger) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            logger.exception('Failed to delete %s', path)


def _delete_directory_if_empty(path: Path) -> None:
    try:
        if path.exists() and not any(path.iterdir()):
            path.rmdir()
    except Exception:
        pass


def _build_logger(cfg: TimelapseConfig) -> logging.Logger:
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('timelapse')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(cfg.log_dir / 'timelapse.log')
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


def _build_config(args: argparse.Namespace) -> TimelapseConfig:
    merged = _base_timelapse_settings()
    merged.update(_collect_env_overrides())
    merged.update(_cli_overrides(args))
    base_dir = _expand_path(merged['base_dir'])
    captures_dir = _expand_path(merged.get('captures_dir')) or base_dir / 'captures'
    hourly_dir = _expand_path(merged.get('hourly_dir')) or base_dir / 'hourly'
    daily_dir = _expand_path(merged.get('daily_dir')) or base_dir / 'daily'

    camera_names = merged.get('camera_names')
    if isinstance(camera_names, str):
        camera_names = [camera_names]
    camera_names = [name for name in (camera_names or []) if name]
    camera_names = list(dict.fromkeys(camera_names))  # preserve order, drop duplicates

    return TimelapseConfig(
        enable=_to_bool(merged['enable']),
        camera_names=camera_names,
        base_dir=base_dir,
        captures_dir=captures_dir,
        hourly_dir=hourly_dir,
        daily_dir=daily_dir,
        frame_interval_seconds=int(merged['frame_interval_seconds']),
        hourly_framerate=int(merged['hourly_framerate']),
        daily_framerate=int(merged['daily_framerate']),
        delete_images_after_hourly=_to_bool(merged['delete_images_after_hourly']),
        delete_hourlies_after_daily=_to_bool(merged['delete_hourlies_after_daily']),
        arena_url=merged.get('arena_url'),
        arena_timeout=float(merged.get('arena_timeout', 10.0)),
        overlay_timestamp=_to_bool(merged.get('overlay_timestamp', False)),
    )


def _collect_env_overrides() -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for key, (env_name, typ) in ENV_OVERRIDES.items():
        raw = os.getenv(env_name)
        if raw is None:
            continue
        overrides[key] = _parse_env_value(raw, typ)
    return overrides


def _cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides = {}
    mapping = {
        'enable': args.enable,
        'camera_names': args.cameras,
        'base_dir': args.base_dir,
        'captures_dir': args.captures_dir,
        'hourly_dir': args.hourly_dir,
        'daily_dir': args.daily_dir,
        'frame_interval_seconds': args.frame_interval,
        'hourly_framerate': args.hourly_framerate,
        'daily_framerate': args.daily_framerate,
        'delete_images_after_hourly': args.delete_images_after_hourly,
        'delete_hourlies_after_daily': args.delete_hourlies_after_daily,
        'arena_url': args.arena_url,
        'arena_timeout': args.arena_timeout,
        'overlay_timestamp': args.overlay_timestamp,
    }
    for key, value in mapping.items():
        if value is not None:
            overrides[key] = value
    return overrides


def _parse_env_value(raw: str, typ: str) -> Any:
    if typ == 'bool':
        return raw.strip().lower() in ['1', 'true', 'yes', 'on']
    if typ == 'int':
        return int(raw.strip())
    if typ == 'float':
        return float(raw.strip())
    if typ == 'list':
        if not raw.strip():
            return []
        return [item.strip() for item in raw.split(',') if item.strip()]
    return raw.strip()


def _expand_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _ensure_directories(cfg: TimelapseConfig) -> None:
    cfg.base_dir.mkdir(parents=True, exist_ok=True)
    cfg.captures_dir.mkdir(parents=True, exist_ok=True)
    cfg.hourly_dir.mkdir(parents=True, exist_ok=True)
    cfg.daily_dir.mkdir(parents=True, exist_ok=True)
    for camera in cfg.camera_names:
        (cfg.captures_dir / camera).mkdir(parents=True, exist_ok=True)
        (cfg.hourly_dir / camera).mkdir(parents=True, exist_ok=True)
        (cfg.daily_dir / camera).mkdir(parents=True, exist_ok=True)


def _save_frame_bytes(frame_bytes: bytes, cfg: TimelapseConfig, logger: logging.Logger,
                      capture_time: float, camera: str) -> None:
    ts = datetime.fromtimestamp(capture_time)
    date_folder = cfg.captures_dir / camera / ts.strftime('%Y%m%d')
    hour_folder = date_folder / ts.strftime('%H')
    hour_folder.mkdir(parents=True, exist_ok=True)
    filename = ts.strftime('%Y%m%d_%H-%M.jpg')
    output_path = hour_folder / filename
    payload = frame_bytes
    if cfg.overlay_timestamp:
        payload = _overlay_timestamp(frame_bytes, ts, logger)
    output_path.write_bytes(payload)
    logger.info('Saved frame from %s to %s', camera, output_path)


def _overlay_timestamp(frame_bytes: bytes, ts: datetime, logger: logging.Logger) -> bytes:
    global _OVERLAY_TIMESTAMP_WARNING_EMITTED
    if Image is None:
        if not _OVERLAY_TIMESTAMP_WARNING_EMITTED:
            logger.warning('Timestamp overlay requested but Pillow is not installed; saving raw frames.')
            _OVERLAY_TIMESTAMP_WARNING_EMITTED = True
        return frame_bytes
    try:
        with Image.open(BytesIO(frame_bytes)) as img:
            img = img.convert('RGB')
            text = ts.strftime('%Y-%m-%d %H:%M:%S')
            font = ImageFont.load_default()
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            if hasattr(overlay_draw, 'textbbox'):
                bbox = overlay_draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width, text_height = overlay_draw.textsize(text, font=font)
            margin = 10
            padding = 6
            x = margin
            y = max(margin, img.height - text_height - margin)
            bg_coords = [
                (x - padding, y - padding),
                (x + text_width + padding, y + text_height + padding),
            ]
            overlay_draw.rectangle(bg_coords, fill=(0, 0, 0, 160))
            overlay_draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
            stamped = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
            buffer = BytesIO()
            stamped.save(buffer, format='JPEG', quality=95)
            return buffer.getvalue()
    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning('Failed to overlay timestamp; saving raw frame: %s', exc)
    return frame_bytes


def _fetch_frame_from_arena(cfg: TimelapseConfig, camera: str) -> bytes:
    if not cfg.arena_url:
        raise RuntimeError('Arena URL not configured')
    url = f'{cfg.arena_url.rstrip("/")}/timelapse_frame/{camera}'
    resp = requests.get(url, timeout=cfg.arena_timeout)
    resp.raise_for_status()
    if not resp.content:
        raise RuntimeError('Arena returned empty frame')
    return resp.content


def _validate_date(value: str) -> str:
    try:
        datetime.strptime(value, '%Y%m%d')
    except ValueError as exc:
        raise ValueError(f'Invalid date "{value}": {exc}') from exc
    return value


def _validate_hour(value: str) -> str:
    if len(value) != 2 or not value.isdigit() or not (0 <= int(value) <= 23):
        raise ValueError(f'Hour must be in HH format, received {value}')
    return value


def _escape_ffmpeg_path(path: Path) -> str:
    return str(path).replace("'", "'\\''")


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ['1', 'true', 'yes', 'on']
    return bool(value)


def _base_timelapse_settings() -> Dict[str, Any]:
    settings = getattr(config, 'TIMELAPSE_SETTINGS', None)
    if settings is None:
        return {
            'enable': True,
            'camera_names': [],
            'base_dir': './timelapse_data',
            'captures_dir': None,
            'hourly_dir': None,
            'daily_dir': None,
            'frame_interval_seconds': 60,
            'hourly_framerate': 24,
            'daily_framerate': 24,
            'delete_images_after_hourly': False,
            'delete_hourlies_after_daily': True,
            'arena_url': 'http://localhost:5084',
            'arena_timeout': 10.0,
            'overlay_timestamp': True,
        }
    return dict(settings)


if __name__ == '__main__':
    raise SystemExit(main())
