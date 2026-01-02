#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
import utils
import requests


VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}


def send_document(path: Path, caption: str = ''):
    if not config.TELEGRAM_TOKEN:
        return None
    data = {
        'chat_id': config.TELEGRAM_CHAT_ID,
        'caption': f'({config.ARENA_NAME}): {caption}' if caption else f'({config.ARENA_NAME})',
        'disable_notification': True
    }
    with path.open('rb') as fh:
        files = {'document': fh}
        return requests.post(
            f'https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendDocument',
            data=data,
            files=files,
            timeout=120
        )


def iter_files(folder: Path, recursive: bool, match: str):
    if recursive:
        iterator = folder.rglob('*')
    else:
        iterator = folder.iterdir()
    for path in sorted(iterator):
        if path.is_file():
            if match and match not in path.name:
                continue
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Send all files in a folder to Telegram (single attempt).',
        epilog=(
            'Examples:\n'
            '  Send all daily timelapse files for a specific date:\n'
            '    /home/pogona/miniconda3/envs/PreyTouch/bin/python '
            '/data/PreyTouch/Arena/scripts/send_folder_to_telegram.py '
            '/data/PreyTouch/output/timelapse/daily/top --match 20251224\n'
            '  Send all files under the timelapse daily folder (recursive):\n'
            '    /home/pogona/miniconda3/envs/PreyTouch/bin/python '
            '/data/PreyTouch/Arena/scripts/send_folder_to_telegram.py '
            '/data/PreyTouch/output/timelapse/daily --recursive\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('folder', help='Folder containing files to send')
    parser.add_argument('--recursive', action='store_true', help='Include files in subfolders')
    parser.add_argument('--match', help='Only send files whose name contains this string')
    args = parser.parse_args()

    folder = Path(args.folder).expanduser()
    if not folder.exists() or not folder.is_dir():
        print(f'Folder not found: {folder}')
        return 1
    if not config.TELEGRAM_TOKEN:
        print('Telegram is not configured (missing TELEGRAM_TOKEN).')
        return 1

    failures = 0
    for path in iter_files(folder, args.recursive, args.match):
        caption = path.name
        if path.suffix.lower() in VIDEO_EXTS:
            resp = utils.send_telegram_video(str(path), caption=caption)
        else:
            resp = send_document(path, caption=caption)
        if resp is None or not resp.ok:
            failures += 1
            status = getattr(resp, 'status_code', 'n/a')
            print(f'Failed sending {path.name} (status={status})')
        else:
            print(f'Sent {path.name}')
    return 0 if failures == 0 else 2


if __name__ == '__main__':
    # Examples:
    #   --match <TEXT> sends only files whose name contains <TEXT>.
    #   --recursive walks subfolders (e.g. daily/top + daily/back) instead of only one folder.
    #   /home/pogona/miniconda3/envs/PreyTouch/bin/python /data/PreyTouch/Arena/scripts/send_folder_to_telegram.py \
    #       /data/PreyTouch/output/timelapse/daily/top --match 20251224
    
    #   python /data/PreyTouch/Arena/scripts/send_folder_to_telegram.py \
    #       /data/PreyTouch/output/timelapse/daily --recursive
    raise SystemExit(main())
