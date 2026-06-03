#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import shutil
import subprocess
import sys
from pathlib import Path

ARENA_ROOT = Path(__file__).resolve().parents[1]
if str(ARENA_ROOT) not in sys.path:
    sys.path.insert(0, str(ARENA_ROOT))

try:
    import config
    from db_models import ORM, Video
    HAS_DB = True
except Exception as exc:
    HAS_DB = False
    DB_IMPORT_ERROR = exc

FFMPEG_BIN = "/usr/bin/ffmpeg" if Path("/usr/bin/ffmpeg").exists() else "ffmpeg"
FFPROBE_BIN = "/usr/bin/ffprobe" if Path("/usr/bin/ffprobe").exists() else "ffprobe"
NICE_BIN = shutil.which("nice")
IONICE_BIN = shutil.which("ionice")


def iter_avi_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".avi":
            yield path


def run_cmd(cmd: list[str], check: bool = True):
    return subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def probe_has_audio(src: Path):
    cmd = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(src),
    ]
    proc = run_cmd(cmd, check=False)
    if proc.returncode != 0:
        return None
    return bool(proc.stdout.strip())


def mp4_is_valid(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return False
    cmd = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = run_cmd(cmd, check=False)
    if proc.returncode != 0:
        return False
    value = proc.stdout.strip()
    if not value or value == "N/A":
        return False
    try:
        return float(value) > 0
    except ValueError:
        return False


def build_ffmpeg_cmd(src: Path, dest: Path, overwrite: bool, quiet: bool, has_audio, threads=1):
    cmd = [
    ]
    if NICE_BIN:
        cmd += [NICE_BIN, "-n", "19"]
    if IONICE_BIN:
        cmd += [IONICE_BIN, "-c3"]
    cmd += [
        FFMPEG_BIN,
        "-hide_banner",
    ]
    if quiet:
        cmd += ["-loglevel", "error"]
    else:
        cmd += ["-loglevel", "warning", "-stats"]
    cmd.append("-y" if overwrite else "-n")
    cmd += [
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        "-pix_fmt",
        "yuv420p",
    ]
    if has_audio is not False:
        cmd += ["-map", "0:a?"]
    if has_audio is not False:
        cmd += ["-c:a", "aac", "-b:a", "128k"]
    if threads is not None:
        cmd += ["-threads", str(threads)]
    cmd.append(str(dest))
    return cmd


def format_returncode(returncode: int):
    if returncode >= 0:
        return str(returncode)
    try:
        return f"{returncode} ({signal.Signals(-returncode).name})"
    except ValueError:
        return str(returncode)


def convert_file(src: Path, dest: Path, overwrite: bool, dry_run: bool, quiet: bool):
    has_audio = probe_has_audio(src)
    tmp_dest = dest.with_name(f"{dest.stem}.tmp{dest.suffix}")
    if tmp_dest.exists():
        tmp_dest.unlink()
    cmd = build_ffmpeg_cmd(src, tmp_dest, True, quiet, has_audio, threads=1)
    if dry_run:
        return True, cmd, None, None

    proc = subprocess.run(cmd, check=False)
    if proc.returncode == 0:
        try:
            os.replace(tmp_dest, dest)
            return True, cmd, None, None
        except OSError as exc:
            if tmp_dest.exists():
                tmp_dest.unlink()
            return False, cmd, f"os.replace failed: {exc}", None

    if tmp_dest.exists():
        tmp_dest.unlink()

    return False, cmd, f"ffmpeg exited with status {format_returncode(proc.returncode)}", None


def should_convert(dest: Path, overwrite: bool):
    if overwrite:
        return True, None
    if not dest.exists():
        return True, None
    if mp4_is_valid(dest):
        return False, "exists"
    return True, "invalid"


def main():
    parser = argparse.ArgumentParser(
        description="Recursively convert .avi files under a folder to .mp4."
    )
    parser.add_argument("folder", help="Root folder to scan")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .mp4 files",
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete .avi after successful conversion",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print ffmpeg commands without executing",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip database updates (file conversion only)",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry videos marked with compression_status=2 (DB errors)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress ffmpeg progress output (errors only)",
    )
    args = parser.parse_args()

    root = Path(args.folder).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"Folder not found or not a directory: {root}", file=sys.stderr)
        return 2

    avi_files = list(iter_avi_files(root))
    if not avi_files:
        print("No .avi files found.")
        return 0

    total = len(avi_files)
    converted = 0
    skipped = 0
    failed = 0

    use_db = not args.no_db
    if use_db:
        if not HAS_DB:
            print(
                f"DB modules not available ({DB_IMPORT_ERROR}). "
                "Run with --no-db to skip DB updates.",
                file=sys.stderr,
            )
            return 2
        if getattr(config, "DISABLE_DB", False):
            print(
                "DB is disabled via config.DISABLE_DB. "
                "Run with --no-db to skip DB updates.",
                file=sys.stderr,
            )
            return 2

    def find_video_record(session, src_path: Path):
        if not use_db:
            return None
        return session.query(Video).filter_by(path=src_path.as_posix()).first()

    if use_db:
        with ORM().session() as s:
            for idx, src in enumerate(avi_files, start=1):
                dest = src.with_suffix(".mp4")
                video_row = find_video_record(s, src)
                status = getattr(video_row, "compression_status", 0) if video_row else None
                should_run, existing_state = should_convert(dest, args.overwrite)

                if status == 1 and not should_run:
                    skipped += 1
                    print(f"[{idx}/{total}] Skip (compressed): {src}")
                    if video_row and video_row.path != dest.as_posix() and not args.dry_run:
                        video_row.path = dest.as_posix()
                        s.commit()
                    if args.delete_source and not args.dry_run:
                        try:
                            src.unlink()
                        except OSError as exc:
                            print(f"Warning: could not delete {src}: {exc}", file=sys.stderr)
                    continue
                if status == 2 and not (args.retry_errors or args.overwrite or existing_state == "invalid"):
                    skipped += 1
                    print(f"[{idx}/{total}] Skip (error status): {src}")
                    continue

                if video_row is None:
                    print(
                        f"[{idx}/{total}] Warning: not found in DB (will convert without DB update): {src}",
                        file=sys.stderr,
                    )

                if not should_run:
                    skipped += 1
                    print(f"[{idx}/{total}] Skip (exists): {dest}")
                    if video_row and not args.dry_run:
                        video_row.path = dest.as_posix()
                        video_row.compression_status = 1
                        s.commit()
                        if args.delete_source:
                            try:
                                src.unlink()
                            except OSError as exc:
                                print(
                                    f"Warning: could not delete {src}: {exc}",
                                    file=sys.stderr,
                                )
                    continue
                if existing_state == "invalid":
                    print(f"[{idx}/{total}] Rebuilding invalid MP4: {dest}", file=sys.stderr)

                print(f"[{idx}/{total}] Converting: {src} -> {dest}")
                ok, cmd, error, note = convert_file(src, dest, args.overwrite, args.dry_run, args.quiet)
                print(" ".join(cmd))
                if note:
                    print(f"Note: {note}", file=sys.stderr)
                if not ok:
                    failed += 1
                    print(f"Failed: {src}", file=sys.stderr)
                    if error:
                        print(f"Reason: {error}", file=sys.stderr)
                    if video_row and not args.dry_run:
                        video_row.compression_status = 2
                        s.commit()
                    continue

                converted += 1
                if video_row and not args.dry_run:
                    video_row.path = dest.as_posix()
                    video_row.compression_status = 1
                    s.commit()
                if args.delete_source and not args.dry_run:
                    try:
                        src.unlink()
                    except OSError as exc:
                        print(f"Warning: could not delete {src}: {exc}", file=sys.stderr)
    else:
        for idx, src in enumerate(avi_files, start=1):
            dest = src.with_suffix(".mp4")
            should_run, existing_state = should_convert(dest, args.overwrite)
            if not should_run:
                skipped += 1
                print(f"[{idx}/{total}] Skip (exists): {dest}")
                continue
            if existing_state == "invalid":
                print(f"[{idx}/{total}] Rebuilding invalid MP4: {dest}", file=sys.stderr)

            print(f"[{idx}/{total}] Converting: {src} -> {dest}")
            ok, cmd, error, note = convert_file(src, dest, args.overwrite, args.dry_run, args.quiet)
            print(" ".join(cmd))
            if note:
                print(f"Note: {note}", file=sys.stderr)
            if not ok:
                failed += 1
                print(f"Failed: {src}", file=sys.stderr)
                if error:
                    print(f"Reason: {error}", file=sys.stderr)
                continue

            converted += 1
            if args.delete_source and not args.dry_run:
                try:
                    src.unlink()
                except OSError as exc:
                    print(f"Warning: could not delete {src}: {exc}", file=sys.stderr)

    print(
        f"Done. total={total} converted={converted} skipped={skipped} failed={failed}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
