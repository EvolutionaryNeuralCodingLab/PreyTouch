#!/usr/bin/env python3
"""
Install cron entries for timelapse stitching and configure supervisor to keep the
capture loop running.

This script:
1. Detects the current Python interpreter and timelapse_recorder.py path.
2. Adds the hourly/daily build commands to the current user's crontab (if missing).
3. Writes or updates the supervisor program that runs `timelapse_recorder.py capture-loop`.
"""
import argparse
import subprocess
import sys
from pathlib import Path
import os
import shutil


CRON_TEMPLATE = """\
2 * * * * cd {workdir} && TIMELAPSE_ENABLE=1 {python} {script} build-hourly-for-today
5 0 * * * cd {workdir} && TIMELAPSE_ENABLE=1 {python} {script} build-daily-for-today
"""

SUPERVISOR_TEMPLATE = """\
[program:prey_touch_captures]
command={python} {script} capture-loop
directory={workdir}
user={user}
autostart=true
autorestart=true
stopsignal=INT
stdout_logfile={log_dir}/timelapse_capture.out.log
stderr_logfile={log_dir}/timelapse_capture.err.log
"""


def run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def ensure_cron(cron_lines: str) -> None:
    try:
        existing = subprocess.run(['crontab', '-l'], check=False, text=True, capture_output=True).stdout
    except FileNotFoundError:
        raise SystemExit('crontab command not found; install cron first.')

    existing = existing or ''
    new_lines = [line for line in cron_lines.strip().splitlines() if line not in existing]
    if not new_lines:
        print('Cron entries already present.')
        return

    updated = existing.rstrip('\n')
    if updated:
        updated += '\n'
    updated += '\n'.join(new_lines) + '\n'
    run(['crontab', '-'], input=updated)
    print('Installed cron entries:')
    for line in new_lines:
        print(f'  {line}')
def ensure_supervisor(conf_path: Path, python_path: str, script_path: Path, workdir: Path) -> None:
    log_dir = workdir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    conf_path.parent.mkdir(parents=True, exist_ok=True)
    user = os.environ.get('SUDO_USER') or os.environ.get('USER') or 'root'
    content = SUPERVISOR_TEMPLATE.format(
        python=python_path,
        script=script_path,
        workdir=workdir,
        user=user,
        log_dir=log_dir,
    )
    if conf_path.exists() and conf_path.read_text() == content:
        print(f'Supervisor config already up to date ({conf_path}).')
    else:
        conf_path.write_text(content)
        print(f'Wrote supervisor config to {conf_path}')

    supervisorctl = shutil.which('supervisorctl')
    if supervisorctl:
        subprocess.run([supervisorctl, 'reread'], check=False)
        subprocess.run([supervisorctl, 'update'], check=False)
        subprocess.run([supervisorctl, 'restart', 'prey_touch_captures'], check=False)
    else:
        print('supervisorctl not found; please reload supervisor manually.')


def main():
    parser = argparse.ArgumentParser(description='Configure timelapse capture and stitching automation.')
    parser.add_argument('--python', default=sys.executable, help='Python interpreter to use in commands')
    parser.add_argument('--script', default=str(Path(__file__).resolve().with_name('timelapse_recorder.py')),
                        help='Path to timelapse_recorder.py')
    parser.add_argument('--supervisor-conf', default='/etc/supervisor/conf.d/prey_touch_captures.conf',
                        help='Supervisor conf file to manage the capture loop')
    parser.add_argument('--workdir', default=str(Path(__file__).resolve().parents[1]),
                        help='Working directory for the capture loop (defaults to Arena/)')
    args = parser.parse_args()

    cron_text = CRON_TEMPLATE.format(python=args.python, script=args.script, workdir=args.workdir)
    ensure_cron(cron_text)
    ensure_supervisor(Path(args.supervisor_conf), args.python, Path(args.script), Path(args.workdir))
    print('Done. Use `crontab -l` and `supervisorctl status prey_touch_captures` to verify.')


if __name__ == '__main__':
    main()
