import pytest
from pathlib import Path
import importlib
import inspect
import config


def test_cam_packages_exist():
    cam_modules = set([cam_dict['module'] for cam_dict in config.cameras.values()])
    for cm in cam_modules:
        if cm == 'flir':
            import PySpin
        elif cm == 'allied_vision':
            import vimba