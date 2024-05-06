import pytest
import os
from pathlib import Path
import config


def test_output_dir_exists():
    assert Path(config.OUTPUT_DIR).exists(), f'Output directory {config.OUTPUT_DIR} does not exist'


def test_output_dir_writable():
    assert os.access(config.OUTPUT_DIR, os.W_OK), f'Output directory {config.OUTPUT_DIR} is not writable'


def test_output_subdirectory():
    """check if all the sub-dirs exist, if not create them"""
    subdirs = ['calibrations', 'captures', 'models', 'experiments', 'events', 'datasets']
    output_dir = Path(config.OUTPUT_DIR)
    for sd in subdirs:
        sd = output_dir / sd
        if not sd.exists():
            sd.mkdir()

