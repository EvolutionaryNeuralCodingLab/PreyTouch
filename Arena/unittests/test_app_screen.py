import pytest
import os
import config
import logging


def test_xinput_cli():
    rc = os.system('xinput > /dev/null')
    if rc == 127:
        print(f'xinput cli not found; Disabling app screen; rc={rc}')
        config.DISABLE_APP_SCREEN = True


def test_app_screen_exists():
    rc = os.system(f'DISPLAY="{config.APP_SCREEN}" xinput 2> /dev/null | grep -i "{config.TOUCH_SCREEN_NAME}"')
    if rc != 0:
        print(f'Screen {config.TOUCH_SCREEN_NAME} was not found in DISPLAY {config.APP_SCREEN}. Working without app screen; rc={rc}')
        config.DISABLE_APP_SCREEN = True
