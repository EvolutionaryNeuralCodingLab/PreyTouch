import os
import config


def test_xinput_cli():
    rc = os.system('xinput > /dev/null')
    if rc != 0:
        print(f'xinput not found')
        config.DISABLE_APP_SCREEN = True


def test_app_screen_exists():
    rc = os.system(f'DISPLAY="{config.APP_SCREEN}" xinput | grep -i "{config.TOUCH_SCREEN_NAME}"')
    if rc != 0:
        print(f'Screen {config.TOUCH_SCREEN_NAME} was not found in DISPLAY {config.APP_SCREEN}')
        config.DISABLE_APP_SCREEN = True
