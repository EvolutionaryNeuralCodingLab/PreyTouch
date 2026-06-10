import cv2
import os
import pandas as pd
import time
import config
from arrayqueues.shared_arrays import Full
from arena import Camera, ArenaException
from cache import RedisCache, CacheColumns as cc

# Ensure the GigE transport layer is on the path regardless of how the
# process was launched (supervisord may only have VimbaUSBTL configured).
_GIGETL_PATH = os.path.realpath(os.path.join(
    os.path.dirname(__file__), '..', 'bin', 'Vimba_6_0',
    'VimbaGigETL', 'CTI', 'x86_64bit'
))
_existing = os.environ.get('GENICAM_GENTL64_PATH', '')
if _GIGETL_PATH not in _existing:
    os.environ['GENICAM_GENTL64_PATH'] = _existing + ':' + _GIGETL_PATH

import vimba

cache = RedisCache()


class AlliedVisionCamera(Camera):

    def configure(self, cam):
        try:
            # Stop any leftover acquisition so configuration features are writable.
            # Also clear AcquisitionFrameRateEnable unconditionally first: on some
            # camera models =True conditionally locks TriggerMode, and it is also
            # not writable when other cameras are streaming on the same USB bus.
            try:
                cam.AcquisitionStop.execute()
            except Exception:
                pass
            try:
                cam.AcquisitionFrameRateEnable.set(False)
            except Exception:
                pass
            cam.ExposureAuto.set('Off')
            cam.ExposureMode.set('Timed')
            cam.ExposureTime.set(self.cam_config['exposure'])
            cam.DeviceLinkThroughputLimit.set(4e8)
            self.logger.debug(f'Throughput: {cam.DeviceLinkThroughputLimit.get():.0e}')
            if self.cam_config.get('reverse_y'):
                cam.ReverseY.set('true')
            if self.cam_config.get('pixel_format'):
                try:
                    cam.set_pixel_format(getattr(vimba.PixelFormat, self.cam_config['pixel_format']))
                except Exception as e:
                    self.logger.warning(f'Could not set pixel format {self.cam_config["pixel_format"]}: {e}')
            trigger_source = self.cam_config.get('trigger_source')
            fps = self.cam_config.get('fps')
            if trigger_source and fps:
                raise ArenaException('must provide either fps or trigger_source')

            if trigger_source:
                cam.TriggerMode.set('Off')
                cam.TriggerSelector.set('FrameStart')
                cam.LineSelector.set(trigger_source)
                cam.LineMode.set('Input')
                cam.TriggerSource.set(trigger_source)
                cam.TriggerActivation.set('RisingEdge')
                cam.TriggerMode.set('On')
                self.logger.debug(f'configured trigger source to: {trigger_source}')
            elif fps:
                cam.TriggerMode.set('Off')
                try:
                    cam.AcquisitionFrameRateEnable.set(True)
                except Exception:
                    self.logger.warning('AcquisitionFrameRateEnable is read-only on this camera, skipping')
                requested_fps = float(fps)
                _, max_fps = cam.AcquisitionFrameRate.get_range()
                actual_fps = min(requested_fps, max_fps)
                if actual_fps < requested_fps:
                    self.logger.warning(f'Requested fps {requested_fps} exceeds camera max {max_fps:.1f}, clamping to {actual_fps:.1f}')
                cam.AcquisitionFrameRate.set(actual_fps)
                self.logger.debug(f'configured fps to: {actual_fps:.1f}')
            else:
                raise ArenaException('bad configuration. must provide either trigger_source or fps in cam_config')

            cam.AcquisitionMode.set('Continuous')
            self.logger.debug('Finish configuration')
        except Exception as exc:
            self.logger.error(f"Exception while configuring camera: {exc}")

    def _run(self):
        try:
            system = vimba.Vimba.get_instance()
            with system as v:
                cam_id = self.cam_config['id']
                cam = v.get_camera_by_id(cam_id)
                with cam:
                    try:
                        self.configure(cam)
                        self.update_time_delta(cam)
                        self.logger.debug('start streaming')
                        cache.append_to_list(cc.RECORDING_CAMERAS, self.cam_name)
                        cam.start_streaming(self._frame_handler, buffer_count=10)
                        self.stop_signal.wait()
                        cache.remove_from_list(cc.RECORDING_CAMERAS, self.cam_name)
                        if self.stop_signal.is_set():
                            self.logger.debug('received stop event')
                    except KeyboardInterrupt:
                        pass
                    finally:
                        self.mp_metadata['cam_fps'].value = 0.0
                        if cam.is_streaming():
                            cam.stop_streaming()
        except vimba.error._LoggedError as exc:
            raise ArenaException(str(exc))

    def _frame_handler(self, cam, frame):
        t0 = time.time()
        waiting_time = 0.1
        if self.stop_signal.is_set():
            return
        try:
            while True:
                try:
                    # Convert Bayer frames to BGR so downstream code stays unchanged.
                    # This saves ~3x USB bandwidth vs sending pre-decoded color.
                    pixel_fmt = frame.get_pixel_format()
                    bayer_formats = {
                        vimba.PixelFormat.BayerRG8: cv2.COLOR_BayerRG2BGR,
                        vimba.PixelFormat.BayerGB8: cv2.COLOR_BayerGB2BGR,
                        vimba.PixelFormat.BayerGR8: cv2.COLOR_BayerGR2BGR,
                        vimba.PixelFormat.BayerBG8: cv2.COLOR_BayerBG2BGR,
                    }
                    if pixel_fmt in bayer_formats:
                        img = cv2.cvtColor(frame.as_numpy_ndarray(), bayer_formats[pixel_fmt])
                    else:
                        img = frame.as_numpy_ndarray()
                    timestamp = frame.get_timestamp() / 1e9 + self.camera_time_delta
                    if not self.is_color_cam():
                        img = img.squeeze()
                    self.frames_queue.put(img, timestamp)
                    self.calc_fps(time.time())
                    break
                except Full:
                    if (time.time() - t0) > waiting_time:
                        if not self.last_queue_warning_time or (time.time() - self.last_queue_warning_time > 60):
                            self.logger.warning(f'Queue is still full after waiting {waiting_time}')
                            self.last_queue_warning_time = time.time()
                        break
        except Exception as exc:
            self.logger.error(f"Exception while getting image from alliedVision camera: {exc}")
        finally:
            cam.queue_frame(frame)


def init():
    info_df = scan_cameras()
    return {cam_name: [AlliedVisionCamera] for cam_name in info_df.index if cam_name != 'unknown'}


def scan_cameras(is_print=True) -> pd.DataFrame:
    system = vimba.Vimba.get_instance()
    cam_names = []
    with system as v:
        cams = v.get_all_cameras()
        if is_print:
            print('AlliedVision cameras found: {}'.format(len(cams)))
        info = []
        for cam in cams:
            info.append(get_cam_info(cam))
            cam_name = 'unknown'
            for n, cam_config in config.cameras.items():
                if cam_config['id'] == cam.get_id():
                    cam_name = n
                    break
            cam_names.append(cam_name)
        info = pd.DataFrame(info, index=cam_names)

    return info


def get_cam_info(cam):
    info = dict()
    info['Camera Name'] = cam.get_name()
    info['Model Name'] = cam.get_model()
    info['Camera ID'] = cam.get_id()
    info['Serial Number'] = cam.get_serial()
    # info['Interface ID'] = cam.get_interface_id()
    return info
