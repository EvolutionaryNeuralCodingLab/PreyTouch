import json
import re
import cv2
import sys
import pickle
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from scipy.spatial import distance
import config
import utils
from loggers import get_logger

ARUCO_DICT = cv2.aruco.DICT_4X4_1000
ARUCO_IDS = np.arange(config.NUM_ARUCO_MARKERS).tolist()
CHARUCO_COLS = 8
DATE_FORMAT = '%Y%m%dT%H%M%S'

class CalibrationError(Exception):
    """"""


class Calibrator:
    def __init__(self, cam_name, resize_dim=None):
        self.cam_name = cam_name
        self.logger = get_logger('calibrator')
        self.calib_params = {
            'mtx': None,
            'dist': None,
            'rvecs': None,
            'tvecs': None,
            'w': None,
            'h': None
        }
        self.newcameramtx = None
        self.resize_dim = resize_dim
        self.calib_images_date = None
        self.current_undistort_folder = Path(config.CALIBRATION_DIR) / 'undistortion' / self.cam_name
        if not self.current_undistort_folder.exists():
            self.current_undistort_folder.mkdir(parents=True, exist_ok=True)
        self.load_calibration()  # load the latest undistortion calibration

    def calibrate_camera(self, is_plot=True, img_dir=None, calib_date=None):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((np.prod(config.CHESSBOARD_DIM), 3), np.float32)
        objp[:, :2] = np.mgrid[:config.CHESSBOARD_DIM[0], :config.CHESSBOARD_DIM[1]].T.reshape(-1, 2)

        self.set_calib_date(calib_date)
        img_files = self.get_calib_images(img_dir)

        self.logger.info(f'start camera {self.cam_name} calibration with {len(img_files)} images')
        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        detected_frames = []
        for i, p in enumerate(img_files):
            gray = cv2.imread(p.as_posix(), 0)
            if self.resize_dim:
                gray = cv2.resize(gray, self.resize_dim)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, config.CHESSBOARD_DIM, None)
            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                corners_ = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners_)
                detected_frames.append((p, corners_))

        h, w = gray.shape[:2]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
        if not ret:
            raise CalibrationError('calibrateCamera failed')
        
        # save calibration params to class
        for k in self.calib_params.copy().keys():
            self.calib_params[k] = locals().get(k)

        self.save_calibration()

        if is_plot:
            self.plot_undistorted(detected_frames)

        self.logger.info('calibration finished successfully')
        err_text = self.calc_projection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
        return err_text

    def set_calib_date(self, calib_date=None):
        if calib_date is None:
            calib_date = datetime.now()
        
        self.calib_images_date = calib_date
        self.load_calibration()

    def plot_undistorted(self, detected_frames):
        cols, rows = 2, len(detected_frames)
        fig, axes = plt.subplots(rows, cols, figsize=(30, 10 * rows))
        for i, (p, corners) in enumerate(detected_frames):
            gray = cv2.imread(p.as_posix(), 0)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            cv2.drawChessboardCorners(frame, config.CHESSBOARD_DIM, corners, True)
            axes[i, 0].imshow(frame)
            axes[i, 0].axis('off')

            undistorted_img = self.undistort_image(gray)
            axes[i, 1].imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_GRAY2RGB))
            axes[i, 1].axis('off')
        
        fig.tight_layout()
        fig.savefig(self.calib_results_image_path, bbox_inches='tight')
        plt.close(fig)

    def undistort_image(self, img) -> np.ndarray:
        return cv2.undistort(img, self.calib_params['mtx'], self.calib_params['dist'], None, self.newcameramtx)

    def calc_undistort_mappers(self):
        mtx, dist = self.calib_params['mtx'], self.calib_params['dist']
        w, h = self.calib_params['w'], self.calib_params['h']
        self.newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    def calc_projection_error(self, objpoints, imgpoints, rvecs, tvecs, mtx, dist):
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        err_text = "total projection error: {}".format(mean_error / len(objpoints))
        return err_text

    def load_calibration(self):
        last_path = get_last_artifact(self.current_undistort_folder, self.cache_prefix, self.calib_images_date)
        if not last_path:
            self.logger.warning('Could not load undistortion; no artifacts found')
            return

        with open(last_path, 'rb') as f:
            self.calib_params = pickle.load(f)
        self.calc_undistort_mappers()

    def save_calibration(self):
        self.logger.info(f'saving calibration to {self.calib_params_path}')
        with self.calib_params_path.open('wb') as f:
            pickle.dump(self.calib_params, f)

    def get_calib_images(self, calib_dir=None):
        calib_dir = Path(calib_dir) or (Path(config.CALIBRATION_DIR) / self.cam_name)
        if not calib_dir.exists():
            raise CalibrationError(f'{calib_dir} not exist')
        img_files = list(calib_dir.glob('*.png'))
        if len(img_files) < config.MIN_CALIBRATION_IMAGES:
            raise CalibrationError(f'found only {len(img_files)} images for calibration, '
                                   f'expected {config.MIN_CALIBRATION_IMAGES}')
        else:
            self.logger.info(f'found {len(img_files)} images for calibration in {calib_dir}')
        return img_files

    @property
    def cache_prefix(self):
        return f'undistortion_{json.dumps(self.resize_dim)}'

    @property
    def calib_params_path(self):
        return self.current_undistort_folder / f'{self.cache_prefix}_{self.calib_images_date.strftime(DATE_FORMAT)}.pkl'

    @property
    def calib_results_image_path(self):
        return self.current_undistort_folder / f'calib_detections_{self.calib_images_date.strftime(DATE_FORMAT)}.jpg'


class CharucoEstimator:
    def __init__(self, cam_name, resize_dim=None, logger=None, is_debug=True, is_undistort=False):
        self.cam_name = cam_name
        self.resize_dim = resize_dim
        self.resize_scale = None
        self.is_debug = is_debug
        self.is_undistort = is_undistort
        # loaded from cache
        self.id_key = 'id'
        self.cached_params = ['homography']

        self.current_realworld_folder = Path(config.CALIBRATION_DIR) / 'real_world' / self.cam_name
        if not self.current_realworld_folder.exists():
            self.current_realworld_folder.mkdir(parents=True, exist_ok=True)
        self.calibrator = Calibrator(cam_name, resize_dim=resize_dim)
        self.mtx = self.calibrator.calib_params['mtx']
        self.dist = self.calibrator.calib_params['dist']
        self.newcam_mtx = self.calibrator.newcameramtx

        self.arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT)
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.state = 0  # 0 - not initiated, 1 - failure, 2 - initiated
        self.markers_image_date = None
        self.image_date = None
        self.logger = get_logger('calibration-pose-estimator') if logger is None else logger

    def __str__(self):
        return self.id_key

    def init(self, img, img_shape=None, is_plot=False):
        try:
            if img_shape is None:
                img_shape = img.shape[:2]
            if self.resize_dim:
                self.resize_scale = (img.shape[0] // self.resize_dim[0], img.shape[1] // self.resize_dim[1])
                if json.dumps(img_shape) != json.dumps(self.resize_dim):
                    raise Exception(f'Image size does not fit. expected: {tuple(self.resize_dim)}, received: {tuple(img_shape)}')

            self.set_image_date_and_load(self.image_date)
            self.state = 2
            if self.is_debug:
                self.logger.info(f'started real-world-coordinates transformer for frames of shape: {img_shape}')
            if is_plot:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = self.plot_calibrated_line(img)
                plt.imshow(img)
                plt.show()
        except Exception as exc:
            if self.is_debug:
                self.logger.error(f'Error in pose-estimator init; {exc}')
            self.state = 1
        return self

    def get_location(self, frame_x, frame_y, is_undistort=True, check_init=True):
        if check_init and not self.is_initiated:
            return
        if self.resize_scale:
            frame_x, frame_y = self.resize_scale[0] * frame_x, self.resize_scale[1] * frame_y
        if is_undistort:
            uv = cv2.undistortPoints(np.array([frame_x, frame_y]).astype('float32'), self.mtx, self.dist, None, self.newcam_mtx).ravel()
        else:
            uv = [frame_x, frame_y]
        uv_1 = np.array([[*uv,1]], dtype=np.float32).T

        result = np.dot(self.homography, uv_1).ravel()
        projected_x = result[0] / result[2]
        projected_y = result[1] / result[2]
        return projected_x, projected_y

    def set_image_date_and_load(self, image_date=None):
        last_path = get_last_artifact(self.current_realworld_folder, self.cache_prefix, image_date)
        if not last_path:
            raise CalibrationError(f'Could not find aruco image for date {image_date}')
        self.calibrator.set_calib_date(image_date)
        self.load_transformation(last_path)

    def load_aruco_image(self, image_path, image_date=None):
        if image_date is None:
            assert re.match(r'\d{8}T\d{6}_\w+', Path(image_path).stem), f'Image filename must be of the pattern {DATE_FORMAT}_<cam_name>.png'
            assert Path(image_path).exists(), f'Image path does not exist: {image_path}'
            self.markers_image_date = Path(image_path).stem.split('_')[0]
        else:
            if isinstance(image_date, datetime):
                image_date = image_date.strftime(DATE_FORMAT)
            self.markers_image_date = image_date

        frame = cv2.imread(image_path)
        frame = self.calibrator.undistort_image(frame)
        if frame.shape[2] > 1:
            gray_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            gray_frame = frame.copy()
            frame = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2RGB)
        return frame, gray_frame

    def find_aruco_markers(self, image_path, image_date=None, is_plot=True, is_rotated=False):
        frame, gray_frame = self.load_aruco_image(image_path, image_date)
        self.logger.info(f'Start Aruco marker detection for image size: {frame.shape}')
        # detect Charuco markers
        marker_corners, marker_ids, rejected = cv2.aruco.detectMarkers(gray_frame, self.arucoDict, parameters=self.arucoParams)
        if marker_ids is None:
            raise Exception('Could not find aruco markers')
        # sort markers ids
        marker_corners = [marker_corners[i] for i in marker_ids.ravel().argsort()]
        marker_ids.sort(axis=0)
        marker_ids, marker_corners = self.validate_detected_markers(marker_ids, marker_corners)
        # create dataset for PnP using aruco top-left (in regular and bottom-right in roatated board) corner detections
        image_points_2D = np.vstack([m[0][0 if not is_rotated else 2] for m in marker_corners]).astype('float32')
        real_world_points_3D = self.get_real_world_points(marker_ids, is_rotated=is_rotated).astype('float32')
        # estimate homography matrix
        self.homography = self.estimate_homography(image_points_2D, real_world_points_3D[:, :2])
        err_text = self.print_detection_error(image_points_2D, real_world_points_3D, marker_ids)

        self.id_key = utils.datetime_string()
        self.save_transformation()
        if is_plot:
            frame = self.plot_aruco_detections(frame, marker_ids, marker_corners, real_world_points_3D)
            cv2.imwrite(self.detected_markers_image_path.as_posix(), frame)
        return frame, err_text

    @staticmethod
    def estimate_homography(image_points, dest_points):
        """Find homography matrix."""
        fp = np.array(image_points)
        tp = np.array(dest_points)
        H, _ = cv2.findHomography(fp, tp, 0)
        return H

    def print_detection_error(self, image_points_2D, real_world_points_3D, marker_ids):
        dists = []
        text = 'Projection Errors: (expected <-> projected = distance [cm]) \n'
        for i, image_point, dest_point in zip(marker_ids, image_points_2D, real_world_points_3D):
            x, y = self.get_location(*image_point, is_undistort=False, check_init=False)
            d = distance.euclidean((x, y), dest_point[:2])
            text += f'{i} {tuple(dest_point[:2].tolist())} <-> ({x:.1f}, {y:.1f}) = {d:.2f} cm\n'
            dists.append(d)
        text += f'Average error: {np.mean(dists):.2f} cm'
        print(text)
        return text

    def validate_detected_markers(self, marker_ids, marker_corners):
        marker_ids = marker_ids.copy().ravel()
        min_markers_amount = 60
        if len(marker_ids) < min_markers_amount:
            raise Exception(f'Not enough detected markers: {len(marker_ids)} < {min_markers_amount}')
        
        missing_aruco = [m for m in ARUCO_IDS if m not in marker_ids]
        print(f'The following markers were not detected: {missing_aruco}')

        # remove marker ids above the configures number
        bad_marker_ids = np.where(marker_ids >= config.NUM_ARUCO_MARKERS)[0].tolist()
        if bad_marker_ids:
            marker_ids = np.delete(marker_ids, bad_marker_ids)
            marker_corners = [m for i, m in enumerate(marker_corners) if i not in bad_marker_ids]

        return marker_ids, marker_corners

    @staticmethod
    def get_real_world_points(marker_ids, n=config.NUM_ARUCO_MARKERS, is_rotated=False) -> pd.DataFrame:
        df = []
        is_even_row = not is_rotated  # regularilly 1st row is even, but if rotated it's odd
        col = 0
        row = 0
        aruco_outer_size = 2 * config.ARUCO_MARKER_SIZE
        iter = range(n) if not is_rotated else range(n-1, -1, -1)
        for i in iter:
            df.append({'marker_id': i, 'row': row, 'col': col,
                        'x': (2 * col + 1 if not is_even_row else 2 * col) * aruco_outer_size,
                        'y': row * aruco_outer_size})
            col += 1
            if (is_even_row and not (col % CHARUCO_COLS)) or (not is_even_row and not (col % (CHARUCO_COLS - 1))):
                row += 1
                is_even_row = not is_even_row
                col = 0
                
        df = pd.DataFrame(df).set_index('marker_id')
        df['z'] = 0
        df = df.loc[marker_ids.ravel()][['x', 'y', 'z']].values
        return df

    def plot_aruco_detections(self, frame, marker_ids, marker_corners, real_world_points_3D):
        font, line_type, font_size = cv2.FONT_HERSHEY_PLAIN, cv2.LINE_AA, 1.8
        for i, (marker_id, corners) in enumerate(zip(marker_ids.ravel(), marker_corners)):
            cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, line_type)
            top_right, top_left, _, _ = self.flatten_corners(corners)
            cv2.circle(frame, top_left, 2, (255, 0, 255), 3)
            # real_x, real_y = real_world_points_3D[i, :2]
            # real_label = f'{marker_id} ({real_x:.1f},{real_y:.1f})'
            real_label = f'{marker_id}'
            cv2.putText(frame, real_label, top_right, font, 1.5, (255, 255, 255), 7, line_type)
            cv2.putText(frame, real_label, top_right, font, 1.5, (255, 0, 255), 3, line_type)
        frame = self.plot_calibrated_line(frame)
        return frame

    def plot_calibrated_line(self, frame, color=(218, 165, 32)):
        font, line_type, font_size = cv2.FONT_HERSHEY_PLAIN, cv2.LINE_AA, 1.8
        # plot center image coord for checking the get_location algorithm
        h, w = frame.shape[:2]
        for frame_pos in [(w // 4, h // 3), (w // 4, h // 2), (w // 4, round(h / 1.3)),
                          (w // 2, h // 3), (w // 2, h // 2), (w // 2, round(h / 1.3)),
                          (3*w // 4, h // 3), (3*w // 4, h // 2), (3*w // 4, round(h / 1.3))]:
            xc, yc = self.get_location(*frame_pos, is_undistort=False, check_init=False)
            cv2.circle(frame, frame_pos, 4, color, 3)
            cv2.putText(frame, f"({xc:.1f},{yc:.1f})", frame_pos, font, 2, (255, 255, 255), 8, line_type)
            cv2.putText(frame, f"({xc:.1f},{yc:.1f})", frame_pos, font, 2, (0, 0, 0), 4, line_type)
        return frame

    def flatten_corners(self, corners):
        corners = corners.reshape(4, 2).astype(int)
        top_left = corners[0].ravel()
        top_right = corners[1].ravel()
        bottom_right = corners[2].ravel()
        bottom_left = corners[3].ravel()
        return top_right, top_left, bottom_right, bottom_left

    def load_transformation(self, cache_file_path: Path=None):
        if not cache_file_path.exists():
            raise Exception(f'Aruco cache file {cache_file_path} does not exist')

        with cache_file_path.open('rb') as f:
            cache = pickle.load(f)
            for k in self.cached_params:
                setattr(self, k, cache[k])

        if self.is_debug:
            self.logger.info(f'Loaded cached transformation from {cache_file_path}')

    def save_transformation(self):
        with self.cache_path.open('wb') as f:
            pickle.dump({k: getattr(self, k) for k in self.cached_params}, f)

    @property
    def cache_prefix(self):
        return f'aruco_transformation_{json.dumps(self.resize_dim)}'

    @property
    def cache_path(self):
        stem = f'{self.cache_prefix}_{self.markers_image_date}'
        return self.current_realworld_folder / f'{stem}.pkl'

    @property
    def detected_markers_image_path(self):
        return self.current_realworld_folder / f'detected_{self.cache_prefix}_{self.markers_image_date}.jpg'

    @property
    def is_initiated(self):
        return self.state > 0

    @property
    def is_on(self):
        return self.state == 2


def get_last_artifact(search_folder, cache_prefix, image_date=None):
    if image_date and isinstance(image_date, str):
        assert re.match(r'\d{8}T\d{6}', image_date), f'Image date must be of the form {DATE_FORMAT}'
        image_date = datetime.strptime(image_date, DATE_FORMAT)

    last_date, last_path = None, None
    for p in Path(search_folder).glob(f'{cache_prefix}*.pkl'):
        sp = p.stem.split('_')
        p_date = sp[-1]
        p_date = datetime.strptime(p_date, DATE_FORMAT)
        if image_date and p_date > image_date:
            continue
        
        if last_date is None or p_date > last_date:
            last_date = p_date
            last_path = p
    return last_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration', action='store_true', help='Run camera calibration based on detection of chessboard images')
    parser.add_argument('--transformation', action='store_true', help='Calculate the transformation between real world and image coordinates, based on Charuco markers')
    parser.add_argument('--cam_name', type=str, help='Camera name')
    parser.add_argument('--charuco_img_path', type=str, help='Path to charuco markers image')
    parser.add_argument('--undistort_frame', type=str, help='Run undistort on frame')
    args = parser.parse_args()

    if args.undistort_frame:
        assert args.cam_name, 'Please specify camera name'
        img = cv2.imread(args.undistort_frame)
        ce = Calibrator(args.cam_name)
        frame = ce.undistort_image(img)
        plt.imshow(frame)
        plt.show
        sys.exit(0)

    assert args.calibration ^ args.transformation, 'Please specify either calibration or transformation'
    if args.transformation:
        assert bool(args.charuco_img_path) ^ args.run_all_charuco, 'Please specify either charuco_img_path or run_all_charuco'
        assert args.cam_name, 'Please specify camera name'
        ce = CharucoEstimator(args.cam_name)
        ce.find_aruco_markers(args.charuco_img_path)

    else:  # calibration
        assert args.cam_name, 'Please specify camera name'
        calibrator = Calibrator(args.cam_name)
        calibrator.calibrate_camera()


if __name__ == "__main__":
    main()
