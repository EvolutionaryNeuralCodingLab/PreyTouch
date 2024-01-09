import cv2
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

THRESHOLD = 5


def get_differentiated_frames(video_path, n_frames=50, output_dir=None):
    """Extracts the most differentiated frames from a video.

    Args:
      video_path: The path to the video file.

    Returns:
      A list of frames that are the most differentiated from each other.
    """

    if output_dir:
        Path(output_dir).mkdir(exist_ok=True, parents=True)
    vid_name = Path(video_path).stem

    # Initialize the video capture object.
    cap = cv2.VideoCapture(video_path)

    # Initialize the list of differentiated frames.
    differentiated_frames = []
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Loop over the frames in the video.
    for frame_id in tqdm(range(total_video_frames), desc=vid_name):
        # Capture the next frame.
        ret, frame = cap.read()

        # If the frame is not None, then process it.
        if frame is not None:
            # Convert the frame to grayscale.
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if frame_id > 0:
                difference = cv2.absdiff(gray_frame, differentiated_frames[-1])
            else:
                difference = THRESHOLD + 1

            if np.mean(difference) > THRESHOLD:
                differentiated_frames.append(gray_frame)
                if output_dir:
                    cv2.imwrite(f'{output_dir}/{vid_name}_{frame_id}.jpg', frame)

        # If there are no more frames, then break out of the loop.
        if not ret or len(differentiated_frames) >= n_frames:
            break

    cap.release()

    # Return the list of differentiated frames.
    return differentiated_frames


if __name__ == "__main__":
    OUTPUT_DIR = '/data/Pogona_Pursuit/output/datasets/cutler/top_camera'

    for video_path_ in Path('/data/Pogona_Pursuit/output/experiments/PV91/20230609/tracking/').glob('*.mp4'):
        d_frames = get_differentiated_frames(video_path_.as_posix(), output_dir=OUTPUT_DIR)


#
# cd maskcut
# python maskcut.py \
# --vit-arch base --patch-size 8 \
# --tau 0.15 --fixed_size 480 --N 3 \
# --num-folder-per-job 1000 --job-index 0 \
# --dataset-path /data/Pogona_Pursuit/output/datasets/cutler/top_camera \
# --out-dir /data/Pogona_Pursuit/output/datasets/cutler/top_camera/annotations

# python demo.py --img-path /data/Pogona_Pursuit/output/datasets/cutler/top_camera/train/top_20230609T080020_7184.jpg \
#   --N 3 --tau 0.15 --vit-arch base --patch-size 8