# Camera Calibration and Real-World Transformation

## Calibrate Camera
1. To calibrate the camera you need to print a chessboard https://github.com/opencv/opencv/blob/4.x/doc/pattern.png
2. Place the printed chessboard inside the arena and take at least 10 images of the chessboard in different orientations from each camera that you want to calibrate.
3. Create a folder with the camera name in output/calibrations and put all the chessboard images there.
4. Cd into the Arena folder and run the following:
```console
python calibration.py --calibration --cam_name <cam_name>
```

## Transformation to real-world coordinates
1. Print a Charuco board that fits exactly the sizes of your arena. You can use the following script for creating such board:
https://github.com/opencv/opencv/blob/4.x/doc/pattern_tools/gen_pattern.py
2. Place the printed charuco board in the arena and take a frame of it from each camera that you'd like to run projection to real-world.
3. Create a folder "charuco_images" in output/calibrations and put the charuco images in there.
4. Run the following script from the Arena folder:
```console
python calibration.py --transformation --run_all_charuco
```