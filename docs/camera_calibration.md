# Camera Calibration and Real-World Transformation

## Calibrate Camera
This process provides the intrinsic matrix of the camera for fixing distortions made by lenses.
1. To calibrate the camera you need to print a chessboard https://github.com/opencv/opencv/blob/4.x/doc/pattern.png. The default size for the chessboard in PreyTouch is 9x6. In case you want to use a different board, you have to specify it's size in Settings > Calibration > CHESSBOARD_DIM
2. Place the printed chessboard inside the arena and take at least 10 images of the chessboard in different orientations from the camera that you want to calibrate, using the Capture button.
3. From within PreyTouch UI, click the "Calibration" button, and then click on "Undistort Camera".
4. Write down the camera name, the date in which the chessboard images were taken and choose all images with "Choose Files" and press "Run".
5. In case the process succeeded, you should see all chessboard images with their undistorted version.
6. Go through the undistorted images and verify that the relevant areas within the frame are less "rounded". If the undistortion isn't well, capture more images of the chessboard and re-run the process.
![undistort Image](/docs/images/undistortion_result.png)

## Transformation to real-world coordinates
1. Print a Charuco board that fits exactly the sizes of your arena. You can use the following script for creating such board:
https://github.com/opencv/opencv/blob/4.x/doc/pattern_tools/gen_pattern.py
![Charuco Image](/docs/images/charuco.png)
2. Place the printed charuco board in the arena and take a frame of it from the camera that you'd like to run projection to real-world.
3. From within PreyTouch UI, click the "Calibration" button, and then click on "Real World Projection".
4. Write down the camera name, the date in which the Charuco image was taken and upload it using "Choose Files" and press "Run".
5. If the process is succeeded, you should see: (1) List of all detected markers with
their expected and projected real-world coordinates, alongside the distance in cm between the 2 dots.
(2) image with all the detected Charuco markers and 9 other random dots 
on the frame with their projection. Go through the projected coordinates of the 
dots and verify that they make sense. If not, capture a new Charuco image and 
re-run this process. If that doesn't help either, then you probably have to 
re-run the undistortion process. 
![Charuco Image](/docs/images/realworld_projection_result.png)