# 3D-stereo-inertial-odometry
Implementation of EKF fusing IMU measurements with stereo images to estimate the 3D pose of an autonomous vehicle. The script can be used on any drive from the KITTI benchmark suite. A full write-up of the implementation is attached above.


## Requirements
- Matlab Computer Vision Toolbox
- SE3 Matlab Tools (http://asrl.utias.utoronto.ca/code/index.html)
- KITTI Benchmark Suite Raw Data (http://www.cvlibs.net/datasets/kitti/raw_data.php)
  - Synced and Rectified grayscale images, GPS/IMU Data, raw data devkit, calibration .txt files
