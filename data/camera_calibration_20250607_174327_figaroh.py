# Camera calibration parameters for FIGAROH
# Generated on: 2025-06-07 17:43:27

import numpy as np

# Camera intrinsic matrix
camera_matrix = np.array([[1101.4773366334907, 0.0, 642.6644842801771], [0.0, 1102.9541640653897, 380.20234515849927], [0.0, 0.0, 1.0]], dtype=np.float32)

# Distortion coefficients
dist_coeffs = np.array([-0.0846105588575387, 0.14179204930913328, 0.001127442817630244, -0.0017247298769922544, -0.06517732350331334], dtype=np.float32)

# Image size (width, height)
image_size = (1280, 720)

# Calibration info
reprojection_error = 0.3922  # pixels
num_calibration_images = 25
