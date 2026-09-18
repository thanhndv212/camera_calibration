# Camera calibration parameters
# Generated: 2026-09-18 21:11:02

import numpy as np

# Camera intrinsic matrix
camera_matrix = np.array([[1101.6496096192534, 0.0, 638.5464084064187], [0.0, 1096.3643841020264, 381.97369221089497], [0.0, 0.0, 1.0]], dtype=np.float32)

# Distortion coefficients
dist_coeffs = np.array([-0.09098390095678721, 0.3587083588312438, 0.0009376263089576702, 0.0011597452240452019, -0.8461366269901047], dtype=np.float32)

# Image size (width, height)
image_size = (1280, 720)

# Calibration info
reprojection_error = 0.8188  # pixels
num_calibration_images = 20
