# Camera calibration parameters
# Generated: 2026-09-18 21:07:52

import numpy as np

# Camera intrinsic matrix
camera_matrix = np.array([[1099.8204832349145, 0.0, 620.2107800109218], [0.0, 1102.39635792446, 364.96244297900523], [0.0, 0.0, 1.0]], dtype=np.float32)

# Distortion coefficients
dist_coeffs = np.array([-0.08974087147046304, 0.18519511667334432, 0.001270897610207209, -0.004809800756176213, -0.13097425830186266], dtype=np.float32)

# Image size (width, height)
image_size = (1280, 720)

# Calibration info
reprojection_error = 1.9711  # pixels
num_calibration_images = 20
