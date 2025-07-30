# Code generated with Microsoft Copilot
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning
from PIL import Image
import cv2
from matplotlib.patches import Ellipse

# physical camera properties you may need to change
# https://telescopicwatch.com/meade-8-lx200-acf-review/ for focal length
focal_length = 3048.0 # 2032.0 #mm on 8 inch LX200 ACF
# https://www.celestron.com/products/neximage-5-solar-system-imager-5mp?srsltid=AfmBOopld100CNHs0HJw5GjLfA6yfL3MUhcrFf0_J6KfTKlHphuWg0kR#specifications
pixel_size = 2.2 #2.2 micron square on celestron nexImage 5, with 2x2 binning on lower resolution?
plate_scale = 206.265 *pixel_size/focal_length  # arcsec/pixel


# Penn State Brand Colors
nittany_navy = "#001E44"
beaver_blue = "#1E407C"
pugh_blue = "#96BEE6"
white = "#FFFFFF"
land_grant = "#6A3028"
penns_forest = "#4A7729"
futures_calling = "#99CC00"


# Load and convert image to grayscale
image = Image.open("EarlyTest.png").convert("L")
image_np = np.array(image)

# Threshold and crop around the dot
_, thresh = cv2.threshold(image_np, 30, 255, cv2.THRESH_BINARY)
coords = cv2.findNonZero(thresh)

padding = 10
x, y, w, h = cv2.boundingRect(coords)

# Ensure we don't go out of bounds
x_start = max(x - padding, 0)
y_start = max(y - padding, 0)
x_end = min(x + w + padding, image_np.shape[1])
y_end = min(y + h + padding, image_np.shape[0])

cropped = image_np[y_start:y_end, x_start:x_end]


# Define 2D Gaussian function
def gaussian_2d(xy, amp, x0, y0, sigma_x, sigma_y, offset):
    x, y = xy
    return offset + amp * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2)))

# Prepare data for fitting
X, Y = np.meshgrid(np.arange(cropped.shape[1]), np.arange(cropped.shape[0]))
initial_guess_2d = (cropped.max(), cropped.shape[1]//2, cropped.shape[0]//2, 3, 3, cropped.min())
popt_2d, _ = curve_fit(gaussian_2d, (X.ravel(), Y.ravel()), cropped.ravel(), p0=initial_guess_2d)
amp, x0, y0, sigma_x, sigma_y, offset = popt_2d


# Extract parameters and compute FWHM in arcseconds
amp, x0, y0, sigma_x, sigma_y, offset = popt_2d
fwhm_x_arcsec = 2.355 * sigma_x * plate_scale
fwhm_y_arcsec = 2.355 * sigma_y * plate_scale

# Generate fitted 2D image and residuals
fitted_2d = gaussian_2d((X, Y), *popt_2d).reshape(cropped.shape)
residuals_2d = cropped - fitted_2d

# Extract central horizontal line for 1D fit
central_row = cropped[cropped.shape[0] // 2, :]
x_data = np.arange(len(central_row))
y_data = central_row

# Define 1D Gaussian
def gaussian_1d(x, amp, x0, sigma, offset):
    return offset + amp * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

# Fit 1D Gaussian
initial_guess_1d = (y_data.max(), len(y_data) // 2, 10, y_data.min())
try:
    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        popt_1d, _ = curve_fit(gaussian_1d, x_data, y_data, p0=initial_guess_1d, bounds=([0, 0, 0, 0], [np.inf, len(x_data), np.inf, np.inf]))
except (RuntimeError, OptimizeWarning):
    # Fallback: try a different initial guess
    initial_guess_2 = (y_data.max() - y_data.min(), np.argmax(y_data), len(y_data)/5, y_data.min())
    try:
        popt_1d, _ = curve_fit(gaussian_1d, x_data, y_data, p0=initial_guess_2, bounds=([0, 0, 0, 0], [np.inf, len(x_data), np.inf, np.inf]))
    except Exception as e:
        print(f"Fit failed completely: {e}")
        popt_1d = None

amp1d, x01d, sigma1d, offset1d = popt_1d
fwhm_1d_arcsec = 2.355 * sigma1d * plate_scale
fitted_1d = gaussian_1d(x_data, *popt_1d)

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Cropped image
axs[0].imshow(cropped, cmap='gray')
axs[0].set_title("Cropped Image", color = nittany_navy)
axs[0].axis('off')


# 1D Gaussian fit with FWHM lines
axs[1].plot(x_data, y_data, label="Data", color=beaver_blue)
axs[1].plot(x_data, fitted_1d, label="1D Gaussian Fit", color=land_grant, linestyle='--')
left_fwhm = x01d - 0.5 * 2.355 * sigma1d
right_fwhm = x01d + 0.5 * 2.355 * sigma1d
axs[1].axvline(left_fwhm, color=penns_forest, linestyle='--', label='FWHM')
axs[1].axvline(right_fwhm, color=penns_forest, linestyle='--')
axs[1].set_title(f"1D Fit (FWHM: {fwhm_1d_arcsec:.2f}\" )", color = nittany_navy)
axs[1].set_xlabel("Pixel", color = nittany_navy)
axs[1].set_ylabel("Intensity", color = nittany_navy)
axs[1].legend()
axs[1].grid(True)

# 2D residuals with FWHM ellipse
axs[2].imshow(residuals_2d, cmap='bwr')
ellipse = Ellipse((x0, y0), width=2.355*sigma_x*2, height=2.355*sigma_y*2,
                  edgecolor='lime', facecolor='none', lw=2)
axs[2].add_patch(ellipse)
axs[2].set_title(f"2D Residuals\nFWHM X: {fwhm_x_arcsec:.2f}\"  |  FWHM Y: {fwhm_y_arcsec:.2f}\"", color = nittany_navy)
axs[2].axis('off')

plt.tight_layout()
plt.show()

