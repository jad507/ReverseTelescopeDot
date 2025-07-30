import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image
from scipy.optimize import curve_fit
plt.ion() 

def gauss(x, mu, sigma, amp):
    return(amp/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2*sigma**2)))

focal_length = 3048.0 # mm
pixel_size = 2.2E-3 # 2.2 um pitch

plate_scale = 206265.0/focal_length # arcseoncds/mm
plate_scale *= pixel_size



img = np.array(Image.open('1-60.bmp'))
plt.figure(1)
plt.clf()
plt.imshow(img)
plt.show()
#img2 = plt.imread('/Users/houndaigen/Desktop/images/30_micon_psf3.bmp')

green = img[:,:,1] #pull out the green data
#calculate the mean of image
green_mean = np.mean(green)
image_background = green - green_mean
q = np.where(image_background <= 3.0*np.std(image_background))
image_threshold = image_background.copy()
image_threshold[q] = 0

# plot the green image
plt.figure(2)
plt.clf()
plt.imshow(image_threshold)
plt.show()


#flatten the image and make a histogram
green_flat = np.sum(image_threshold,axis=0)
plt.figure(3)
plt.clf()
plt.title('1D Pixel Histogram')
plt.hist(green_flat, bins=100)
plt.show()

# average_background = np.mean(green_flat)

# subtract the background
# background_flat = green_flat - average_background

# fit the spot 
p0 = [1880, 10, 1400] # gauss
x_pix = np.arange(0, len(green_flat), 1)
pix_range = (p0[0] - 150, p0[0] + 150)
line_roi = x_pix[pix_range[0]:pix_range[1]] # ROI for a line to fit
fit_region = np.arange(line_roi.min(), line_roi.max(), 0.1)
popt, pcov = curve_fit(gauss, line_roi, green_flat[line_roi], p0=p0)


plt.figure(4)
plt.clf()
plt.xlabel('Pixels')
plt.ylabel('DN')
plt.title('1D background subtracted image')
plt.xlim(popt[0]-6*popt[1], popt[0]+6*popt[1])
plt.plot(green_flat, '-C0')
plt.plot(fit_region, gauss(fit_region, *popt), '--C2')
plt.show()

print('Centroid: {0:.1f}'.format(popt[0]))
print('Sigma: {0:.1f}, FWHM: {1:.1f} pixels'.format(popt[1], popt[1]*2.355))
print('Size of spot: {0:.2f} arcseconds'.format(popt[1]*2.355*plate_scale))

# # calculate standard deviation of image
# noise = np.std(background_flat)
# q = np.where(background_flat <= 3.0*noise)
# threshold_image = background_flat.copy()
# threshold_image[q] = 0

# plt.figure(5)
# plt.clf()
# plt.title('Threshold Image')
# plt.plot(threshold_image)
# plt.show()
