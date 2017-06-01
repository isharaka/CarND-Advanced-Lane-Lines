import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



# Return a binary image where sobel gradient in the specified orientation
# is within specified thresholds
def abs_sobel_threshold(img, orient='x', thresh_min=0, thresh_max=255):
    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1)
        
    # Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # Scale to 8-bit (0 - 25then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a mask of 1's where the scaled gradient magnitude 
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # Return this mask as binary_output image
    return binary_output

# Return a binary image where the magnitude of sobel gradient 
# is within specified thresholds
def mag_threshold(img, sobel_kernel=3, mag_threshold=(0, 255)):
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the magnitude 
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Scale to 8-bit (0 - 25and convert to type = np.uint8
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_threshold[0]) & (gradmag <= mag_threshold[1])] = 1
    # Return this mask as binary_output image
    return binary_output

# Return a binary image where the direction of sobel gradient 
# is within specified thresholds
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the x and y gradients
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Create a binary mask where direction thresholds are met
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    # Return this mask as binary_output image
    return binary_output

if __name__ == '__main__':
    # Read in an image and grayscale it
    image = cv2.imread('test_images/straight_lines1.jpg')
    image = np.asarray(image)[:,:,::-1].copy()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Calculate sobel thresholded images
    gradx = abs_sobel_threshold(gray, orient='x', thresh_min=20, thresh_max=100)
    grady = abs_sobel_threshold(gray, orient='y', thresh_min=20, thresh_max=100)
    mag_binary = mag_threshold(gray, sobel_kernel=3, mag_threshold=(30, 100))
    dir_binary = dir_threshold(gray, sobel_kernel=15, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    cv2.imwrite('output_images/sobel_x.jpg',255*gradx)
    cv2.imwrite('output_images/sobel_y.jpg',255*grady)
    cv2.imwrite('output_images/sobel_combined.jpg',255*combined)
    cv2.imwrite('output_images/sobel_mag.jpg',255*mag_binary)
    cv2.imwrite('output_images/sobel_dir.jpg',255*dir_binary)

    # Plot the result
    f, ax = plt.subplots(2, 2)
    print (ax.shape)
    f.tight_layout()
    ax[0][0].imshow(image)
    ax[0][0].set_title('Original Image')
    ax[0][1].imshow(combined, cmap='gray')
    ax[0][1].set_title('Combined')
    ax[1][0].imshow(gradx, cmap='gray')
    ax[1][0].set_title('Sobel x')
    ax[1][1].imshow(grady, cmap='gray')
    ax[1][1].set_title('Sobel y')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()