import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from perspective import perspective_transform
from calibration import calibration_data
from sobel import abs_sobel_threshold, mag_threshold, dir_threshold


# Preprocess a frame from camera so that lane features can be extracted
# Returns a binary image with same width and height as the frame with
# ones in pixels where lanes are likely to be present
def preprocess_frame(img, mtx, dist, M, M_inv, debug=False):
    # COrrect for camera distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # Convert to HLS color space and separate the channels
    hsv = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hsv[:,:,0]
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]

    # Calculate sobel gradients in x direction.
    # Lane lines are almost vertical therefore x direction sobel gradeient will 
    # pick the lane markings
    sxbinary = abs_sobel_threshold(l_channel, orient='x', thresh_min=20, thresh_max=100)
    sxbinary = 1 * sxbinary

    # S channel picked the lanes the best. It is also invariant with regards to
    # the colour of the lane lines
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 170) & (s_channel <= 255)] = 1

    # Combine information from color thersholding and sobel gradient
    combined = np.zeros_like(s_channel)
    combined[( ( sxbinary == 1 ) | ( s_binary == 1) )] = 1

    if debug==True:
        color_binary = np.dstack(( np.zeros_like(s_channel), sxbinary, s_binary))
        cv2.imwrite('output_images/pre_process_combined.jpg',np.asarray(255*color_binary)[:,:,::-1].copy())

    # Warp the preprocessed image to a birds eye view
    binary_warped = cv2.warpPerspective(combined, M, (img.shape[1], img.shape[0]))

    if debug==True:
        cv2.imwrite('output_images/pre_process_warped.jpg',255*binary_warped)

    return undist, combined, binary_warped

# Given a binary image with ones in pixels where lanes are likely to be present
# pick the pixels where the left and right lanes are and fit quadratic curves
# to left and right lanes
def fit_lanes_init(binary_warped, debug=False):
    binary_warped = np.uint8(255.0 * binary_warped)
    # Take a histogram of the bottom half of the image. This determines the
    # most likely location of the lanes in the bottom lines of the frame
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)

    if debug == True:
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        if debug == True:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if debug == True:
        print (leftx.shape)
        print (rightx.shape)

        f, ax = plt.subplots(1,1)
        ax.plot(rightx, righty, 'ro')
        plt.show()
        f.savefig('output_images/fit_lanes_init_pixels.jpg')

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        f, ax = plt.subplots(1,1)
        ax.imshow(out_img)
        ax.plot(left_fitx, ploty, color='yellow')
        ax.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

        plt.show()
        f.savefig('output_images/fit_lanes_init.jpg')

        print(len(left_lane_inds), len(right_lane_inds))

    return left_fit, right_fit, len(left_lane_inds), len(right_lane_inds)


# Given a binary image with ones in pixels where lanes are likely to be present
# and the likey quadratic curves for left and right lanes
# pick the pixels where the left and right lanes are and fit quadratic curves
# to left and right lanes
def fit_lanes(binary_warped, left_fit, right_fit, debug=False):
    binary_warped = np.uint8(255.0 * binary_warped)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Identify the nonzero pixels in x and y within the margin
    # from the likely lane curves
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if debug == True:
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        f, ax = plt.subplots(1,1)
        ax.imshow(result)
        ax.plot(left_fitx, ploty, color='yellow')
        ax.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

        plt.show()
        f.savefig('output_images/fit_lanes.jpg')

        print(len(left_lane_inds), len(right_lane_inds))

    return left_fit, right_fit, len(left_lane_inds), len(right_lane_inds)


# Given a binary image with ones in pixels where lanes are likely to be present
# and the likey quadratic curves for left and right lanes
# calculate the curve radius of left and right lanes, lane width
# and the position of the car relative the middle of the lane
def lane_curvature(binary_warped, xm_per_pix, ym_per_pix, left_fit, right_fit):
    # Calculate points on the curves representing left and right lane
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # We are avaluating the curvature of the lanes at points neares to the camera
    y_eval = np.max(ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

    # Calculate the new radii of curvature in meters
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Calculate the points where left and right lanes intersects the bottom of the frame
    left_lane_pos = left_fitx[np.uint16(y_eval)]
    right_lane_pos = right_fitx[np.uint16(y_eval)]

    # Calculate lane width in meters
    lane_width = (right_lane_pos-left_lane_pos) * xm_per_pix

    # Calculate the position of the car relative the mid point of the lane in meters
    lane_deviation = (((right_lane_pos+left_lane_pos) / 2) - (binary_warped.shape[1]/2)) * xm_per_pix
    
    return left_curverad, right_curverad, lane_width, lane_deviation


# Given an undistorted frame from the camera
# and the likey quadratic curves for left and right lanes
# draw a polygon representing the lane on to the image
def lane_image( undist, M_inv, left_fit, right_fit ):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(undist[:,:,0]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Calculate points on the curves representing left and right lane
    ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M_inv, (undist.shape[1], undist.shape[0])) 

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return result

if __name__ == '__main__':
    image = mpimg.imread('test_images/test5.jpg')

    mtx, dist = calibration_data()
    M, M_inv = perspective_transform()

    undist, combined, binary_warped = preprocess_frame(image, mtx, dist, M, M_inv, debug=True)

    plt.imshow(undist)
    plt.show()

    plt.imshow(combined)
    plt.show()

    plt.imshow(binary_warped, cmap='gray')
    plt.show()

    print(binary_warped.shape)
    print(np.max(binary_warped))


    left_fit, right_fit, left_pixels, right_pixels = fit_lanes_init(binary_warped, debug=True)

    left_fit, right_fit, left_pixels, right_pixels = fit_lanes(binary_warped, left_fit, right_fit, debug=True)

    ym_per_pix = 30/720
    xm_per_pix = 3.7/700

    left_curverad, right_curverad, lane_width, lane_deviation = lane_curvature(binary_warped, xm_per_pix, ym_per_pix, left_fit, right_fit)
    print(left_curverad, 'm', right_curverad, 'm', lane_width, 'm', lane_deviation, 'm')

    result = lane_image( undist, M_inv, left_fit, right_fit )
    cv2.imwrite('output_images/overlay.jpg',np.asarray(result)[:,:,::-1].copy())

    plt.imshow(result)
    plt.show()