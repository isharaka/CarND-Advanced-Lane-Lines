import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from moviepy.editor import VideoFileClip

from calibration import calibration_data
from perspective import perspective_transform
from findlanes import fit_lanes_init, fit_lanes, lane_image, lane_curvature, preprocess_frame


class Lane:
    def __init__(self,fit,n_lane_pixels):
        self.fit = np.array(fit)
        self.n_lane_pixels = n_lane_pixels
        
    def __str__(self):
        return 'a: {:+f} b: {:+f} c: {:+f}'.format(self.fit[0], self.fit[1], self.fit[2])
    
    def distance(self,other):
        ret = 0;
        for i in  range(len(self.fit)):
            ret = ret + ((self.fit[i]-other.fit[i])**2) / (abs(self.fit[i]) * abs(other.fit[i]))
        
        return ret / len(self.fit)

# conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension


left_lane_buffer = []
right_lane_buffer = []

def pipeline(img):
    global left_lane_buffer, right_lane_buffer

    mtx, dist = calibration_data()
    M, M_inv = perspective_transform()
    undist, combined, warped = preprocess_frame(img, mtx, dist, M, M_inv)

    if(len(left_lane_buffer) < 1):
        left_fit, right_fit, left_pixels, right_pixels = fit_lanes_init(warped)
    else:
        left_fit, right_fit, left_pixels, right_pixels = fit_lanes(warped,left_lane_buffer[-1].fit,right_lane_buffer[-1].fit)

        if( (left_pixels < left_lane_buffer[-1].n_lane_pixels/2) or (right_pixels < right_lane_buffer[-1].n_lane_pixels/2)):
            left_fit, right_fit, left_pixels, right_pixels = fit_lanes_init(warped)


    if(len(left_lane_buffer) >= 50):
        del left_lane_buffer[0]
    if(len(right_lane_buffer) >=50):
        del right_lane_buffer[0]


    left_lane_buffer.append(Lane(left_fit,left_pixels))
    right_lane_buffer.append(Lane(right_fit,right_pixels))

    left_distance = 0;
    if(len(left_lane_buffer) > 1):    
        left_distance = left_lane_buffer[-1].distance(left_lane_buffer[-2]) 

    right_distance = 0;
    if(len(right_lane_buffer) > 1):
        right_distance = right_lane_buffer[-1].distance(right_lane_buffer[-2])

    smooth_left_fit = sum(lane.fit for lane in left_lane_buffer)/len(left_lane_buffer)
    smooth_right_fit = sum(lane.fit for lane in right_lane_buffer)/len(right_lane_buffer)

    left_curverad, right_curverad, lane_width, lane_deviation = lane_curvature(warped, xm_per_pix, ym_per_pix, smooth_left_fit, smooth_right_fit)

    overlay = lane_image(img, M_inv, smooth_left_fit, smooth_right_fit)

    curvature = 'L: {:+010.2f}m R: {:+010.2f}m'.format(left_curverad, right_curverad)
    lanedata = 'W: {:+010.2f}m D: {:+010.2f}m'.format(lane_width, lane_deviation)

    cv2.putText(overlay,curvature,(700,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(overlay,lanedata,(700,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)

    return undist, combined, overlay


def process_image(image):
    stage1, stage2, result = pipeline(image)
    return result
 
image = mpimg.imread('test_images/test6.jpg')

stage1, stage2, result = pipeline(image)

# Plot the result
f, ax = plt.subplots(2, 2, figsize=(24, 9))
f.tight_layout()

ax[0][0].imshow(image)
ax[0][0].set_title('Original Image')
ax[0][1].imshow(stage1)
ax[0][1].set_title('Stage 1')
ax[1][0].imshow(stage2)
ax[1][0].set_title('Stage 2')
ax[1][1].imshow(result)
ax[1][1].set_title('Pipeline Result')

plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

output_clip = 'output.mp4'
input_clip = VideoFileClip("project_video.mp4")
clip = input_clip.fl_image(process_image) #NOTE: this function expects color images!!
clip.write_videofile(output_clip, audio=False)