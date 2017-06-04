## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration_chess.jpg "Chess"
[image2]: ./output_images/calibration.jpg "Undostorted"
[image3]: ./test_images/test5.jpg "Binary Example"
[image4]: ./output_images/pre_process_combined.jpg "Binary Example Output"
[image5]: ./output_images/perspective_in.jpg "Perspective In"
[image6]: ./output_images/perspective_out.jpg "Perspective Out"
[image7]: ./output_images/perspective_out2.jpg "Perspective Test2"
[image8]: ./output_images/perspective_test.jpg "Perspective Test"

[image9]: ./output_images/fit_lanes_init.jpg "Initial lane fitting"
[image10]: ./output_images/fit_lanes_init_pixels.jpg "Initial lane fitting Lane pixels"
[image11]: ./output_images/fit_lanes.jpg "Subsequent lane fitting"
[image12]: ./output_images/overlay.jpg "Overlay"
[image13]: ./output_images/fit_lanes_init_problem.jpg "Initial lane fitting"


[video1]: ./output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

See function `calibrate` in `calibration.py` for the implentation of camera calibration

Camera calbration was done using `cv2.calibrateCamera()` function. This requires a set of points in the images and the correponding set of images in the real world.

The former set was obtained using `cv2.findChessboardCorners`. 
The latter was calcualted as a grid based on the prior knowledge of the chess board pattern.

For beter accuracy multiple camera images were used to extract the above sets of points.

Once the distortion matrices were obtained `cv2.undistort` was used to correct for distortion in new images.

*Camera calibration correction on a chessboard image*
![alt text][image1] 



### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

See Camera Calibration section.
*Camera calibration correction on a test image*
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

See function`preprocess_frame` in `findlanes.py`.

I used the following combination of color and gradient thresholds to generate a binary image.

sobel x gradient - Lane lines are almost vertical therefore x direction sobel gradeient will pick the lane markings
S channel in HLS space - Saturation channel picked the lanes the best. It is also invariant with regards to the colour of the lane lines

Here's an example of thersholding output for test image test5.jpg. 

*Original image*
![alt text][image3]
*green - soble x gradient threholding blue - S channel thersholding*
![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

See function`perspective_transform` in `perspective.py`.

I used `cv2.getPerspectiveTransform` calculate transformation matrix (and the inverse transformation matrix). I picked the source and destination points required for this function manually. I assumed that the lane lines were straight, parellel and on a flat surface.

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 696, 455      | 1128, 0        | 
| 1077, 692      | 1128, 719      |
| 238, 692     | 290, 719      |
| 591, 455      | 290, 0        |

Once the transformation matrices were calcualted `cv2.warpPerspective` does the transformation.

*Image used to calculate perspective transform with the reference points marked*
![alt text][image5]
*The same image after perspective transformation*
![alt text][image6]
*Warped image after inverse perspective transformation*
![alt text][image7]

I verified that my perspective transform was working as expected by testing the trasform on a curved section of the road and verifying that the lanes appear parallel in the warped image.

*Perspective transformation tested on a curved section of the road. The lanes appear parellel in the transformed image*
![alt text][image8]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

See function`fit_lanes_init` in `findlanes.py`.

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image9]
![alt text][image10]

See function`fit_lanes` in `findlanes.py`.
![alt text][image11]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

See function`lane_curvature` in `findlanes.py`.



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

See function`lane_image` in `findlanes.py`.

 Here is an example of my result on a test image:

![alt text][image12]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

See function `pipeline` in `main.py` for the final implementation of the complete pipeline.

Here's a [link to my video result](./output.mp4)
![alt text][video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Where possible I used the code snippets from the lessons and quizes.

One problem I faced was in fitting curves to lanes. With broken lane lines sometimes, there were not enough lane pixels. This is more apparant when the lane inside the curves goes out of the warped frame.

*Incorrect fitting on test image test6.jpg*
![alt text][image13]

I used averaging the fit over a number of previous frames to mitigate this. See lines 84-86 in function `pipeline` in `main.py`.

It may be possible to improve the fit in such cases by artificially combining the lane fragments befre fitting.

