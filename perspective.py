import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from calibration import calibration_data

# Return perspective transformation matrix and its inverse
def perspective_transform(debug=False):
    image = mpimg.imread('test_images/straight_lines1.jpg')

    # Undistort using mtx and dist
    mtx, dist = calibration_data()
    undist = cv2.undistort(image, mtx, dist, None, mtx)

    # Source and destination points were selected manually
    src = np.float32([[696,455],[1077,692],[238,692],[591,455]])
    dst = np.float32([[1128,0],[1128,719],[290,719],[290,0]])

    if (debug==True):
        pts = np.array([[696,455],[1077,692],[238,692],[591,455]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(undist,[pts],True,(255,0,0))

        cv2.imwrite('output_images/perspective_in.jpg',np.asarray(undist)[:,:,::-1].copy())

        plt.imshow(undist)
        plt.show()

    # use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    if (debug==True):
        # use cv2.warpPerspective() to warp your image to a top-down view
        warped = cv2.warpPerspective(undist, M, (image.shape[1], image.shape[0]))

        cv2.imwrite('output_images/perspective_out.jpg',np.asarray(warped)[:,:,::-1].copy())

        plt.imshow(warped)
        plt.show()

    return M, M_inv

if __name__ == '__main__':
    image = mpimg.imread('test_images/straight_lines1.jpg')

    mtx, dist = calibration_data()
    M, M_inv = perspective_transform(debug=True)
    
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(undist)
    ax1.set_title('Undistorted Image')
    ax2.imshow(warped)
    ax2.set_title('Warped Image')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
