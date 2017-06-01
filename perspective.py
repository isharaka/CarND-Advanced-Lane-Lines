import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from calibration import calibration_data

def corners_unwarp(img, nx, ny):
    #Convert to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        # draw corners
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # define 4 source points src = np.float32([[,],[,],[,],[,]])
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # define 4 destination points dst = np.float32([[,],[,],[,],[,]])
        img_size = (gray.shape[1], gray.shape[0])
        dst = np.float32([[100,100,], [img_size[0]-100,100], [img_size[0]-100,img_size[1]-100], [100,img_size[1]-100]])
        # use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # use cv2.warpPerspective() to warp your image to a top-down view
        warped = cv2.warpPerspective(undist, M, img_size)
    else:
        warped = img
        M = None

    return warped, M


def perspective_transform():
    image = mpimg.imread('test_images/straight_lines1.jpg')

    # Undistort using mtx and dist
    mtx, dist = calibration_data()
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    src = np.float32([[696,455],[1077,692],[238,692],[591,455]])
    dst = np.float32([[1128,0],[1128,719],[290,719],[290,0]])

    pts = np.array([[672,438],[1128,688],[290,688],[617,438]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(undist,[pts],True,(255,0,0))

    #plt.imshow(undist)
    #plt.show()

    # use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(undist, M, (image.shape[1], image.shape[0]))

    #plt.imshow(warped)
    #plt.show()

    return M, M_inv

if __name__ == '__main__':
    nx = 8 # the number of inside corners in x
    ny = 6 # the number of inside corners in y

    img = cv2.imread('camera_cal/calibration4.jpg')

    # Undistort using mtx and dist
    mtx, dist = calibration_data()
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    warped, M = corners_unwarp(undist, nx, ny)


    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax2.imshow(undist)
    ax2.set_title('Undistorted Image')
    ax3.imshow(warped)
    ax3.set_title('Undistorted and Warped Image')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

    M = perspective_transform()
    image = mpimg.imread('examples/screenshot_from_project_video.mp4.png')
    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax2.imshow(warped)
    ax2.set_title('Pipeline Result')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
