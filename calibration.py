import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

CALIBRATION_FILE = "calibration.p"

# Calibrate camera and return calibration and distortion data
def calibrate(debug=False):
    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # prepare object and image points
    nx = 9
    ny = 6

    objpoints = []
    imgpoints = []

    objp = np.zeros((nx*ny,3),np.float32)
    objp[:,:2]=np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    for fname in images:
        # Read image file
        print("Reading file " + fname)
        img = cv2.imread(fname)        

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, draw corners
        if ret == True:
            print (str(corners.shape[0]) + " corneres detected")
            imgpoints.append(corners)
            objpoints.append(objp)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

        if (debug==True):
            plt.imshow(img)
            plt.show()

    # Calculate calibration and distortian matrices
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2], None, None)

    # Save calibration data
    dist_pickle = {}
    dist_pickle['mtx'] = mtx
    dist_pickle['dist'] = dist
    pickle.dump( dist_pickle, open( CALIBRATION_FILE , "wb" ) )

    return mtx, dist

# Retrieve saved calibration data 
def calibration_data():
    dist_pickle = pickle.load(open( CALIBRATION_FILE, "rb"))
    return dist_pickle['mtx'], dist_pickle['dist']

if __name__ == '__main__':
    mtx, dist = calibrate()
    #mtx, dist = calibration_data()

    img = cv2.imread('test_images/straight_lines1.jpg')
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    cv2.imwrite('output_images/calibration_in.jpg',img)
    cv2.imwrite('output_images/calibration_out.jpg',undist)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(np.asarray(img)[:,:,::-1].copy())
    ax1.set_title('Original Image')
    ax2.imshow(np.asarray(undist)[:,:,::-1].copy())
    ax2.set_title('Undistorted Image')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    plt.show()