#!/usr/bin/python3.5

import os
import cv2
import numpy as np
import pickle
import glob

CALCAM_FILENAME = "cam_cal.pkl"
M = Minv = mtx = dist = rvecs = tvecs = None

def startUp():
    global M,Minv, mtx, dist, rvecs, tvecs
    try:
        [M, Minv, mtx, dist, rvecs, tvecs] = pickle.load(open("cam_cal.pkl", "rb"))
    except:
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.
        images = glob.glob('camera_cal/calibration*.jpg')
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                #img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                #cv2.imshow('img', img)
                #cv2.waitKey(40)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        M, Minv = getPerspectiveMatrix()

        #save all parameters.
        pickle.dump([M,Minv, mtx, dist, rvecs, tvecs], open('cam_cal.pkl', 'wb'))

# sobel transformation
def procSobel(img,thresh_min = 20,thresh_max = 100):

    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    sobely = cv2.Sobel(rgb, cv2.CV_64F, 0, 1)
    abs_sobely = np.absolute(sobely)
    sobelx = cv2.Sobel(rgb, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))
    sxbinary = np.zeros_like(scaled_sobelx)
    sybinary = np.zeros_like(scaled_sobely)
    sybinary[(scaled_sobely >= thresh_min) & (scaled_sobely <= thresh_max)] = 1
    sxbinary[(scaled_sobelx >= thresh_min) & (scaled_sobelx <= thresh_max)] = 1
    return sxbinary, sybinary


def getPerspectiveMatrix():
    org = np.float32([[300, 660], [1010, 660], [700, 460], [586, 460]])
    dst = np.float32([[300, 700], [1010, 700], [1010, 120], [300, 120]])
    M = cv2.getPerspectiveTransform(org, dst)
    Minv = cv2.getPerspectiveTransform(dst,org)
    return M, Minv



def maskHSVYellowAndWhite(orig_img):
    # Global variables for tunning process.
    midH = 23
    midS = 18
    midV = 230
    thr = 19
    hsv = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HLS)
    # transform from BRG to HSV

    # get yellow mask
    maskY = cv2.inRange(hsv, np.array([22 - 3, 125 - 90, 180 - 100]), np.array([22 + 3, 125 + 90, 100 + 70]))
    # get withe mask
    maskW = cv2.inRange(hsv, np.array([0, midS - 30, midV - 25]), np.array([176, midS + 16, midV + 25]))
    maskS = cv2.inRange(hls, np.array([0, 80, 90]), np.array([255, 255, 255]))
    # to join both mask I have to do an OR between them,
    # finally make a BRG image with 255 in all dots yellow or white
    mask = np.bitwise_or(maskW, maskY)
    #mask = np.bitwise_or(mask, maskS)
    mask3 = np.copy(orig_img)
    mask3[:, :, 0] = mask
    mask3[:, :, 1] = mask
    mask3[:, :, 2] = mask

    # apply mask with and bitwise operation to remove allother pixels.
    maskedImage = np.bitwise_and(mask3, orig_img)
    return mask3

def doImageProcess(image):
    global mtx, dist
    #undistort the image using the matrix from calibration
    image = cv2.undistort(image, mtx, dist, None, mtx)
    # Apply blur to original image with a small kernel.

    bluredImage = cv2.GaussianBlur(image, (3, 3), 0)

    # remove all pixels that is not white or yellow
    maskedImage = maskHSVYellowAndWhite(bluredImage)
    sobx, soby = procSobel(maskedImage,20,100)
    return sobx
    #return soby

def warped(img):
    global M
    return cv2.warpPerspective(img, M,(img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)

def dewarped(img):
    global Minv
    return cv2.warpPerspective(img, Minv,(img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)

def curveStepOne(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image/
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped)) * 255)
    #cv2.imshow("debg1",out_img[0])
    #binary_warped[0]
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
    return left_fit,right_fit


def main():
    startUp()

    #cap = cv2.VideoCapture("harder_challenge_video.mp4")
    cap = cv2.VideoCapture("challenge_video.mp4")
    #cap = cv2.VideoCapture("project_video.mp4")
    [valid,img] = cap.read()
    imgMod = warped(doImageProcess(img))
    curveStepOne(imgMod)
    cv2.imshow("Test",imgMod);
    while valid:
        imgMod = warped(doImageProcess(img))
        cv2.imshow("Org", img);
        cv2.imshow("Test", imgMod*255);
        cv2.waitKey(1)
        [valid, img] = cap.read()

        #cv2.waitKey(1)
        #curveStepOne(cv2.cvtColor(imgMod,cv2.COLOR_HSV2BGR))



if __name__ == "__main__":
    main()