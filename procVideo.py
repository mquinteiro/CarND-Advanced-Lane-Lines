#!/usr/bin/python3.5

import os
import cv2
import numpy as np

CALCAM = "cal"
def startUp():
    if os.path.exists(cam_cal.pkl):
        [mtx, dist, rvecs, tvecs] = pickle.load(open("cam_cal.pkl", "rb"))
    else:

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
                img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                #cv2.imshow('img', img)
                #cv2.waitKey(40)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

