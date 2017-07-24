#!/usr/bin/python3.5

import os
import cv2
import numpy as np
import pickle
import glob
from copy import deepcopy
import math
from time import time
CALCAM_FILENAME = "cam_cal.pkl"
M = Minv = mtx = dist = rvecs = tvecs = None
#project calibration
orgPers = np.float32([[300, 660], [1010, 660], [700, 460], [586, 460]]) #project calibration
dstPers = np.float32([[300, 720], [980, 720], [980, 100], [300, 100]])
videoFileName = "project_video.mp4"
yLenXample = 29.0

#chalenger calibration
# orgPers = np.float32([[344,660],[933,660],[666,462],[620,462]]) #chalenger calibration
# videoFileName = "challenge_video.mp4"
# dstPers = np.float32([[300, 720], [980, 720], [980, 50], [300, 50]])
# yLenXample = 29.0
#
# #Hard chalenger calibration
# orgPers = np.float32([[344,660],[933,660],[743,500],[519,500]]) #chalenger calibration
# videoFileName = "harder_challenge_video.mp4"
# dstPers = np.float32([[500, 720], [780, 720], [780, 50], [500, 50]])
# yLenXample = 18.0

res=[]
xprop = 1
yprop = 1
nwindows = 12
leftx_base_old = dstPers[0,0]
rightx_base_old = dstPers[1,0]
old_lane_Width =0
old_left_fit = (0,0,leftx_base_old)
old_right_fit = (0,0,rightx_base_old)
isLastValid= False
ploty = np.linspace(0, 719, 720)
baseTime = time()
def startUp():
    global M, Minv, mtx, dist, rvecs, tvecs, xprop, yprop, ploty
    # xproportion is the width of the lane proyected/ real size in m.
    xprop = 3.7/(float)(dstPers[1,0]-dstPers[0,0])
    # the Y rectangle was 29m so:
    yprop = 29.0/(float)(dstPers[0,1]-dstPers[2,1])
    try:
        [z,M, Minv, mtx, dist, rvecs, tvecs] = pickle.load(open("cam_cal.pkl", "rb"))
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
                # img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                # cv2.imshow('img', img)
                # cv2.waitKey(40)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        M, Minv = getPerspectiveMatrix()

        # save all parameters.
        pickle.dump([M, Minv, mtx, dist, rvecs, tvecs], open('cam_cal.pkl', 'wb'))


# sobel transformation
def procSobel2(img, thresh_min=20, thresh_max=100):
    # rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    abs_sobely = np.absolute(sobely)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    scaled_sobely = np.uint8(255 * abs_sobely / np.max(abs_sobely))
    sxbinary = np.zeros_like(scaled_sobelx)
    sybinary = np.zeros_like(scaled_sobely)
    sybinary[(scaled_sobely >= thresh_min) & (scaled_sobely <= thresh_max)] = 1
    sxbinary[(scaled_sobelx >= thresh_min) & (scaled_sobelx <= thresh_max)] = 1
    return sxbinary, sybinary

def procSobel(img, thresh_min=20, thresh_max=100,y=True):
    if len(img.shape==3):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if y:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sbinary


def getPerspectiveMatrix():

    global orgPers, dstPers
    M = cv2.getPerspectiveTransform(orgPers, dstPers)
    Minv = cv2.getPerspectiveTransform(dstPers, orgPers)
    return M, Minv


def maskHSVYellowAndWhite(orig_img):
    # Global variables for tunning process.
    midH = 23
    midS = 18
    midV = 230
    thr = 19
    hsv = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
    # value = hsv[:, :, 2]
    # minV = np.min(value)
    # maxV = np.max(value)
    # hsv[:, :, 2] = (255.0 * (value - minV) / float(maxV - minV))
    hls = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HLS)
    lab = cv2.cvtColor(orig_img, cv2.COLOR_BGR2LAB)
    luv = cv2.cvtColor(orig_img, cv2.COLOR_BGR2Luv)
    # transform from BRG to HSV

    # get yellow mask
    '''maskY = cv2.inRange(hsv, np.array([22 - 3, 75, 140]), np.array([22 + 3,100, 180]))
    maskW = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([15, 10, 230]))
    maskS = cv2.inRange(hls, np.array([0, 80, 90]), np.array([255, 255, 255]))
    mask4 = cv2.inRange(hsv, np.array([100, 0, 170]), np.array([180, 50, 150]))
    #Positive Filters
    pf1 = cv2.inRange(hsv, np.array([150 - 1, 2 - 6, 98 - 10]), np.array([150 + 1, 2 + 6, 98 + 10]))
    pf2 = cv2.inRange(hsv,np.array([23-7,113-10,230-45]),np.array([23+7,113+10,255]))
    pf3 = cv2.inRange(hsv,np.array([21-3,62-36,126-25]),np.array([21+3,62+56,126+25]))
    pf4 = cv2.inRange(hsv, np.array([27-3,0,183]), np.array([27+3,0+39,183+67]))
    pf5 = cv2.inRange(hsv, np.array([117-11,82-47,102-19]), np.array([117+11,82+47,102+19]))
    nf1 = np.int8(np.logical_not(cv2.inRange(hsv, np.array([15 - 1, 22 - 1, 239 - 1]), np.array([15 + 1, 22 + 1, 239 + 1]))))*255
    nf2 = np.int8(np.logical_not(
        cv2.inRange(hsv, np.array([119-19,0,40]), np.array([140,240,255])))) * 255
    '''
    #maskY = cv2.inRange(hsv, np.array([22 - 3, 125 - 90, 180 - 181]), np.array([22 + 3, 125 + 90, 181 + 70]))
    # get withe mask
    #maskW = cv2.inRange(hsv, np.array([0, midS - 30, midV - 30]), np.array([176, midS + 16, midV + 25]))
    #maskS = cv2.inRange(hls, np.array([0, 80, 90]), np.array([255, 255, 255]))
    maskW = cv2.inRange(luv, np.array([220, 0,0]), np.array([255, 255,255]))
    maskY = cv2.inRange(lab, np.array([0, 0, 155]), np.array([255, 255, 255]))
    # to join both mask I have to do an OR between them,
    # finally make a BRG image with 255 in all dots yellow or white
    #mask = maskW
    mask = np.bitwise_or(maskW, maskY)
    #mask = np.bitwise_or(mask, maskS)
    #mask = np.bitwise_or(mask, mask4)
    #mask=maskY
    # mask = pf1

    #positive filters needs or operation
    # mask = np.bitwise_or(pf1, pf2)
    # mask = np.bitwise_or(mask, pf3)
    # mask = np.bitwise_or(mask, pf4)
    #mask = np.bitwise_or(mask, pf5)
    # and with negative filters
    # mask = np.bitwise_and(mask, nf1)
    #mask = np.bitwise_and(mask, nf2)
    mask3 = np.copy(orig_img)
    mask3[:, :, 0] = mask
    mask3[:, :, 1] = mask
    mask3[:, :, 2] = mask

    # apply mask with and bitwise operation to remove allother pixels.
    maskedImage = np.bitwise_and(mask3, orig_img)
    myLuv = np.bitwise_and(maskW,luv[:,:,0])
    #cv2.imshow("LAB",maskY)
    #cv2.imshow("LUV", myLuv)
    #return mask3
    return mask
def fquad(fit,y):
    y2=np.array([y**2,y,1])
    return np.dot(fit,y2)

def checkParalell(fitL, fitR, normal_distance):
    #as we supose tangent to de curve of the row in the bottom we can assume that:
    posL = fquad(fitL, ploty)
    porR = fquad(fitR, ploty)
    distance= (porR-posL)
    dRatio = sum(distance)/normal_distance/720.0
    ok= True
    if  dRatio > 1.5 or dRatio < 0.5:
        ok = False
    dRatio = min(distance)/normal_distance
    if  dRatio < 0.25:
        ok = False
    dRatio = max(distance) / normal_distance
    if dRatio > 2:
        ok = False
    return ok

    '''nfit2 = gen_parallel(fit1, distance, 0, 720)
    ypoints=np.arange(0,720)
    comparativa = np.array([])
    resy= fquad(fit2,ypoints)
    resy2 = fquad(nfit2, ypoints)
    print(np.dstack((ypoints,resy,resy2)))
    print(resy-resy2)'''

def doImageProcess(image):

    global mtx, dist
    image = np.copy(image)
    #poly= np.array([[image.shape[1]//2,0],[0,image.shape[0]//2],[0,image.shape[0]//2]])
    #image = cv2.fillPoly(image,[poly],(0,0,0))

   # Apply blur to original image with a small kernel.

    bluredImage = cv2.GaussianBlur(image, (3,3),0)

    # remove all pixels that is not white or yellow
    maskedImage = maskHSVYellowAndWhite(bluredImage)
    sobx, soby = procSobel2(maskedImage, 20, 150)
    filter =  soby * 255

    image[:,:,0]= filter
    image[:, :, 1] = filter
    image[:, :, 2] = filter
    maskedImage = np.bitwise_and(filter, maskedImage)
    #cv2.imshow("filtered",maskedImage)

    #return cv2.cvtColor(maskedImage,cv2.COLOR_BayerRG2GRAY)
    return maskedImage




def warped(img):
    global M
    return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)



def dewarped(img):
    global Minv
    return cv2.warpPerspective(img, Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)

maskHistory = []
nMemoryFrames = 5
isLeftValid = False
isRightValid = False
def curveStepOne(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image/
    maskHistory.append(binary_warped)
    for m in maskHistory:
        binary_warped = np.bitwise_or(binary_warped,m)
    if len(maskHistory)>nMemoryFrames:
        del maskHistory[0]
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped)) )
    global leftx_base_old, rightx_base_old, old_left_fit, old_right_fit, old_lane_Width, isLastValid,isRightValid,isLeftValid
    midpoint = binary_warped.shape[1]//2
    if not isLeftValid:
        histogram = np.sum(binary_warped[binary_warped.shape[0] * 4 // 5:, :midpoint], axis=0)
        if np.max(histogram[:])==0:
            leftx_base = leftx_base_old
        else:
            leftx_base = np.argmax(histogram[:])
    else:
        leftx_base = leftx_base_old
    if not isRightValid:
        histogram = np.sum(binary_warped[binary_warped.shape[0]*4 // 5:, midpoint:], axis=0)
        # Create an output image to draw on and  visualize the result
        # cv2.imshow("debg1",out_img[0])
        # binary_warped[0]
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        if np.max(histogram[:]) ==0:
            rightx_base=rightx_base_old
        else:
            rightx_base = np.argmax(histogram[:]) + midpoint
        laneWidth = (rightx_base-leftx_base)*xprop
    else:
        rightx_base = rightx_base_old
        laneWidth = old_lane_Width


    # Choose the number of sliding windows

    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base.astype(int)
    rightx_current = rightx_base.astype(int)
    # Set the width of the windows +/- margin
    margin = 60
    # Set minimum number of pixels found to recenter window
    minpix = 60
    maxpix = (margin*window_height*2)*.6
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    left_lane_xs = []
    right_lane_xs = []
    left_windows_failure = 0
    right_windows_failure = 0
    # Step through the windows one by one
    previous_left_offset = 0
    previous_right_offset = 0
    stopRight = 0
    stopLeft = 0
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0]  - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if not stopLeft and len(good_left_inds) > minpix and len(good_left_inds)< maxpix:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            previous_left_offset = leftx_current
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            previous_left_offset = leftx_current - previous_left_offset
        else:
            leftx_current += previous_left_offset
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 0,255), 2)
            if not stopLeft:
                left_windows_failure += 1
        if (not stopRight) and (len(good_right_inds) > minpix) and (len(good_right_inds)< maxpix):
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            previous_right_offset = rightx_current
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            previous_right_offset = rightx_current - previous_right_offset
        else:
            rightx_current += previous_left_offset
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 0, 255), 2)
            if (not stopRight):  right_windows_failure += 1
            if len(good_right_inds) > minpix:
                rightx_current += previous_left_offset
        if len(good_left_inds) < minpix and len(good_right_inds) > minpix:
            previous_left_offset = previous_right_offset
        #stop geting data
        left_lane_xs.append(leftx_current)
        right_lane_xs.append(rightx_current)
        if not stopLeft and (leftx_current<80):
            right_windows_failure += nwindows-window
            left_windows_failure += nwindows - window
            stopLeft = True
        if not stopRight and (rightx_current>1180):
            right_windows_failure += nwindows - window
            left_windows_failure += nwindows - window
            stopRight = True


    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)




    minimumW= 3
    maxFailuers = nwindows-minimumW
    fakeLeft = False
    fakeRight = False
    # Fit a second order polynomial to each
    isLastValid = False
    isLeftValid = False
    isRightValid = False

    #sanity checks.
    # 1) verify number of valid windows and extract left and right line pixel positions
    if left_windows_failure <=maxFailuers:
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        left_fit = np.polyfit(lefty, leftx, 2)
        isLeftValid = True
    else:
        pass
    if right_windows_failure <=maxFailuers:
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        right_fit = np.polyfit(righty, rightx, 2)
        isRightValid = True

    # 2) Verify paralells (distance)

    if isLeftValid and isRightValid:
        goodParalel = checkParalell(left_fit,right_fit,3.9/xprop)
        if not goodParalel:
            if right_windows_failure > left_windows_failure:
                isRightValid = False
            else:
                isLeftValid = False

    # 3) verify angles

    if isLeftValid:
        left_curvature = curvatureLine(left_fit,ploty)
        if np.min(np.abs(left_curvature)) < 50:
            isLeftValid= False
    if isRightValid:
        right_curvature = curvatureLine(right_fit,ploty)
        if np.min(np.abs(right_curvature)) < 50:
            isRightValid= False

    # reconstruct failures

    if (not isLeftValid) and (not isRightValid):
        left_fit = deepcopy(old_left_fit)
        right_fit = deepcopy(old_right_fit)
    elif (not isLeftValid) or (not isRightValid):
        if not isLeftValid:
            left_fit = gen_parallel(right_fit, -3.9 / xprop, 0, out_img.shape[0], xprop / yprop)

        if not isRightValid:
            right_fit = gen_parallel(left_fit, 3.9 / xprop, 0, out_img.shape[0], xprop / yprop)
    else:
        isLastValid = True
    leftx_base_old =  fquad(left_fit,720)
    rightx_base_old = fquad(right_fit, 720)
    laneWidth = old_lane_Width= (rightx_base_old - leftx_base_old) * xprop
    old_left_fit = deepcopy(left_fit)
    old_right_fit = deepcopy(right_fit)


    return left_fit, right_fit, laneWidth, isLeftValid, isRightValid,out_img

def gen_parallel(fit, offset, startx, endx, aberration=1):
    nx = []
    ny = []
    aberration = 1
    for x in range(endx):
        dxdy = (2*fit[0]*x + fit[1])
        y = fit[0]*(x/aberration)**2 + fit[1]*(x/aberration) + fit[2]
        ang = math.atan(dxdy)
        nx.append(x/aberration+math.sin(-ang)*offset)
        ny.append(y+math.cos(-ang)*offset)
    newFit= np.polyfit(nx, ny, 2)
    #print(newFit[0]*720**2+newFit[1]*720+newFit[2])
    return newFit

def curvature(leftx,rightx,ploty,y_eval):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = yprop  # meters per pixel in y dimension
    xm_per_pix = xprop  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / \
                    (2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / \
                     (2 * right_fit_cr[0])
    #print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    return left_curverad, right_curverad

def curvatureLine(fit,ploty):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = yprop  # meters per pixel in y dimension
    xm_per_pix = xprop  # meters per pixel in x dimension

    # fit_cr gives a real X when introduce a proyected y

    left_fitx = fit[0]*ploty**2+fit[1]*ploty+fit[2]
    fit_cr = np.polyfit(ploty*yprop, left_fitx*xprop,2)

    # Calculate the new radii of curvature in all points
    curverad = ((1 + (2 * fit_cr[0] * ploty *yprop + fit_cr[1]) ** 2) ** 1.5) / \
                    (2 * fit_cr[0])

    # Now our radius of curvature is in meters in all points
    return curverad

def takeBrick(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(-20,20):
            brick=img[y-20:y+20,x-25+i:x+25+i,:]
            cv2.imwrite("train/line_" + str((int)(time()*1000))+".png",brick)
    if event == cv2.EVENT_MBUTTONDOWN:
        for i in range(-20,20):
            brick=img[y-20:y+20,x-25+i:x+25+i,:]
            cv2.imwrite("train/other_" + str((int)(time()*1000))+".png",brick)


def getOffsetFromCenter(left_fit, right_fit):
    l = fquad(left_fit,720)
    r = fquad(right_fit,720)
    return (640-l-(r-l)/2)*xprop


def main():
    global img
    startUp()
    cleanImage=False
    cap = cv2.VideoCapture(videoFileName)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    [valid, img] = cap.read()
    dotsL = np.array((img.shape[0], 2), dtype=np.uint8)
    imgMod = warped(doImageProcess(img))
    #left_fit, right_fit,,, = curveStepOne(imgMod)
    #cv2.imshow("Test", imgMod)
    cv2.imshow("Mix", img)
    cv2.setMouseCallback('Mix', takeBrick)
    font = 'FONT_HERSHEY_SIMPLEX'
    frame =0
    while valid:
        frame +=1
        if frame == 2:
            pass
        baseTime = time()
        # undistort the image using the matrix from calibration
        img = cv2.undistort(img, mtx, dist, None, mtx)
        print("Undistort {:3.2f}".format(1000 * (time() - baseTime)))
        imgMod = warped(img)
        print("Filters {:3.2f}".format(1000 * (time() - baseTime)))
        imgMod = doImageProcess(imgMod)
        print("warped {:3.2f}".format(1000*(time() - baseTime)))
        polW = np.zeros(img.shape,dtype='uint8')
        polWP = np.zeros(img.shape, dtype='uint8')

        dst = np.zeros(img.shape)
        left_fit, right_fit, laneWidth, lwf, rwf,out_img = curveStepOne(imgMod)
        print("Curve {:3.2f}".format(1000 * (time() - baseTime)))
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        dotsL = np.dstack((left_fitx, ploty))
        dotsR = np.dstack((right_fitx, ploty))
        par_fit = gen_parallel(left_fit, 3.7 / xprop, 0, 720)
        parx_fit = fquad(par_fit,ploty)
        dotsPar = np.dstack((parx_fit,ploty))
        poligon = np.concatenate((np.int32(dotsL), np.flip(np.int32(dotsR), axis=1)), axis=1)
        print("Prepare polys {:3.2f}".format(1000 * (time() - baseTime)))
        if lwf:
            lColor = (255, 0, 255)
        else:
            lColor = (0,0,255)
        if rwf:
            rColor = (255, 0, 255)
        else:
            rColor = (0,0,255)
        if rwf and lwf:
            mColor = (0, 255, 0)
        elif rwf or lwf:
            mColor = (0, 255, 255)
        else:
            mColor = (0,0,255)

        #cv2.fillPoly(polW, [dstPers.astype(int)], (0, 0, 255)) #original calibration polygon
        cv2.fillPoly(polWP, np.int32(poligon), mColor)
        cv2.polylines(polWP, np.int32(dotsL), False, lColor, thickness=10)
        cv2.polylines(polWP, np.int32(dotsR), False, rColor, thickness=10)
        polT = cv2.perspectiveTransform(poligon.astype(float),Minv)
        cv2.fillPoly(polW, polT.astype(int), mColor)
        polT = cv2.perspectiveTransform(dotsL.astype(float),Minv)
        cv2.polylines(polW, polT.astype(int), False, lColor,thickness=10)
        polT = cv2.perspectiveTransform(dotsR, Minv)
        cv2.polylines(polW, polT.astype(int), False, rColor,thickness=10)
        #cv2.polylines(polW, np.int32(dotsPar), False, (0,0, 255), thickness=10)
        '''persp3c = np.zeros(img.shape)
        persp3c[:,:,0]= 0
        persp3c[:,:,1] = imgMod * 255
        persp3c[:,:,2] = imgMod * 255'''
        #dw = np.uint8(dewarped(polW))
        print("dewarped {:3.2f}".format(1000 * (time() - baseTime)))
        dst = cv2.addWeighted(img, .7, np.uint8(polW), .3, 0.0)
        print("Weighted 1 {:3.2f}".format(1000 * (time() - baseTime)))
        cv2.addText(dst,"Width: {:.2}     Frame: {:}".format(laneWidth,frame),(10,30),font,15,(200,0,0))
        cv2.addText(dst, "Left ok: {:} Right ok: {:}".format(lwf,rwf), (10, 60), font, 15, (200, 0, 0))
        print("Text 1 {:3.2f}".format(1000 * (time() - baseTime)))
        lc, rc = curvature(left_fitx, right_fitx, ploty, 700)
        mlc = curvatureLine(left_fit,ploty)[700]
        mrc = curvatureLine(right_fit, ploty)[700]
        mpc = curvatureLine(par_fit, ploty)[700]

        cv2.addText(dst, "Left curvature: {:.0f} Right curvature: {:.0f}".format(lc, rc),
                    (10, 90 ), font, 15, (255, 0, 255))
        center_distance = getOffsetFromCenter(left_fit,right_fit)
        cv2.addText(dst, "Distance to lane center: {:.2f}".format(center_distance),
                    (10, 120), font, 15, (255, 0, 255))
        print("Curvature 1 {:3.2f}".format(1000 * (time() - baseTime)))
        '''for i in range(1):
            lc, rc = curvature(left_fitx,right_fitx,ploty,720-40-i*80)
            cv2.addText(dst, "Pos y {} Left curvature: {:.0f} Right curvature: {:.0f}".format(720-40-i*80,lc, rc), (10, 60+20*i), font, 15, (255, 0, 255))
        print("Text 2 {:3.2f}".format(1000 * (time() - baseTime)))'''

        #cv2.imshow("Org", img)
        #cv2.imshow("Persp", out_img)
        out2 = cv2.resize(polWP,(320,180))
        out3 = cv2.resize(out_img, (320, 180))

        dst[0:180,960:]=out2
        dst[185:365, 960:] = out3

        if cleanImage:
            cv2.imshow("Mix",img)
        else:
            cv2.imshow("Mix", dst)
        cv2.imwrite(videoFileName[:-4]+"/"+videoFileName[:-4]+'_{:04}'.format(frame)+'.jpg',dst)
        if(laneWidth>4):
            pass

        k=chr(cv2.waitKey(5)&255)
        print("Draw {:3.2f}".format(1000 * (time() - baseTime)))
        if k=='p':
            k=cv2.waitKey(100)
            while k!='c' and k!='p':
                k = chr(cv2.waitKey(100)&255)
                pass
        if k == 't':
            #Enter in training mode
            cleanImage=not cleanImage
        if k == 'm':
            frame=547
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        if k == 'f':
            frame+=60
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        if k=='r':
            if frame >60:
                frame-=60
            else:
                frame=0
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        [valid, img] = cap.read()



if __name__ == "__main__":
    main()
