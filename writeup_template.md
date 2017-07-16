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

[image1]: ./output_images/undistorted.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road untransformed"
[image3]: ./output_images/undistorted-jpg.jpg "Road Transformed"
[image4]: ./output_images/perspCalImage.jpg "Rectangle"
[image5]: ./output_images/perspTransformed.jpg "Top view"

[image6]: ./output_images/persRestored.jpg "Inverse"
[video1]: ./project_video.mp4 "Video"
[image8]: ./SegmentDistance.png "SegmentDistance"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "mqAdvancedLanes.ipynb" steps 1 to 9.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

That applied to a real world image

![alt text][image2]
 produce this images:
![alt text][image3]



#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][imageb]

#### 3. Perspective transform.

To do measurements in the car images we will preform a perspective transform, suposing a plat plane ground.

First I check where is the photo taken and I found it!!!! that it is [here](https://www.google.es/maps/@37.4398602,-122.2485646,3a,75y,321.74h,82.59t/data=!3m6!1e1!3m4!1sDkPzleAWMIFzyiAlx8VPzw!2e0!7i13312!8i6656?hl=en)

So I can measure the distance between lines with the help of satellite view in Google
maps

![satelite measurement][image8]

I calculate around  14.5 between segments and a width of 14m 4 lanes, so 1 lane 3.5m

Thats a good reference to calibrate the perspective transform.

Over a undistorted image we fill get the original points and select the destination points
so we made a rectangle of 3.5m/29m H/W

|Source|Destination|
|:----:|:---:|
|300,660|300,700|
|110,660|1010,700|
|700,460|1010,20|
|586,460|300,20|


So we can get the transformation matrix with `M=getTransform(org,dst)`

and it inverse as

`Minv=getTransform(dst,org)`


as a result we can view three images Original with the known rectangle, the perspective transformation and finally the inverse

![rectangle][image4]
![top][image5]
![perspective][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
