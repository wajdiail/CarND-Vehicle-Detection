**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./images/org_img_car.png
[image2]: ./images/org_img_noncar.png
[image3]: ./images/hog_feature_car_4_4_4.png
[image4]: ./images/hog_feature_noncar_4_4_4.png
[image5]: ./images/hog_feature_car_4_4_9.png
[image6]: ./images/hog_feature_noncar_4_4_9.png
[image7]: ./images/hog_feature_car_6_8_9.png
[image8]: ./images/hog_feature_noncar_6_8_9.png 
[image9]: ./images/hog_feature_car_luv_final.png 
[image10]: ./images/hog_feature_noncar_luv_final.png 
[image11]: ./images/org_img_car_hsv.png
[image12]: ./images/org_img_noncar_hsv.png
[image13]: ./images/org_img_car_luv.png
[image14]: ./images/org_img_noncar_luv.png
[image15]: ./images/search_area_1.png
[image16]: ./images/search_area_1.5.png
[image17]: ./images/final_test_1.png
[image18]: ./images/heat_test_1.png
[image19]: ./images/final_test_2.png
[image20]: ./images/heat_test_2.png
[image21]: ./images/final_test_3.png
[image22]: ./images/heat_test_3.png
[image23]: ./images/final_test_4.png
[image24]: ./images/heat_test_4.png
[image25]: ./images/final_test_5.png
[image26]: ./images/heat_test_5.png
[image27]: ./images/final_test_6.png
[image28]: ./test_images/test_1.png
[image29]: ./test_images/test_2.png
[image30]: ./test_images/test_3.png
[image31]: ./test_images/test_4.png
[image32]: ./test_images/test_5.png
[image33]: ./test_images/test_6.png
[image34]: ./images/after_fp_filter1.png
[image35]: ./images/after_fp_filter2.png
[image36]: ./images/after_fp_filter3.png
[image37]: ./images/after_fp_filter4.png
[image38]: ./images/after_fp_filter5.png
[image39]: ./images/after_fp_filter6.png
[video1]: ./project_output.mp4 

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the third code cell of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

|![alt text][image1]|![alt text][image2]|
|:-:|:-:|
|**Car**|**Non Car**|

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Below are some examples of using various HOG parameters

HOG parameters of `orientations=4`, `pixels_per_cell=4` and `cells_per_block=4`:

|![alt text][image3]|![alt text][image4]|
|:-:|:-:|
|**Car**|**Non Car**|

HOG parameters of `orientations=9`, `pixels_per_cell=4` and `cells_per_block=4`:

|![alt text][image5]|![alt text][image6]|
|:-:|:-:|
|**Car**|**Non Car**|

HOG parameters of `orientations=9`, `pixels_per_cell=6` and `cells_per_block=8`:

|![alt text][image7]|![alt text][image8]|
|:-:|:-:|
|**Car**|**Non Car**|

Different Color Spaces: 

Color Space: `HSV`

|![alt text][image11]|![alt text][image12]|
|:-:|:-:|
|**Car**|**Non Car**|

Color Space: `LUV`

|![alt text][image13]|![alt text][image14]|
|:-:|:-:|
|**Car**|**Non Car**|

I noticed that HSV, HLS converted images very similar and LUV, YCrCb, YUV were similar. I felt LUV is better and used it as my color space in this project

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters on color spaces and HOG parameters but the LUV and the below parameters yielded me a better view of the car compared to others also the training accuracy was slightly better.

HOG parameters of `orientations=9`, `pixels_per_cell=8` and `cells_per_block=8`:

| ![alt text][image9] | ![alt text][image10] |
|:-:|:-:|
|**Car**|**Non Car**|


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

After extracting the HOG features. I also extracted the historgram of color features and combined together to form the final features vector.

I then trained a linear SVM using the above extracted final feature vector and got an accuracy of 98.4% . The code is contained in the 8th cell of the ipython notebook

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search was implemented on the `orginal scale of 1` and on `scale of 1.5`. Instead of overlap, I defined how many cells to step. The details are below

Window to search: `ystart=350, ystop=500, xstart=750,xstop=1280`
Cells per step: `2`

The code is contained in the 9th cell of the ipython notebook in the function name `find_cars`
##### The visualization of search window:

|![alt text][image15]|![alt text][image16]|
|:-:|:-:|
|**Scale=1**|**Scale =1.5**|

|![alt text][image17]|![alt text][image18]|
|:-:|:-:|
|**Scale=1**|**Scale =1.5**|


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using LUV 3-channel HOG features and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image17]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

1. First I scoped the search window very specific to road and car moving path which elimated fasle positives generating from trees, skies and other side of traffic. Then I created a heapmap and thresholded to remove the weeker boxes. The final boxex optained from the label were sent to an merge algorithm which basically merges the 30% overlapping boxes.

The code cells : 10,11,12,13,14
functions: `remove_false_positive` `MergeBoxes`

### Here are six frames and their corresponding heatmaps:

![alt text][image28] ![alt text][image18]
![alt text][image29] ![alt text][image20]
![alt text][image30] ![alt text][image22]
![alt text][image31] ![alt text][image24]
![alt text][image32] ![alt text][image26]
![alt text][image33] ![alt text][image28]

### After applying false positive filter

|**Before**|**After**|
|:-:|:-:|
|![alt text][image18]| ![alt text][image34]|
|![alt text][image20]| ![alt text][image35]|
|![alt text][image22]| ![alt text][image36]|
|![alt text][image24]| ![alt text][image37]|
|![alt text][image26]| ![alt text][image38]|
|![alt text][image28]| ![alt text][image39]|



### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image17]
![alt text][image19]
![alt text][image21]
![alt text][image23]
![alt text][image25]
![alt text][image27]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* The pipeline will fail if there are cars in the lane of the driving car.Iniitial I applied the sliding window search on half of the image horizontally which have covered the cars on any lane. As the pipeline was performing slow due to limited computed power. I focussed the solution specific to this project and hence further limited the search area. But it can be easily changed to detect cars other on areas of the road.

* The accuray of the classifer can be further increased by using deep learning ConvNets. Again its a matter of data and compute power.

* More robust algorithm of elimating false positives and drawing tight bounding box can be used.





