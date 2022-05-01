# Introduction

Over the last two decades the use of analytics has become pervasive in professional sports leagues. One component of interest is player location tracking with respect to the dimensions of the playing surface. The primary means of collecting this information is done through GPS tracking through wearable technology or manual annotations. Our project seeks to accomplish this through computer vision techniques.

Focusing on the sport of football (soccer), we will utilize available broadcast feeds to convert the video frames into a two-dimensional, overhead representation of the displayed player locations based on the corresponding portion of the pitch in the frame of the image. This will be accomplished by extracting the lines from the pitch and the point/s at which a player is making contact with the pitch, and applying the appropriate projection to transform the found coordinates into their two-dimensional overhead representation. Success of this process for a variety of still images for multiple pitches will establish the foundation for us to convert an entire broadcast feed into this overhead viewpoint. 

# Background and Related Work

Our project is focused on coupling multiple existing techniques towards a new application. The problem can be broken into 3 primary categories; Line Detection, Object Detection, and Projection. By identifying the standard lines of the pitch, we can compare them with a proportion overhead representation of standard pitch line to solve for a projection transformation. Once a transformation matrix is solved for, identified objects can be mapped into the same overhead space. Further object categorization can be used to provide more detail to the generated overhead representation. 

When investigating the best applications of line detection we found the most useful prior research to be lane detection for autonomous vehicles. The _Tracking soccer players aiming their kinematical motion analysis_ article discussed how to split blobs or groups of more that one player with detection which we will need to utilize when recognizing a detected groups of players. For the image projection from the video projection to the overhead projection, the course material related to assignment 2 has been our reference. 

# Progress So Far

## Pitch Line Detection

### Pixel Isolation 

The first step in line detection for our problem is to remove unnecessary pixels that could result in the identification of unimportant lines. To accomplish this we looked at two different techniques.

#### Red, Green, Blue Filtering (RGB)

Our first attempt at this isolation was using the RGB pixel values included within the image. The thought process was that because the pitch lines are always white, we would be able to isolate the white pixels. There were two issues with this approach. The first issue is fairly obvious, there are a multitude of unimportant white pixels throughout the image, including those in billboards and stadium features that obscure our goal. The second issues arises with the camera quality and image compression. Because the lines are constantly bordering the green grass of the field, the pitch line pixels often had elevated green values. If the RGB filtering was too focused on removing the green grass we would lose large pitch line sections, if the filtering was too relaxed in the green channel, large portions of the grass would remain included.

#### Hue, Saturation, Lightness Filtering (HSL)

Looking at solutions for road line detection we noticed their implementation of HSL filtering. HSL is a conversion to a new pixel definition space from the standard RGB space. Using the cv2 package, this conversion results in 3 values per pixel, ranging from 0 to 255. The 3 channels are true to their name with hue representing the color of the pixel, saturation representing the boldness of the pixel, and lightness representing the brightness of the pixel. For hue, we've found the best filter to be 0-80, isolating the green spectrem. Saturation is filtered to 150-255, and lightness is filtered to 20-255. 

### Line Detection

After isolating the pitch line pixels with the best HSL results so far, we have experimented with Canny Edge Detection and Hough Transforms. The cv2 package includes functions for each of these tools. Canny Edge Detection does a better job of picking up all visible pitch lines, unfortunatly it also detects the other objects remaining on the grass. Hough Transform does a better job at looking to extract only the pitch lines, but struggles when detecting non-linear aspects of the pitch like the center circle and top of the goalie's box. Hough Transform also struggles when the camera image skews some of the pitch lines, making them non-linear.  

### Results

| Original | HSL Filter | HSL Canny | HSL Hough
| :-----------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------: |  :---------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------: | 
| <img src="documentation/Pitch_Lines/broadcast_img_0.jpg" alt="image_name" width="400"/> | <img src="documentation/Pitch_Lines/HSL_Output_0.png" alt="image_name" width="400"> | <img src="documentation/Pitch_Lines/Canny_Output_0.png" alt="image_name" width="400"> |<img src="documentation/Pitch_Lines/Hough_Output_0.png" alt="image_name" width="400"> |
| <img src="documentation/Pitch_Lines/broadcast_img_1.jpg" alt="image_name" width="400"/> | <img src="documentation/Pitch_Lines/HSL_Output_1.png" alt="image_name" width="400"> | <img src="documentation/Pitch_Lines/Canny_Output_1.png" alt="image_name" width="400"> |<img src="documentation/Pitch_Lines/Hough_Output_1.png" alt="image_name" width="400"> |
| <img src="documentation/Pitch_Lines/broadcast_img_2.jpg" alt="image_name" width="400"/> | <img src="documentation/Pitch_Lines/HSL_Output_2.png" alt="image_name" width="400"> | <img src="documentation/Pitch_Lines/Canny_Output_2.png" alt="image_name" width="400"> |<img src="documentation/Pitch_Lines/Hough_Output_2.png" alt="image_name" width="400"> |
| <img src="documentation/Pitch_Lines/broadcast_img_3.jpg" alt="image_name" width="400"/> | <img src="documentation/Pitch_Lines/HSL_Output_3.png" alt="image_name" width="400"> | <img src="documentation/Pitch_Lines/Canny_Output_3.png" alt="image_name" width="400"> |<img src="documentation/Pitch_Lines/Hough_Output_3.png" alt="image_name" width="400"> |



## Object Detection

The objects planned for detection are team A players, team B players, referees, goalies and the soccer ball. 
Detecting the goals is not necessary and adds a layer of complexity, so we removed it from object detection. Since we are looking to detect objects from live video camera angles, much from the images in the training dataset are taken from recorded soccer games.
Originally, the images were labelled by using different color bounding boxes per object category illustrated in the table below, but we ran into some difficulties when trying code the extraction of boundary points and category label.

| Image | Object Detection | 
| ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | 
| <img src="documentation/data/original/sample_1.png" alt="Sample 1" width="400"/> | <img src="documentation/data/detection_images_original/sample_1_detect.png" alt="Sample 1 Detected" width="400"> |
| <img src="documentation/data/original/sample_2.png" alt="Sample 2" width="400"/> | <img src="documentation/data/detection_images_original/sample_2_detect.png" alt="Sample 2 Detected" width="400"> |


The images were re-labelled using "LabelImg" that provides boundary coordinates and labels for manually drawn boundary boxes. See the table below for an example.

| Image | Label Dateframe | 
| ----------------------------------------------------------------------------------- |--------------------------------------------------------------------------------------------------- | 
| <img src="documentation/data/detection_images_original/detect_labelimg.png" alt="Sample 1" width="400"/> | <img src="documentation/data/detection_images_original/detect_df.png" alt="Sample 1 Labels" width="400"> |


The next step for object detection, is to develop a model with this training dataset and apply it to some test images and, eventually, video streams.
The final step will be refinement of the model that may include increasing the training dataset images and improving the code.

## Pitch Line Projection

When Assignment 2 first came out, we noticed how much overlap it had with this part of our Final Project, so a lot of the progress toward this section since our Project Proposal comes from Assignment 2. For the Pitch Line Projection, we're going to repurpose code from Assigntment 2.

We'll likely use an alternative to ORB/SIFT for feature point selection, likely using components of the line detection that we have already completed combined with some form of corner detection. We would then combine this with RANSAC, similar to Part 3 of Assignment 2 along with the Projective matrix code from Part 2 of Assignment 2 to identify the best mapping of the video projection to the overhead projection.

# Revised Research Plan


## 4/10/2022

### Seth

- Pitch Line Projection

    Implement one or more algorithms for feature point selection for the lines on the field, including ORB/SIFT, Harris Corner Detection, and others if needed.

### Lucas

- Pitch Line Detection

    Come up with a secondary filtering process with comparable results to HSL filtering.

## 4/17/2022

### Seth

- Pitch Line Projection

    Finish repurposing A2 code to apply RANSAC and the projective matrix calculation for the Pitch Line Projection.

    Run experiments for the Pitch Line Projection using the different methods for feature point selection to identify which method(s) will be best to use.

### Bryant

- Object Detection Algorithm

    The next step for object detection, is to develop a model with the training dataset and apply it to test images to testing and improve the detection. This algorithm will need to output detected object's bounding box for the relative position of the player on the field and labeling the object in object recognition.

    The final step will be refinement of the model that may include increasing the training dataset images and improving the code.

### Lucas

- Pitch Line Detection

    Create step in the pipeline for appropriately aggregating multiple pixel filters to produce the most isolated pitch lines for detection.

## 4/21/2022

### Bryant and Seth

- Object Recognition
    
    For object recognition, another algorithm for labeling the detected objects will need to be developed based off the work on the object detection algorithm. Performing experiments on still and video streams to test the algorithm. The algorithm will need to output a label for a specific detected object in order to categorize the object.

## 4/28/2022

### Bryant and Seth and Lucas

- Time Series Analysis

    To improve the accuracy of tracking objects for when they leave the ground, we'll need to analyze sequences of images from the video to help determine the elevation of the soccer ball overtime. We will likely implement something similar to Hidden Markov Models to help reduce the times that the soccer ball takes strange paths in the mapping during flight.

## 5/1/2022 or Final Due Date

### Bryant, Seth, and Lucas

- Final Project Report

    Compile all documentation and generate the report.


# References

- Hough Transform: Underestimated Tool in the Computer Vision Field
    https://www.researchgate.net/profile/Simon-Karpenko/publication/228573007_Hough_Transform_Underestimated_Tool_In_The_Computer_Vision_Field/links/0fcfd51487c0d13691000000/Hough-Transform-Underestimated-Tool-In-The-Computer-Vision-Field.pdf

- A Computer Vision based Lane Detection Approach
    https://www.kuet.ac.bd/webportal/ppmv2/uploads/15695142241555251476A%20Computer%20Vision%20based%20Lane%20Detection%20Approach.pdf

- Tracking soccer players aiming their kinematical motion analysis
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.86.6699&rep=rep1&type=pdf

- Semantic annotation of soccer videos: automatic highlights identification
    https://doi.org/10.1016/j.cviu.2003.06.004

- Soccer video and player position dataset
    https://doi.org/10.1145/2557642.2563677
  
- LabelImg Documentation: https://github.com/tzutalin/labelImg

- Advanced Lane Detection Using Computer Vision: https://towardsdatascience.com/teaching-cars-to-see-advanced-lane-detection-using-computer-vision-87a01de0424f
