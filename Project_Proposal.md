# Football Mapping

Seth Mize, Lucas Franz, Bryant Cornwell

# Project Description

Over the last two decades the use of analytics has become pervasive in professional sports leagues. One component of interest is player location tracking with respect to the dimensions of the playing surface. The primary means of collecting this information is done through GPS tracking through wearable technology or manual annotations. Our project seeks to accomplish this through computer vision techniques.

Focusing on the sport of football (soccer), we will utilize available broadcast feeds to convert the video frames into a two-dimensional overhead representation of the visible player locations in correlation to the visible portion of the pitch. This will be accomplished by extracting the lines from the pitch and the point/s at which a player is making contact with the pitch, and applying the appropriate projection to transform the found coordinates into their two-dimensional overhead representation. Being successful in this project from still images from a variety of angles, areas of the pitch, and different pitches will establish the foundation to be able to convert an entire broadcast feed into this overhead viewpoint. 


# Reading List

- Hough Transform: Underestimated Tool in the Computer Vision Field
    https://www.researchgate.net/profile/Simon-Karpenko/publication/228573007_Hough_Transform_Underestimated_Tool_In_The_Computer_Vision_Field/links/0fcfd51487c0d13691000000/Hough-Transform-Underestimated-Tool-In-The-Computer-Vision-Field.pdf

- A Computer Vision based Lane Detection Approach
    https://www.kuet.ac.bd/webportal/ppmv2/uploads/15695142241555251476A%20Computer%20Vision%20based%20Lane%20Detection%20Approach.pdf

- Tracking soccer players aiming their kinematical motion analysis
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.86.6699&rep=rep1&type=pdf

- Semantic annotation of soccer videos: automatic highlights identification
    https://d1wqtxts1xzle7.cloudfront.net/34408817/cviu2003-pp-with-cover-page-v2.pdf?Expires=1646590248&Signature=aVtgwtvky8bFOSgQeK92x7uqnscTwuyvMCSH4McGr78m3VCOdyaiNN7EQEs77q1u7Rvg55m1RgKl13pUwqH0EBnWpo2~Ze1vOJ15YjU1M4mbsu340r8MD2GyIk2NyaHEbsUQzUpdD78MhddpEBWGqFQ2-TyqY1kd6NNHaUZMjAax4tRTRuBHx28ciF-XWafxUgKd3sugnIhcje86yvfXWy8D4aQDbbfmjRi9gofqePw8zdTCTPFFU5hJqQkAdLQR3rG1QzaSyxRNwlsveUmvT7ZWTau~xr3n80tdHiNsTsScmxuUl-E1KV6KaWJh17b7PBNeEqu-98krH3oVB-JEYQ__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA

 - Soccer video and player position dataset
    https://d1wqtxts1xzle7.cloudfront.net/48709988/Soccer_Video_and_Player_Position_Dataset20160909-8093-1b4mz52-with-cover-page-v2.pdf?Expires=1646590361&Signature=RFWt~FaREWMKOmJD9P2gDYZcGlIrNPupcUF8s-0WDjnEK~birNVzT9cSjK~Rfo8nO79gPwpdOTWv6SOYpGoiPskyrxyA45862M78xeXhhq9EIiopm2Qe8i8Wa1z77g3PeMJarywxF8k5F9up0VpWXa0pvzoshXOUM2c1e1i2GDw09a49BikqFPKsqF9pZVS3ZQB~T6p0KMy42LEM9s3eoDww2vl3aa5ebJq7UDfwCetCeAEa7~mDC-y~I2DDjKykh6KAeOYluXG4cCxBREYE8uAQhhGQ8p4-op6~5Yx5V53oG3Ko27bDv-o0HVhJmLIwcW0huTa-89azXyVpSudXww__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA

- 

# Research Plan and Time-line

When executing this project we will assign primary responsibilities but plan to work with constant collaboration.

Primary Responsibilities:
- Lucas Franz - Line Detection and Pitch-Line Extraction
- Bryant Cornwell - Object Detection and Recognition
- Seth Mize - Projection 

## Stage 1: "Identification" (March 26)

This project has multiple recognition components that can be carried out in isolation and in parallel without disrupting the other

### Pitch Line Identification

Based on a still-frame of a football match, identifying the lines painted on the pitch. They have generally consistent dimensions from pitch-to-pitch, so these lines can be later used to determine the necessary parameters for projecting the still frame into a 2-d overhead representation.

### Object Detection

Identify objects (people, ball, and goals) in the image to be able to determine the point at which the contact the pitch (used as their location for the projection into the overhead view).


## Stage 2: "Projection and Interim Project Report" (April 3)

### Still-Frame to 2-d Projection

Generate the mapping from the 2-d gridspace to the equivalent gridspace of the still-frame. Use this mapping to project the location of objects from the still-frame gridspace to the 2-d gridspace.

This work can be partially done in parallel with Stage 1, using expected outputs to generate the projection process. Refinement can be done once Stage 1 is finished. At this stage, the projection of the balls and players may be inaccurate if they leave the ground substantially.

### Interim Project Report

This can be done in parallel with the work of Stage 1 and Stage 2.

## Stage 3: "Refinement" (April 23)

At this point, the stated task will have been completed and made available the ability to return to the original image to extrapolate further information to be included in the two-dimensional representation

### Person Recognition and Categorization

Be able to identify a detected object as a person and categorize it into.

- Team A
- Team B
- Referee
- Other

### Ball Recognition

Be able to identify a detected object as a soccer ball. If possible, determine whether the ball is on the ground or in flight. If this not possible from a still frame, it may be easier when analyzing the frames in sequence in Stage 4.

### Goal Recognition

Be able to identify a detected object as one of the goals.

## Stage 4: "Sequential Frame Analysis" (April 30 / Deadline)

With all of the previous stages completed, video frames can be processed on an individual basis and then stitched back together to produce the overhead video representation of the game.

### Event Detection

Identifying player possession of the ball and passing between players of the same team.

Being able to identify when players jump or the ball is kicked into the air to be able to determine elevation and improve the projection for objects in flight.

Given that the goal and ball detections, identifications, and projections are accurate, we should be able to identify when a goal is scored.

### Filtering Non-Gameplay

When analyzing frames in sequence, we should be able to filter out frames that are not part of the gameplay (e.g. ads, close-ups, audience).

### Video Compilation and Output

Compile the processed frames into a watchable video.

# Data and Experiments

## Data Collection

We intend to use video footage from FIFA gameplay posted to youtube (https://www.youtube.com/results?search_query=fifa+full+match).

Additionally, we plan to use data for experimentation / validation from a dataset that contains still-frame pictures of the same time from multiple angles (https://datasets.simula.no/alfheim/).

## Experiments

We're going to have several stages of experiments corresponding to the different types of work in each stage.

- Line Projection

    Using the multiple-still-frame dataset, validate that our algorithm correctly maps the multiple images to the same 2-d projection (for where they overlap).

- Player Detection and Recognition

    Hand label players into their category and use the hand-labeled data to determine accuracy of the player detection and recognition.

- Event Detection

    Hand label possession of ball to determine accuracy of ball possession recognition and passing.

    TBD: Identify a method to validate the accuracy of frame-to-frame movement prediction.
