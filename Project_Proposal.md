# Football Mapping

Seth Mize, Lucas Franz, Bryant Cornwell

# Project Description

Over the last two decades the use of analytics has become pervasive in professional sports leagues. One component of interest is player location tracking with respect to the dimensions of the playing surface. The primary means of collecting this information is done through GPS tracking through wearable technology or manual annotations. Our project seeks to accomplish this through computer vision techniques.

Focusing on the sport of football (soccer), we will utilize available broadcast feeds to convert the video frames into a two-dimensional, overhead representation of the displayed player locations based on the corresponding portion of the pitch in the frame of the image. This will be accomplished by extracting the lines from the pitch and the point/s at which a player is making contact with the pitch, and applying the appropriate projection to transform the found coordinates into their two-dimensional overhead representation. Success of this process for a variety of still images for multiple pitches will establish the foundation for us to convert an entire broadcast feed into this overhead viewpoint. 


# Reading List

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

# Research Plan and Time-line

When executing this project we will assign primary responsibilities but plan to work with constant collaboration.

Primary Responsibilities:

- Lucas Franz - Line Detection and Pitch-Line Extraction

- Bryant Cornwell - Object Detection and Recognition

- Seth Mize - Projection 

## Stage 1: "Identification" (March 26)

This project has multiple recognition components that can be carried out in parallel without disrupting the other components.

### Pitch Line Identification

Based on a still-frame of a football match, identify the lines painted on the pitch. They have generally consistent dimensions from pitch-to-pitch, so these lines can be later used to determine the necessary parameters for projecting the still frame into a 2-d overhead representation.

### Object Detection

Identify objects (people, ball, and goals) in the image to be able to determine the point at which they contact the pitch (used as their location for the projection into the overhead view).

## Stage 2: "Projection and Interim Project Report" (April 3)

### Still-Frame to 2-d Projection

Generate the mapping from the 2-d gridspace to the equivalent gridspace of the still frame. Use this mapping to project the location of objects from the still-frame gridspace to the 2-d gridspace.

This work can be partially done in parallel with Stage 1, using expected outputs to generate the projection process. Refinement can be done once Stage 1 is finished. At this stage, the projection of the balls and players may be inaccurate if they leave the ground substantially.

### Interim Project Report

This can be done in parallel with the work of Stage 1 and Stage 2.

## Stage 3: "Refinement" (April 23)

At this point, we will be able to expand on the previously implemented features to improve our 2-d projection methods by being able to identify specific object categories and potentially improve accuracy for objects that are in flight.

### Person Recognition and Categorization

Be able to identify a detected object as a person and categorize it into:

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

We intend to use video footage from FIFA gameplay posted to youtube (https://www.youtube.com/results?search_query=fifa+full+match). For hand-labelling frames of the video, we will only select a reasonable amount of frames to be able to tune and validate algorithms. The exact number of images we will use is TBD; however, given the number of frames in the video, we a have a large sample space to select from.

Additionally, we plan to use data for experimentation / validation from a dataset that contains still-frame pictures of the same time from multiple angles (https://datasets.simula.no/alfheim/).

## Experiments

We're going to have several stages of experiments corresponding to the different types of work in each stage.

- Projection

    Using a dataset including three simultaneous camera angles of overlapping viewpoints, we can conduct the projection into the 2-d overhead perspective independently and measure consistency by judging the distances between projections for the overlapping portions of the images.

- Player Detection and Recognition

    Hand label players into their category and use the hand-labeled data to determine accuracy of the player detection and recognition.

- Event Detection

    Hand label possession of ball to determine accuracy of ball possession recognition and passing.

    TBD: Identify a method to validate the accuracy of frame-to-frame movement prediction.

- Other Experiments

    As we go through the project, we will develop additional experiments as needed as the project evolves.
