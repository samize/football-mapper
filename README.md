# Football Mapper
Project to create 2-d minimap representation of football (soccer) games.

Seth Mize, Lucas Franz, Bryant Cornwell

# Project Description
Over the last two decades the use of analytics has become pervasive in professional sports leagues. One component of interest is player location tracking with respect to the dimmensions of the playing surface. The primary means of collecting this information is done through GPS tracking through wearable technology or manual annotations. Our project seeks to accomplish this through computer vision techniques.

Focusing on the sport of football (soccer), we will utlize available broadcast feeds to convert the video frames into a two-dimmensional overhead representation of the visible player locations in correlation to the visible portion of the pitch. This will be accomplished by extracting the lines from the pitch and the point/s at which a player is making contact with the pitch, and applying the appropriate projection to transform the found coordinates into their two-dimmensional overhead representation. Being successful in this project from still images from a variety of angles, areas of the pitch, and different pitches will establish the foundation to be able to convert an entire broadcast feed into this overhead viewpoint. 

- Identify Objects in still image

- Classify objects in still image

- Track object movement in video

- Translate tracking to 2d space*

- Recognize events in the game*
# Reading List

    (Need to update as we find the papers)

# Research Plan and Timeline

The first task for this project will be to isolate lines from a still images to identify if the image contains a pitch/field. If the image contains a field, overlay gridlines on the pitch based on dimensions that are provided as parameters of the program. Object detection algorithms for players, ball, etc. can be developed while developing the image grid space. Once the image grid space is determined, positions from object detection relative to the field can be used to mapped into the image grid space. Determine the position of the players on the field based on the provided image grid space and object detection. Using object positions, event detection such as ball possession or (other events) can be explored.    

## Timeline

The dates below are subject to change. Documentation of the project will be alongside the corresponding work below.

- March 7th - Project proposal 

- March 28th - Field/Pitch image grid space and 2-D projection

- March 28th - Object detection

- April 3rd - Interim project report

- April 18th - Object recognition (Team A player, Team B player, referee, Team A Goalie, Team B Goalie…)

- April 18th - Object positioning in the image grid space and mapping to 2-D projection

- May 6th - Event recognition/detection

# Plan for Data and Experiments

Describe how you’ll evaluate your project. 

What image data will you use? Planning on using image data from youtube videos of soccer game footage. 

How will you measure whether or not the output of your approach is “correct”? Manually review or label images... (may require meeting to discuss)  

How many images will you use? We will be using images from videos, so hundreds if not thousands of images for testing and tuning. The exact number we are unsure of yet.