# Football Mapping

Seth Mize, Lucas Franz, Bryant Cornwell

# Project Description

Over the last two decades the use of analytics has become pervasive in professional sports leagues. One component of interest is player location tracking with respect to the dimmensions of the playing surface. The primary means of collecting this information is done through GPS tracking through wearable technology or manual annotations. Our project seeks to accomplish this through computer vision techniques.

Focusing on the sport of football (soccer), we will utlize available broadcast feeds to convert the video frames into a two-dimmensional overhead representation of the visible player locations in correlation to the visible portion of the pitch. This will be accomplished by extracting the lines from the pitch and the point/s at which a player is making contact with the pitch, and applying the appropriate projection to transform the found coordinates into their two-dimmensional overhead representation. Being successful in this project from still images from a variety of angles, areas of the pitch, and different pitches will establish the foundation to be able to convert an entire broadcast feed into this overhead viewpoint. 


# Reading List


# Research Plan and Time-line

When executing this project we will assign primary responsibilities but plan to work with constant collaboration.

Primary Responsibilities:
- Lucas Franz - Line Extraction
- Bryant Cornwell - Object Recognition
- Seth Mize - Projection

## Stage 1: "Recognition" (March 26)

This project has two recognition components that can be carried out in isolation and in parrellel without disrupting the other

- Pitch Line Identification
- Player-Pitch Contact Identification

## Stage 2: "Projection" (April 9)

Once the pertinent information has be identified and isolated we can move to projection

- Project Pitch Lines to 2D
- Use Calculated Projection of Lines to Map Players

## Stage 3: "Refinement" (April 23)

At this point, the stated task will have been completed and made available to ability to return to the original image to extrapolate further information to be included in the two-dimmensional representation

- Player Categorization into Teams
- Ball Identification
- Referee Identification

## Stage 4: "Video" (April 30)

With all pieces in place, video frames can be processed on an individual basis and then stitched back together to produce the overhead video representation of the game


# Data and Experiments

