# Football Mapper
Project to create 2-d minimap representation of football (soccer) games.

Seth Mize, Lucas Franz, Bryant Cornwell

# Methods
## Object Detection
Attempt to build model using TensorFlow 2:
- Utilized the following tutorial for setting up an object detection model: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html
- Instead of using a pre-trained model, I wanted to create my own using the information from tensorflow: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md 
- Using faster_rcnn_resnet50_coco.config
- Needed checkpoint files from a pretrained model in order to create a 		model.
- Found checkpoint files for this config and the checkpoint 		version was not supported for version 2.
- Not enough information/guidance for troubleshooting all of the errors as the tutorial for creating a model seem dated. Finding another approach.

## Overhead view base photo

Source: https://conceptdraw.com/a1992c3/preview