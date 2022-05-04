# A majority of this code was derived from the following object detection tutorial:
# Source: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/plot_object_detection_saved_model.html

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
import json
from pathlib import Path
import sys
import os


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def main(input_image, output_image, output_coords):
    model = 'TensorFlow/workspace/training_model/exported-models/saved_model'
    labels_path = 'TensorFlow/workspace/training_model/annotations/labels_objects.pbtxt'
    #image_np = np.array(Image.open('../documentation/data/original/sample_1.png'))
    image_np = np.array(Image.open(input_image))
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Add model and detection points for bounding box
    detect_fn = tf.saved_model.load(model)
    detections = detect_fn(input_tensor)
    category_index = label_map_util.create_category_index_from_labelmap(labels_path, use_display_name=True)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()
    #  Adds the percentage scores to the label.
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index=category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False)
    # Prepare to export the bounding box coordinates for the image.
    coords = detections['detection_boxes']
    scores = detections['detection_scores']
    # De-normalize the coordinates to pixel values.
    coords[scores >= 0.3, (0, 2)] = coords[scores >= 0.3, (0, 2)] * image_np.shape[0]
    coords[scores >= 0.3, (1, 3)] = coords[scores >= 0.3, (1, 3)] * image_np.shape[1]
    # Generate a Json file of Coordinates for the image.
    with open(output_coords, 'w+') as file:
        file.write(json.dumps(coords.tolist(), indent=4))
    # Export the image with detections.
    test = Image.fromarray(image_np_with_detections)
    test.save(output_image)
    print('Done')


if __name__ == '__main__':
    # Code takes an input directory and runs the program for every file within that directory.
    # The code outputs a output images with objects detected and a json file with the bounding box coordinates.
    input_directory = Path(sys.argv[1])
    output_directory = Path(sys.argv[2])

    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    for file in os.listdir(input_directory):
        print(file)
        file = Path(file)
        input_path = input_directory / file
        output_path = output_directory / file.name.replace(file.suffix, '.png')
        coords_path = output_directory / file.name.replace(file.suffix, '.json')

        main(input_path, output_path, coords_path)
