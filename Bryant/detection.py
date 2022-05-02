# A majority of this code was derived from the following object detection tutorial:
# Source: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/plot_object_detection_saved_model.html

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import json
from pathlib import Path
import sys
import os


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


"""
for image_path in IMAGE_PATHS:

    print('Running inference for {}... '.format(image_path), end='')

    image_np = load_image_into_numpy_array(image_path)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    plt.figure()
    plt.imshow(image_np_with_detections)
    print('Done')
plt.show()
"""


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

    coords = detections['detection_boxes']
    coords[:, (0, 2)] = coords[:, (0, 2)] * image_np.shape[0]
    coords[:, (1, 3)] = coords[:, (1, 3)] * image_np.shape[1]
    #print(coords)
    with open(output_coords, 'w+') as file:
        file.write(json.dumps(coords.tolist(), indent=4))

    test = Image.fromarray(image_np_with_detections)
    test.save(output_image)
    print('Done')


if __name__ == '__main__':

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
