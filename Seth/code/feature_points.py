from pathlib import Path
from copy import deepcopy
import os

import numpy as np
import cv2 as cv
from tqdm import tqdm

from part1 import orb_sift_match, pad_image
from part2 import get_projection_matrix, test_transition_matrix, apply_transformation, convert_vector

def centroid(points):

    return np.sum(points, axis=0) / points.shape[0]

def ransac(point_matches, sample_size, iterations, threshold, min_match_sample_size):
    # Referenced the Wiki article on RANSAC to develop a better understanding of how
    # to implement it, including referencing the pseudocode.

    best_transform = None
    best_sample = None
    best_sample_test_size = 0
    best_error = np.inf

    i = 0
    with tqdm(total = iterations) as pbar:
        while i < iterations:
            pbar.update(1)
            sample_space = np.random.choice(len(point_matches), size=sample_size*11, replace=False)
            sample_space = point_matches[sample_space]
            np.random.shuffle(sample_space)
            sample, sample_test = sample_space[:sample_size], sample_space[sample_size:10*sample_size]
            try:
                transform = get_projection_matrix(sample)
            except np.linalg.LinAlgError as err:
                continue
            sample_test = np.array([sample_test[j] for j in range(sample_test.shape[0]) if test_transition_matrix(transform, [sample_test[j]]) < threshold])
            if len(sample_test) > min_match_sample_size or best_transform is None:
                if best_transform is None:
                    new_sample = sample_space
                else:
                    new_sample = np.concatenate([sample, sample_test])
                error = test_transition_matrix(transform, new_sample)
                if error < best_error or (error == best_error and len(sample_test) > best_sample_test_size):
                    best_transform = transform
                    best_error = error
                    best_sample = new_sample
                    best_sample_test_size = len(sample_test)

            i += 1

    best_sample_a = best_sample[:,0,:]
    centroid_a = centroid(best_sample_a)
    best_sample_b = best_sample[:,1,:]
    centroid_b = centroid(best_sample_b)

    return best_transform, best_sample, centroid_a, centroid_b

class FeaturePointExtractor:

    def __init__(self, image_path):

        self.load_file(image_path)

    def load_file(self, image_path):

        image = cv.imread(image_path)
        self._image = image
        self._gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        self.mask()

    def sharpen(self):

        image_filter = np.array([[0, -1, 0],
                                 [-1, 5, -1],
                                 [0, -1, 0]])

        self._gray_image = cv.filter2D(self._gray_image, -1, image_filter)

    def mask(self, lowerb=100, upperb=255):
        mask = cv.inRange(self._gray_image, lowerb, upperb)
        self._gray_image = cv.bitwise_and(self._gray_image, mask)

    @property
    def potential_corners(self):
        return self._potential_corners

    @property
    def image(self):
        return self._image

    @property
    def gray_image(self):
        return self._gray_image

    def harris_corner_detection(self, threshold):
        dst = cv.cornerHarris(self._gray_image, 2, 3, 0.04)
        dst = cv.dilate(dst, None)
        image = deepcopy(self.image)
        self._potential_corners = np.transpose((dst > threshold * dst.max()).nonzero())
        image[dst > threshold * dst.max()] = [0,0,255]
        return image

    def blobs(self):
        detector = cv.SimpleBlobDetector_create()
        keypoints = detector.detect(self.gray_image)

        im_with_keypoints = cv.drawKeypoints(self.image, keypoints, np.array([]), (255,0,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return im_with_keypoints

if __name__ == '__main__':
    """
    above = FeaturePointExtractor('above.jpg')
    above.sharpen()
    acorners = above.harris_corner_detection(0.05)
    above_points = above.potential_corners
    """

    cleaned_directory = Path('D:/Users/Olev/data/football-mapper/TV_soccer/cleaned/')
    corner_directory = Path('D:/Users/Olev/data/football-mapper/TV_soccer/corners/')

    for file in tqdm(os.listdir(cleaned_directory)):
        fpe = FeaturePointExtractor(str(cleaned_directory / file))
        #fpe.sharpen()
        fpe.mask(150,255)
        corners = fpe.harris_corner_detection(0.2)
        output_points = fpe.potential_corners
        cv.imwrite(str(corner_directory / file), corners)

    point_matches = []

    """
    for ap in tqdm(above_points, desc='Above Points', unit='points', position=1):
        for op in tqdm(output_points, desc='Output Points', unit='points', position=2, leave=False):
            point_matches.append([ap, op])
    np.random.shuffle(point_matches)
    point_matches = np.array(point_matches)

    transform_matrix, shared_coordinates, centroid_a, centroid_b = ransac(point_matches, 4, 1000, 0.75, 10)
    transformed = apply_transformation(corners, transform_matrix)

    cv.imwrite('transformed.jpg', transformed)
    """


