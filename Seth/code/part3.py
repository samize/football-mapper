#!/usr/local/bin/python3

import sys
from pathlib import Path
from copy import deepcopy
import warnings

import numpy as np
import cv2

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

    print(len(point_matches))
    i = 0
    while i < iterations:
        sample_space = deepcopy(point_matches)
        np.random.shuffle(sample_space)
        sample, sample_test = sample_space[:sample_size], sample_space[sample_size:]
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
                print(i, error)
                best_transform = transform
                best_error = error
                best_sample = new_sample
                best_sample_test_size = len(sample_test)

        i += 1

    print('Best sample error:', test_transition_matrix(best_transform, best_sample))
    best_sample_a = best_sample[:,0,:]
    centroid_a = centroid(best_sample_a)
    best_sample_b = best_sample[:,1,:]
    centroid_b = centroid(best_sample_b)

    return best_transform, best_sample, centroid_a, centroid_b

def map_single_point(matrix, point):

    point = convert_vector(point)
    point = np.matmul(np.linalg.inv(matrix), point)
    point = point / point[2]
    point = point[:2]

    return point

def average_pixels(image_a, image_b):

    BLACK = np.array([0,0,0])
    new_image = np.zeros(shape=image_a.shape)

    for x in range(image_a.shape[1]):
        for y in range(image_a.shape[0]):
            axy = image_a[y,x]
            bxy = image_b[y,x]

            if (axy == BLACK).all():
                new_image[y,x] = bxy
            elif (bxy == BLACK).all():
                new_image[y,x] = axy
            else:
                new_image[y,x] = (axy + bxy) / 2
    
    new_image = np.rint(new_image).astype(int)
    return new_image

def stitch(image_a, image_b, centroid_a, centroid_b):

    height_a, width_a, _ = image_a.shape
    height_b, width_b, _ = image_b.shape

    #centroid_delta = np.rint(centroid_b - centroid_a).astype(int)
    centroid_delta = np.rint(centroid_b).astype(int) - np.rint(centroid_a).astype(int)

    pad_a_top = round((centroid_delta[0] > 0) * abs(centroid_delta[0]))
    pad_a_left = round((centroid_delta[1] > 0) * abs(centroid_delta[1]))
    
    pad_b_top = round((centroid_delta[0] < 0) * abs(centroid_delta[0]))
    pad_b_left = round((centroid_delta[1] < 0) * abs(centroid_delta[1]))

    pad_a_bottom = max(height_a + pad_a_top, height_b + pad_b_top) - height_a
    pad_a_right = max(width_a + pad_a_left, width_b + pad_b_left) - width_a

    pad_b_bottom = max(height_a + pad_a_top, height_b + pad_b_top) - height_b
    pad_b_right = max(width_a + pad_a_left, width_b + pad_b_left) - width_b

    new_height_a = height_a + pad_a_top + pad_a_bottom
    new_width_a = width_a + pad_a_left + pad_a_right
    new_height_b = height_b + pad_b_top + pad_b_bottom
    new_width_b = width_b + pad_b_left + pad_b_right

    pad_a_right = pad_a_right + (new_width_b > new_width_a) * (new_width_b - new_width_a)
    pad_b_right = pad_b_right + (new_width_b < new_width_a) * (new_width_a - new_width_b)
    pad_a_bottom = pad_a_bottom + (new_height_b > new_height_a) * (new_height_b - new_height_a)
    pad_b_bottom = pad_b_bottom + (new_height_b < new_height_a) * (new_height_a - new_height_b)

    image_a = cv2.copyMakeBorder(image_a, pad_a_top, pad_a_bottom, pad_a_left, pad_a_right, borderType=cv2.BORDER_CONSTANT)
    image_b = cv2.copyMakeBorder(image_b, pad_b_top, pad_b_bottom, pad_b_left, pad_b_right, borderType=cv2.BORDER_CONSTANT)

    stitched = average_pixels(image_a, image_b)

    return stitched

def main(image_1, image_2, output):

    image_a = cv2.imread(str(Path(image_1)))
    image_b = cv2.imread(str(Path(image_2)))

    point_matches = np.array(orb_sift_match(image_1, image_2, threshold=0.75, nfeatures=2500))
    point_matches = point_matches[:,-2:]
    point_matches = np.array(list(map(list, point_matches)))
    transform_matrix, shared_coordinates, centroid_a, centroid_b = ransac(point_matches, 4, 5000*int(np.sqrt(len(point_matches))), 0.75, max(int(0.1*len(point_matches)), 6))
    centroid_b_t = map_single_point(transform_matrix, centroid_b)
    transformed = apply_transformation(image_b, transform_matrix)
    # transformed_centroid = deepcopy(transformed)

    stitched = stitch(image_a, transformed, centroid_a, centroid_b_t)
    # Display output
    cv2.imwrite(str(Path(output)), stitched)

    # Visualize the point-matches' centroids on the output images
    # transformed_centroid = cv2.circle(transformed_centroid, np.rint(centroid_b_t).astype(int), 5, (0,255,0), -1)
    # image_a = cv2.circle(image_a, np.rint(centroid_a).astype(int), 5, (0,255,0), -1)
    # image_b = cv2.circle(image_b, np.rint(centroid_b).astype(int), 5, (0,255,0), -1)
    # cv2.imwrite('outputs/part3/image_a_centroid.jpg', image_a)
    # cv2.imwrite('outputs/part3/image_b_centroid.jpg', image_b)
    # cv2.imwrite('outputs/part3/image_b_t_centroid.jpg', transformed_centroid)
    # cv2.imwrite('outputs/part3/image_stitched.jpg', stitch(image_a, transformed_centroid, centroid_a, centroid_b_t))

    

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    try:
        image_1, image_2, output = sys.argv[1:]
    except:
        raise Exception(f'Usage: python3 part3.py <image_1> <image_2> <output_image>')

    main(Path(image_1), Path(image_2), Path(output))
