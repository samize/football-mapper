from pathlib import Path
import os
import itertools
from copy import deepcopy

import cv2
from tqdm import tqdm

from load_above import *
from part2 import main, get_projection_matrix, test_transition_matrix


def use_pairing(pairings):

    left_pixels = []
    right_pixels = []
    for pair in pairings:
        left, right = pair
        if left in left_pixels or right in right_pixels:
            return False
        left_pixels.append(left)
        right_pixels.append(right)

    return True

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

if __name__ == '__main__':

    image = cv2.imread('../Lucas/broadcast_img_0.jpg')
    ground = cv2.imread('above-black-and-red.png')

    pixels = [(592, 229), (331, 173), (1013, 637), (981, 393), (420, 256)]

    red_pixels = [(79, 236), (79, 269), (83, 99), (83, 104), (83, 162), (83, 212), (83, 236), (83, 269), (83, 294), (83, 343), (83, 401), (83, 406), (88, 99), (88, 406), (108, 212), (108, 294), (158, 162), (158, 220), (158, 221), (158, 284), (158, 285), (158, 343), (320, 99), (320, 211), (320, 212),
                  (320, 294), (320, 406), (482, 162), (482, 220), (482, 221), (482, 284), (482, 285), (482, 343), (532, 212), (532, 294), (552, 99), (552, 406), (557, 99), (557, 104), (557, 162), (557, 212), (557, 236), (557, 269), (557, 294), (557, 343), (557, 401), (557, 406), (561, 236), (561, 269)]

    red_pixels = [(320, 99), (320, 211), (320, 212), (320, 294), 
                  (320, 406), (482, 162), (482, 220), (482, 221), 
                  (482, 284), (482, 285), (482, 343), (532, 212), 
                  (532, 294), (552, 99), (552, 406), (557, 162), 
                  (557, 212), (557, 236), (557, 269), (557, 294), (557, 343)]

    pixel_pairs = list(itertools.product(pixels, red_pixels))

    for index, combination in tqdm(enumerate(itertools.combinations(pixel_pairs, 5)), total=96560646):

        image_copy = deepcopy(image)

        if not use_pairing(combination):
            continue

        transition_matrix = get_transition_matrix(4, combination[:-1])

        pixel, red_pixel = combination[-1]

        new_red_pixel = np.matmul(transition_matrix, red_pixel)
        new_red_x = new_red_pixel[0] / new_red_pixel[2]
        new_red_y = new_red_pixel[1] / new_red_pixel[2]

        distance = (pixel[0] - new_red_x) ** 2 + (pixel[1] - new_red_y) ** 2
        if distance > 20:
            continue

        for x in range(ground.shape[1]):
            for y in range(ground.shape[0]):

                pixel = ground[y, x, :]
                if is_dark_pixel(pixel):
                    continue
                new_point = np.matmul(transition_matrix, np.array([x, y, 1]))
                x_, y_ = int(new_point[0]), int(new_point[1])

                if 0 <= x_ and x_ <= image_copy.shape[1] and 0 <= y_ and y_ <= image_copy.shape[0]:

                    image_copy[y_, x_, :] = pixel

        cv2.imwrite(f'output_0/{index}.jpg', image_copy)
