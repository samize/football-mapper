from pathlib import Path
import os
import itertools
from copy import deepcopy
import warnings
import json

import cv2
from tqdm import tqdm

from load_above import *
from part2 import main, get_projection_matrix, test_transition_matrix, get_transition_matrix, convert_vector


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
    best_sample_b = best_sample[:,1,:]

    return best_transform, best_sample

if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    image = cv2.imread('../Lucas/output_0.jpg')
    ground = cv2.imread('above-black-and-red.png')

    pixels = [(592, 229), (331, 173), (1013, 637), (981, 393), (420, 256)]

    red_pixels = [(79, 236), (79, 269), (83, 99), (83, 104), (83, 162), (83, 212), (83, 236), (83, 269), (83, 294), (83, 343), (83, 401), (83, 406), (88, 99), (88, 406), (108, 212), (108, 294), (158, 162), (158, 220), (158, 221), (158, 284), (158, 285), (158, 343), (320, 99), (320, 211), (320, 212),
                  (320, 294), (320, 406), (482, 162), (482, 220), (482, 221), (482, 284), (482, 285), (482, 343), (532, 212), (532, 294), (552, 99), (552, 406), (557, 99), (557, 104), (557, 162), (557, 212), (557, 236), (557, 269), (557, 294), (557, 343), (557, 401), (557, 406), (561, 236), (561, 269)]


    red_pixels = [(482, 343), 
                  (532, 212), (532, 294), 
                  (557, 162), (557, 212)]

    red_pixels = [(482, 343), (532, 212), (532, 294), (557, 162), (557, 212)]

    pixel_pairs = list(itertools.product(pixels, red_pixels))

    best_match_count = 0
    best_match = None
    best_matrix = None
    match_counts = {}

    for index, combination in tqdm(enumerate(itertools.combinations(pixel_pairs, 4)), total=2300):

        #if index < 7100 or index > 7450:
        #   continue

        image_copy = deepcopy(image)

        #if not use_pairing(combination):
        #   continue

        try:
            matrix = get_transition_matrix(4, combination)
            transition_matrix = np.linalg.inv(matrix)
        except np.linalg.LinAlgError as error:
            continue

        #pixel, red_pixel = combination[-1]

        #distance = (pixel[0] - new_red_x) ** 2 + (pixel[1] - new_red_y) ** 2
        #if distance > 250:
        #   continue

        pixels = [vector[0] for vector in combination]
        red_pixels = [vector[1] for vector in combination]

        skip = False
        for pixel, red_pixel in zip(pixels, red_pixels):
            new_red_pixel = np.matmul(transition_matrix, convert_vector(red_pixel))
            new_red_x = new_red_pixel[0] / new_red_pixel[2]
            new_red_y = new_red_pixel[1] / new_red_pixel[2]
            try:
                distance = (pixel[0] - new_red_x) ** 2 + (pixel[1] - new_red_y) ** 2
                if distance > 400:
                    skip=True
                    break
                new_red_x, new_red_y = int(new_red_x), int(new_red_y)
            except ValueError as error:
                skip=True
                break
            except OverflowError as error:
                skip=True
                break
            image_copy = cv2.circle(image_copy, (new_red_x, new_red_y), 5, (255,0,0), thickness=-1)

        if skip:
            continue
        
        matches = 0
        for x in range(ground.shape[1]):
            if skip:
                break
            for y in range(ground.shape[0]):
                if skip:
                    break
                pixel = ground[y, x, :]
                if is_dark_pixel(pixel):
                    continue
                new_point = np.matmul(transition_matrix, np.array([x, y, 1]))
                x_, y_ = new_point[0] / new_point[2], new_point[1] / new_point[2]
                try:
                    x_, y_ = int(x_), int(y_)
                except ValueError as error:
                    skip = True
                    break

                if 0 <= x_ and x_ < image_copy.shape[1] and 0 <= y_ and y_ < image_copy.shape[0] and is_white_pixel(image_copy[y_,x_,:], threshold=175):
                    matches += 1
                    image_copy = cv2.circle(image_copy, (x_,y_), 3, (0,0,255), -1)
                    #image_copy[y_, x_, :] = (0,0,255)

        if skip:
            continue
        if matches >= best_match_count:
            best_match_count = matches
            best_match = image_copy
            best_matrix = matrix
            cv2.imwrite(f'output_0_x/{index}.jpg', image_copy)

        match_counts[index] = matches
    
    with open('matches.json', 'w+') as file:
        file.write(json.dumps(match_counts, indent=4))

    broadcast = cv2.imread('broadcast_img_0.png')

    for x in range(broadcast.shape[1]):
        for y in range(broadcast.shape[0]):
            if is_black_pixel(broadcast[y,x,:], threshold=0):
                new_point = np.matmul(best_matrix, np.array([x,y,1]))
                new_x, new_y = new_point[0] / new_point[2], new_point[1] / new_point[2]
                new_x, new_y = int(new_x), int(new_y)
                ground = cv2.circle(ground, (new_x, new_y), 1, (0,0,255), -1)

    cv2.imwrite('mapped.jpg', ground)