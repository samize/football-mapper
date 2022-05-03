from pathlib import Path
import os
import itertools
from copy import deepcopy
import warnings
import json
import sys

import cv2
import numpy as np
from tqdm import tqdm
import scipy.special

import load_above
from part2 import get_projection_matrix, test_transition_matrix, get_transition_matrix, convert_vector
from part2 import main as part2main

# Identify Feature Points

# Project Feature Points using RANSAC

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

def create_overhead_projection(frame_file: Path, corner_pixels_file: Path, field_file: Path, output_file: Path):

    #frame_file = documentation/test_clips/807-2_hough
    #corner_pixels_file = documentation/test_clips/807-2_corners
    #field_file = Seth/overhead-space.png
    #output_file = test.npy

    if not isinstance(frame_file, Path):
        frame_file = Path(frame_file)

    if not isinstance(corner_pixels_file, Path):
        corner_pixels_file = Path(corner_pixels_file)

    if not isinstance(field_file, Path):
        field_file = Path(field_file)

    if not isinstance(output_file, Path):
        output_file = Path(output_file)

    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True, exist_ok=True)

    output_directory = output_file.parent
    

    image = cv2.imread(str(frame_file))
    ground, red_pixels, line_pixels = load_above.main(str(field_file))
    
    with open(corner_pixels_file, 'r') as file:
        pixels = json.loads(file.read())

    pixel_pairs = list(itertools.product(pixels, red_pixels))

    best_match_count = 0
    #best_match = None
    best_pairings = None
    best_matrix = None
    match_counts = {}

    combinations = []

    for pair1 in pixel_pairs:
        for pair2 in pixel_pairs:
            for pair3 in pixel_pairs:
                for pair4 in pixel_pairs:

    #total_combinations = scipy.special.comb(4*, )

    for index, combination in enumerate(itertools.combinations(pixel_pairs, 4)):

        #image_copy = deepcopy(image)

        try:
            matrix = get_transition_matrix(4, combination)
            transition_matrix = np.linalg.inv(matrix)
        except np.linalg.LinAlgError as error:
            continue

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
            #image_copy = cv2.circle(image_copy, (new_red_x, new_red_y), 5, (255,0,0), thickness=-1)

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
                if load_above.is_dark_pixel(pixel):
                    continue
                new_point = np.matmul(transition_matrix, np.array([x, y, 1]))
                x_, y_ = new_point[0] / new_point[2], new_point[1] / new_point[2]
                try:
                    x_, y_ = int(x_), int(y_)
                except ValueError as error:
                    skip = True
                    break

                if 0 <= x_ and x_ < image.shape[1] and 0 <= y_ and y_ < image.shape[0] and load_above.is_white_pixel(image[y_,x_,:], threshold=175):
                    matches += 1
                    #image_copy = cv2.circle(image_copy, (x_,y_), 3, (0,0,255), -1)
                    #image_copy[y_, x_, :] = (0,0,255)

        if skip:
            continue
        if matches >= best_match_count:
            print(matches)
            best_match_count = matches
            #best_match = image_copy
            best_pairings = combination
            best_matrix = matrix
            #cv2.imwrite(f'output_0_x/{index}.jpg', image_copy)

        match_counts[index] = matches
    
    #with open('matches.json', 'w+') as file:
    #    file.write(json.dumps(match_counts, indent=4))

    with open(output_directory / output_file.name.replace(output_file.suffix, '_pairings.json'), 'w+') as file:
        file.write(json.dumps(best_pairings, indent=4))

    np.save(output_file, best_matrix)

if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    frame_file = Path(sys.argv[1])
    corner_pixels_file = Path(sys.argv[2])
    field_file = Path(sys.argv[3])
    output_file = Path(sys.argv[4])

    create_overhead_projection(frame_file, corner_pixels_file, field_file, output_file)

    #frame_file = documentation/test_clips/807-2_hough
    #corner_pixels_file = documentation/test_clips/807-2_corners
    #field_file = Seth/overhead-space.png
    #output_file = test.npy
