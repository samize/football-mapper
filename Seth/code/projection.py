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
    red_pixels = red_pixels[24:]

    
    with open(corner_pixels_file, 'r') as file:
        pixels = json.loads(file.read())

    pixels = [tuple(pixel) for pixel in pixels]
    red_pixels = [tuple(pixel) for pixel in red_pixels]
    pixel_pairs = list(itertools.product(pixels, red_pixels))

    best_match_count = 0
    #best_match = None
    best_pairings = None
    best_matrix = None
    match_counts = {}

    combinations = set()

    #pixel_pairs = [[(529, 150), (482, 162)], 
    #                [(918, 113), (557, 162)], 
    #                [(931, 183), (532, 212)],
    #                [(1069, 168), (557, 212)],
    #                    [(1148, 481), (482, 343)]]

    for pair1 in tqdm(pixel_pairs, desc='Pair1', unit='combinations'):
        for pair2 in pixel_pairs:
            if pair1[0][0] > pair2[0][0]:
                continue
            if pair1[0][0] == pair2[0][0] and pair1[0][1] > pair2[0][1]:
                continue
            if pair1[0] == pair2[0] or pair1[1] == pair2[1]:
                continue
            for pair3 in pixel_pairs:
                if pair2[0][0] > pair3[0][0]:
                    continue
                if pair2[0][0] == pair3[0][0] and pair2[0][1] > pair3[0][1]:
                    continue
                if pair1[0] == pair3[0] or pair2[0] == pair3[0] or pair1[1] == pair3[1] or pair2[1] == pair3[1]:
                    continue
                for pair4 in pixel_pairs:
                    if pair3[0][0] > pair4[0][0]:
                        continue
                    if pair3[0][0] == pair4[0][0] and pair3[0][1] > pair4[0][1]:
                        continue
                    if pair1[0] == pair4[0] or pair2[0] == pair4[0] or pair3[0] == pair4[0] or pair1[1] == pair4[1] or pair2[1] == pair4[1] or pair3[1] == pair4[1]:
                        continue
                    combination = (pair1, pair2, pair3, pair4)
                    if combination in combinations:
                        continue
                    combinations.add(combination)

    combinations = list(combinations)
    #total_combinations = scipy.special.comb(4*len(combinations), 4)

    for index, combination in enumerate(tqdm(combinations)):

        image_copy = deepcopy(image)

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
                if distance > 100:
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
        in_frame = 0
        for x, y in line_pixels:
            new_point = np.matmul(transition_matrix, np.array([x, y, 1]))
            x_, y_ = new_point[0] / new_point[2], new_point[1] / new_point[2]
            try:
                x_, y_ = int(x_), int(y_)
            except ValueError as error:
                skip = True
                break
            if 0 <= x_ and x_ < image.shape[1] and 0 <= y_ and y_ < image.shape[0]:
                if load_above.is_white_pixel(image[y_,x_,:], threshold=175):
                    matches += 1
                in_frame += 1
                image_copy = cv2.circle(image_copy, (x_,y_), 3, (0,0,255), -1)
                image_copy[y_, x_, :] = (0,0,255)
        """
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
        """
        if skip:
            continue
        if matches >= best_match_count:
            perc_in_frame = matches / in_frame
            perc_of_points = matches / len(line_pixels)
            print(perc_in_frame, perc_of_points)
            best_match_count = matches
            #best_match = image_copy
            best_pairings = combination
            best_matrix = matrix
            cv2.imwrite(str(output_directory / output_file.name.replace(output_file.suffix, f'_{index}_map.png')), image_copy)

        match_counts[index] = matches
    
    #with open('matches.json', 'w+') as file:
    #    file.write(json.dumps(match_counts, indent=4))

    with open(output_directory / output_file.name.replace(output_file.suffix, '_pairings.json'), 'w+') as file:
        file.write(json.dumps(best_pairings, indent=4))

    np.save(output_file, best_matrix)

def map_objects(matrix_file: Path, objects_file: Path, field_file: Path, output_file: Path):

    matrix = np.load(matrix_file)
    with open(objects_file, 'r') as file:
        objects_coords = json.loads(file.read())

    image = cv2.imread(str(field_file))

    for object_ in objects_coords:
        x_min, y_min, x_max, y_max = object_
        point = (int((x_max + x_min)/2), int(y_max))
        new_point = np.matmul(matrix, np.array([point[0], point[1], 1]))
        new_x, new_y = new_point[0] / new_point[2], new_point[1] / new_point[2]
        new_x, new_y = int(new_x), int(new_y)
        image = cv2.circle(image, (new_x, new_y), 1, (0,0,255), -1)

    cv2.imwrite(str(output_file), image)

if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    frame_file = Path(sys.argv[1])
    corner_pixels_file = Path(sys.argv[2])
    objects_file = Path(sys.argv[3])
    field_file = Path(sys.argv[4])
    output_file = Path(sys.argv[5])

    matrix_file = output_file.parent / output_file.name.replace(output_file.suffix, '.npy')
    create_overhead_projection(frame_file, corner_pixels_file, field_file, matrix_file)
    map_objects(matrix_file, objects_file, field_file, output_file)

    #frame_file = documentation/test_clips/807-2_hough
    #corner_pixels_file = documentation/test_clips/807-2_corners
    #field_file = Seth/overhead-space.png
    #output_file = test.npy
