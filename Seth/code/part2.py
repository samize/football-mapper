#!/usr/local/bin/python3

import sys
import warnings

import cv2
import numpy as np


def convert_vector(x):

    x = list(x) + [1]
    return np.array(x)

def test_transition_matrix(transition_matrix, pairs):
    
    result = 0
    for pair in pairs: 
        x, b = map(convert_vector, pair)

        Ax = np.matmul(transition_matrix, x)
        # Divide by z_ to get back to (x_, y_, 1) space.
        Ax = Ax / Ax[2]
        Ax = np.rint(Ax)
        b = np.rint(b)
        result += np.linalg.norm(Ax - b)

    return result / len(pairs)

def apply_transformation(img, transform_array):

    # Get transformation to apply
    # Take the inverse to perform inverse warping
    # transform_array = np.linalg.inv(transform_array)

    # Initialize new image
    top_left = np.array([0,0,1])
    top_right = np.array([img.shape[1], 0, 1])
    bottom_left = np.array([0, img.shape[0], 1])
    bottom_right = np.array([img.shape[1], img.shape[0], 1])

    inv_transform_array = np.linalg.inv(transform_array)
    top_left = np.matmul(inv_transform_array, top_left)
    top_left = (top_left / top_left[2])[:2]
    top_right = np.matmul(inv_transform_array, top_right)
    top_right = (top_right / top_right[2])[:2]
    bottom_left = np.matmul(inv_transform_array, bottom_left)
    bottom_left = (bottom_left / bottom_left[2])[:2]
    bottom_right = np.matmul(inv_transform_array, bottom_right)
    bottom_right = (bottom_right / bottom_right[2])[:2]
    max_y = np.rint(max([top_left[0], top_right[0], bottom_left[0], bottom_right[0]])).astype(int)+1
    max_x = np.rint(max([top_left[1], top_right[1], bottom_left[1], bottom_right[1]])).astype(int)+1

    #new_img = np.zeros(shape=(max_y, max_x, 3))
    new_img = np.zeros(shape=(max_x, max_y, 3))
    #print(new_img.shape, img.shape, max_x, max_y, top_left, top_right, bottom_left, bottom_right)

    # Loop through coordinates and
    # apply inverse transformation
    for r in range(new_img.shape[0]):
        for c in range(new_img.shape[1]):
            current_coor = np.array([c, r, 1])
            old_coor = np.matmul(transform_array, current_coor)
            old_x = old_coor[0] / old_coor[2]
            old_y = old_coor[1] / old_coor[2]
            # Update new image if old coordinate
            # is within the image coordinate bounds
            if old_y < img.shape[0] and \
                old_x < img.shape[1] and \
                old_y > 0 and \
                old_x > 0:
                    new_img[r, c] = np.rint(apply_interpolation(old_x, old_y, img)).astype(int)

    return new_img


def apply_interpolation(x, y, img):

    x_ceiling = np.ceil(x)
    y_ceiling = np.ceil(y)

    x_remainder = x % 1
    y_remainder = y % 1

    # Apply Bilinear Interpolation
    if x % 1 > 0 and y % 1 > 0 and \
        x_ceiling < img.shape[1] and y_ceiling < img.shape[0]:

            top_left = img[int(y_ceiling-1), int(x_ceiling-1)]
            top_right = img[int(y_ceiling-1), int(x_ceiling)]
            bot_left = img[int(y_ceiling), int(x_ceiling-1)]
            bot_right = img[int(y_ceiling), int(x_ceiling)]

            top_left_weight = (1 - x_remainder) * (1 - y_remainder)
            top_right_weight = (x_remainder) * (1 - y_remainder)
            bot_left_weight = (1 - x_remainder) * (y_remainder)
            bot_right_weight = (x_remainder) * (y_remainder)

            return top_left * top_left_weight + \
                top_right * top_right_weight + \
                bot_left * bot_left_weight + \
                bot_right * bot_right_weight

    # Apply Horizontal Linear Interpolation
    elif x % 1 > 0 and y % 1 == 0 and x_ceiling < img.shape[1]:

        left = img[int(y), int(x_ceiling - 1)]
        right = img[int(y), int(x_ceiling)]

        left_weight = 1 - x_remainder
        right_weight = x_remainder

        return left * left_weight + \
            right * right_weight

    # Apply Vertical Linear Interpolation
    elif x % 1 == 0 and y % 1 > 0 and y_ceiling < img.shape[0]:

        top = img[int(y_ceiling - 1), int(x)]
        bot = img[int(y_ceiling), int(x)]

        top_weight = 1 - y_remainder
        bot_weight = y_remainder

        return top * top_weight + \
            bot * bot_weight

    # Interpolation not necessary
    else: return img[int(y), int(x)]


def build_matching_coordinates(n):
    matching_points = []

    i = 5
    while i < len(sys.argv):
        matching_points += [[
            # Need to formation matching point arrays for linalg.solve
            # Degrees of freedom effect homogenous coordinate size?
            (int(sys.argv[i]), int(sys.argv[i+1])), 
            (int(sys.argv[i+2]), int(sys.argv[i+3]))
        ]]
        i += 4

        if len(matching_points) == n:
            break

    return matching_points


def get_translation_matrix(point_pairs):

    a = point_pairs[0][1][0] - point_pairs[0][0][0]
    b = point_pairs[0][1][1] - point_pairs[0][0][1]

    transition_matrix = np.array([[1, 0, a],
                                   [0, 1, b],
                                   [0, 0, 1]])

    return transition_matrix


def get_euclidean_matrix(point_pairs):

    solution_matrix = []
    solution_vector = []
    for pair in point_pairs:
        x, y = pair[0]
        x_, y_ = pair[1]

        solution_matrix += [[x, -1 * y, 1, 0],
                            [y,      x, 0, 1]]
        solution_vector += [x_, y_]

    solution_matrix = np.array(solution_matrix)
    solution_vector = np.array(solution_vector)

    a, b, c, d = np.linalg.solve(solution_matrix, solution_vector)

    transition_matrix = np.array([[a, -b, c],
                                  [b,  a, d],
                                  [0,  0, 1]])
    return transition_matrix


def get_affine_matrix(point_pairs):

    solution_matrix = []
    solution_vector = []
    for pair in point_pairs:
        x, y = pair[0]
        x_, y_ = pair[1]

        solution_matrix += [[x, y, 1, 0, 0, 0],
                            [0, 0, 0, x, y, 1]]
        solution_vector += [x_, y_]

    solution_matrix = np.array(solution_matrix)
    solution_vector = np.array(solution_vector)

    a, b, c, d, e, f = np.linalg.solve(solution_matrix, solution_vector)

    transition_matrix = np.array([[a, b, c],
                                  [d, e, f],
                                  [0,  0, 1]])
    return transition_matrix


def get_projection_matrix(point_pairs):

    solution_matrix = []
    solution_vector = []
    for index, pair in enumerate(point_pairs):
        x, y = pair[0]
        x_, y_ = pair[1]

        z_x = [0, 0, 0, 0]
        z_y = [0, 0, 0, 0]
        z_1 = [0, 0, 0, 0]
        z_x[index], z_y[index], z_1[index] = -x_, -y_, -1

        solution_matrix += [[x, y, 1, 0, 0, 0, 0, 0] + z_x,
                            [0, 0, 0, x, y, 1, 0, 0] + z_y,
                            [0, 0, 0, 0, 0, 0, x, y] + z_1]
        solution_vector += [0, 0, -1]

    solution_matrix = np.array(solution_matrix)
    solution_vector = np.array(solution_vector)

    a, b, c, d, e, f, g, h, _, _, _, _ = np.linalg.solve(solution_matrix, solution_vector)

    transition_matrix = np.array([[a, b, c],
                                  [d, e, f],
                                  [g, h, 1]])
    return transition_matrix


def get_transition_matrix(n, point_pairs):

    if n == 1 and len(point_pairs) >= 1:
        return get_translation_matrix(point_pairs)
    elif n == 2 and len(point_pairs) >= 2:
        return get_euclidean_matrix(point_pairs)
    elif n == 3 and len(point_pairs) >= 3:
        return get_affine_matrix(point_pairs)
    elif n == 4 and len(point_pairs) >= 4:
        return get_projection_matrix(point_pairs)
    else:
        raise Exception(f'Passed N and Pairs Received Do Not Match.\nPassed N {n}, Number of Pairs: {len(point_pairs)}')

def main(passed_n, first_img, second_img, output_img, passed_coordinates, output_method='save'):

    if len(passed_coordinates) > 0:

        matching_coordinates = []

        for i in range(0, len(passed_coordinates), 2):
            matching_coordinates += [[passed_coordinates[i].split(','), passed_coordinates[i+1].split(',')]]

        matching_coordinates = np.array(matching_coordinates)
        matching_coordinates = matching_coordinates.astype('int')

        # Get transformation matrix
        transformation_matrix = get_transition_matrix(passed_n, matching_coordinates)

        # Get first image as array
        first_img = cv2.imread(first_img)

        
        # Apply transformation
        output = apply_transformation(first_img, transformation_matrix)

        if output_method == 'save' and output_img is not None:
        # Save output
            cv2.imwrite(output_img, output)
        elif output_method == 'return':
            return output
        else:
            raise Exception(f'output_method must be "save" or "return". If output_method is "save", output_img must not be None.\noutput_method: {output_method}\noutput_img: {output_img}')


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # Store arguments in variables
    try:
        passed_n = int(sys.argv[1])
        first_img = sys.argv[2]
        second_img = sys.argv[3]
        output_img = sys.argv[4]
        passed_coordinates = sys.argv[5:]
    except:
        raise Exception(f'Usage: python3 part2.py <n> <img1> <img2> <outImg> <x11,y11> <x12,y12> ... <x1n,y1n> <x21,y21> <x22,y22> ... <x2n,y2n>')

    main(passed_n, first_img, second_img, output_img, passed_coordinates)
