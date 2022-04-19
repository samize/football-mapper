#!/usr/local/bin/python3

import sys
import itertools
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, freeze_support
from functools import partial
import warnings

from PIL import ImageDraw, Image, ImageOps
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm, trange


SAVE_IMG = False
MULTIPROCESS = True
runtime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

def euclidean_distance(point1, point2):
    # Deprecated, no longer used
    
    return np.linalg.norm(point1 - point2)


def distance(point1, point2, kind='euclidean'):
    # Deprecated, no longer used

    if kind == 'euclidean':
        return euclidean_distance(point1, point2)
    else:
        raise Exception(f'{kind} is not an acceptable kind for distance(point1, point2, kind)')


def pad_image(image, top, bottom, left, right, color=(0,0,0)):
    # Deprecated, no longer used

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    width_padded = image.shape[1] + left + right
    height_padded = image.shape[0] + top + bottom
    padded_image = Image.new('RGB', (width_padded, height_padded), color)
    padded_image.paste(Image.fromarray(image), (left, top))
    padded_image.save('pad_image_temp.png')
    return cv2.imread('pad_image_temp.png')


def old_match_points_code(descs1, descs2):
    
    # Two for loops to iterate through each feature point and calculate the Euclidean Distance
    point_list = []
    match_list = []
    distance_list = []

    for p1, desc1 in zip(keypoints1, descriptors1):
        nearest_distance = np.inf
        second_distance = np.inf
        nearest_match_point = None
        nearest_match_desc = None
        second_match_point = None
        second_match_desc = None

        for p2, desc2 in zip(keypoints2, descriptors2):
            point_distance = distance(desc1, desc2)
            if nearest_match_point is None:
                nearest_match_point = p2
                nearest_distance = point_distance
                nearest_match_desc = desc2
                continue
            elif second_match_point is None and point_distance > nearest_distance:
                second_match_point = p2
                second_distance = point_distance
                second_match_desc = desc2
                continue

            if point_distance < nearest_distance:
                second_distance = nearest_distance
                second_match_point = nearest_match_point
                second_match_desc = nearest_match_desc
                nearest_distance = point_distance
                nearest_match_point = p2
                nearest_match_desc = desc2
            elif point_distance < second_distance:
                second_match_point = p2
                second_match_desc = desc2

        # feature match
        distance_closest = distance(desc1, nearest_match_desc)
        distance_2ndclosest = distance(desc1, second_match_desc)
        match = distance_closest / distance_2ndclosest
        if match < threshold:
            match_list.append(match)
            distance_list.append(nearest_distance)
            point_list.append([p1, nearest_match_point, match, nearest_distance])
            cv2.line(test_img, p1[0].astype(int), ((nearest_match_point[0][0]+img1.shape[1]).astype(int),
                                                   nearest_match_point[0][1].astype(int)), (255, 0, 0))

    if len(point_list) > 0:
        output_path = Path(f'outputs/{runtime}/{int(threshold*100)}')
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(f'{output_path}/{image_a.name}_{image_b.name}.png', test_img):
            raise Exception("Image not saved.")

        raise Exception(type(point_list))
        match_sum = np.sum(point_list[:,2], dtype=object)
        raise Exception(str(point_list[:,2]) + '\n' + str(match_sum))
        nearest_sum = np.sum(point_list[:,3], dtype=object)
        return [True, image_a.name, image_b.name, match_sum, nearest_sum]
    else:
        return [False, image_a.name, image_b.name, np.inf, np.inf]


def match_points(descs1, descs2):

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    return matcher.knnMatch(descs1, descs2, 2)


def load_image(image_path: Path):

    return cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)


def orb_sift_match(image_a, image_b, threshold=0.75, nfeatures=500):

    global SAVE_IMG

    image_a, image_b = map(Path, [image_a, image_b])

    img1 = load_image(image_a)
    img2 = load_image(image_b)
    
    x_max = max(img1.shape[1], img2.shape[1])
    y_max = max(img1.shape[0], img2.shape[0])

    # you can increase nfeatures to adjust how many features to detect
    orb = cv2.ORB_create(nfeatures=nfeatures)

    (keypoints1, descriptors1) = orb.detectAndCompute(img1, None)
    (keypoints2, descriptors2) = orb.detectAndCompute(img2, None)

    desc_matches = match_points(descriptors1, descriptors2)
    point_matches = []

    for closest, next_closest in desc_matches:

        ratio = closest.distance / next_closest.distance
        if ratio < threshold:
            point_matches.append((closest.queryIdx, closest.trainIdx, ratio, closest.distance,
                                 keypoints1[closest.queryIdx].pt, keypoints2[closest.trainIdx].pt))

    return point_matches


def check_pair_for_matches(pair):

    """
    pairings = case[1]
    threshold = case[0]
    matches = {pair: orb_sift_match(pair[0], pair[1], threshold=threshold) for pair in pairings}
    # Select only matched pairs
    #matched_pairs = [str(pair) for pair in matches if matches[pair]]
    matched_pairs = [f'{pair} | {len(matches[pair])}' for pair in matches]
    with open(f'{runtime}/matched_pairs_{threshold}.txt.', 'w+') as file:
        file.write('\n'.join(matched_pairs))
    """
    
    matches_ij = orb_sift_match(pair[0], pair[1])
    matches_ji = orb_sift_match(pair[1], pair[0])
    if not matches_ij or not matches_ji:
        sums = 1.7 * 10 ** 308
    else:
        sums = (np.sum(np.array(matches_ij)[:,3]) + np.sum(np.array(matches_ji)[:,3])) / (len(matches_ij) + len(matches_ji))
    return (pair[0], pair[1], sums)


def cluster_images(images, k):

    global MULTIPROCESS
    # Get all possible pairs of images
    pairings = [pair for pair in itertools.product(images, images) if pair[0] != pair[1] and pair[0] < pair[1]]

    if MULTIPROCESS:
        with Pool(8) as p:
            matched_pairs = list(tqdm(p.imap_unordered(check_pair_for_matches, pairings), total=len(pairings)))
    else:
        matched_pairs = list(map(check_pair_for_matches, tqdm(pairings)))

    file_names = []
    for tup in matched_pairs:
        if tup[0] not in file_names:
            file_names.append(tup[0])
        if tup[1] not in file_names:
            file_names.append(tup[1])
    file_names = list(pd.Series(file_names).drop_duplicates().sort_values())
    file_dict = {}
    for index, file_name in enumerate(file_names):
        file_dict[file_name] = index

    distances = np.zeros((len(file_names), len(file_names)))
    for tup in matched_pairs:
        idx1 = file_dict[tup[0]]
        idx2 = file_dict[tup[1]]
        distances[idx1][idx2] = tup[2]
        distances[idx2][idx1] = tup[2]

    clustering = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='complete')

    labels = clustering.fit(distances).labels_

    clusters = {}

    for index, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(file_names[index])

    return clusters


def is_same_object(file_a, file_b):
    # String sorcery

    return Path(file_a).name.split('_')[0] == Path(file_b).name.split('_')[0]


def accuracy_pairwise_cluster(clusters):

    files = {}
    for label in clusters:
        for file in clusters[label]:
            files[file] = label

    tp = 0
    tn = 0
    n = len(files)

    for file_a in files:
        for file_b in files:
            if file_a != file_b:
                same_object = is_same_object(file_a, file_b)
                # tp += is same object AND has same label
                tp += same_object * (files[file_a] == files[file_b])
                # tp += not same object AND not has same label
                tn += (1-same_object) * (files[file_a] != files[file_b])

    return (tp+tn) / (n * (n-1))


def main(images, output, k=10):

    clusters = cluster_images(images, k=k)
    try:
        print(f'Accuracy: {accuracy_pairwise_cluster(clusters)}')
    except:
        print(f'Accuracy: File pattern could not be determined for accuracy calculation.')
    with open(output, 'w+') as file:
        output_text = '\n'.join([' '.join([value for value in clusters[label]]) for label in clusters])
        print(output_text)
        file.write(output_text)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # Step 1 Determine ORB Matching
    try:
        k = int(sys.argv[1])
        images = sys.argv[2:-1]
        output = sys.argv[-1]
    except:
        raise Exception(f'Usage: python3 part1.py <k> <img_1> <img_2> ... <img_n> <output_file>')

    freeze_support()
    main(images, output, k)
