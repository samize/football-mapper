
import sys
import os
from pathlib import Path
import json

import cv2

import PitchFeatures


if __name__ == '__main__':

    print(f'Getting folder documentation/test_images/{sys.argv[1]}...')

    curr_path = str(os.getcwd())
    new_path = curr_path + '\\documentation\\test_clips'
    os.chdir(new_path)
    os.mkdir(sys.argv[1]+'_hough')
    os.mkdir(sys.argv[1]+'_corners')
    
    os.chdir(sys.argv[1])
    source_files = os.listdir()
    os.chdir("..")

    for file_name in source_files:
        true_file_name = sys.argv[1] + '/' + file_name
        im = cv2.imread(true_file_name)
        hough, corner, corners = PitchFeatures.run_corners(im)

        json_file = file_name.replace(f'{Path(file_name).suffix}', '.json')

        cv2.imwrite(sys.argv[1]+'_hough/'+file_name, hough)
        cv2.imwrite(sys.argv[1]+'_corners/'+file_name, corner)
        with open(sys.argv[1]+'_corners/'+json_file, 'w+') as file:
            file.write(json.dumps(corners, indent=4))