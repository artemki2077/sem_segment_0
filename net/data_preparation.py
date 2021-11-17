import os
import json
import random
import argparse
from typing import List

import cv2
from tqdm import tqdm

from config import DATASETS


def prepare_data(datasets: List, data_path: str, proportion_test_images: float, json_name: str) -> None:
    """
    :param json_name: Name for the json file.
    :param datasets: Names of datasets that will be used to generate the json file.
    :param proportion_test_images: proportion of test images.
    :param data_path: path data.
    """

    path = []
    for i in datasets:
        path.append(os.path.join(data_path, i, 'images'))
        path.append(os.path.join(data_path, i, 'masks'))
    images = []
    for image_path in path:
        
        g = image_path.split('\\') # '\\' if windows,  '/' if linux
        if 'images' in g:
            for img in os.listdir(image_path):
                print(os.path.join(image_path, img))
                images.append(os.path.join(image_path, img))

    shuffle_images = random.sample(images, len(images))

    # create dictionary
    train_test_json = {'train': [], 'test': []}

    # filling in dictionary for json file
    for j, image_path in tqdm(enumerate(shuffle_images)):
        try:
            img = image_path.replace('images', 'masks')
            img = os.path.splitext(img)[0]
            img = img + '.png'
            img_dict = {'image_path': image_path, 'mask_path': img}
            if cv2.imread(image_path) is None:
                print('broken images: ' + image_path)
                continue
            elif cv2.imread(img) is None:
                print('broken images: ' + img)
                continue
            elif j < len(shuffle_images) * proportion_test_images:
                train_test_json['test'].append(img_dict)
            else:
                train_test_json['train'].append(img_dict)
        except KeyError:
            print(' no masks for ', image_path)

    # write json file
    with open(os.path.join(data_path, json_name), 'w') as f:
        json.dump(train_test_json, f, indent=4)


def parse_args() -> argparse.Namespace:
    """
    Parsing command line arguments with argparse.
    """
    parser = argparse.ArgumentParser('script for model testing.')
    parser.add_argument('-p', '--data_path', type=str, default='data', help='path to Dataset')
    parser.add_argument('--prop_valid', type=float, default=0.2, help='Number of test validation images')
    parser.add_argument('-j', '--json_name', type=str, default='data.json', help='Name of the json file to save.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    prepare_data(datasets=DATASETS, data_path=args.data_path, proportion_test_images=args.prop_valid,
                 json_name=args.json_name)
