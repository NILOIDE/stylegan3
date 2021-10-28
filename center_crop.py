from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import os
import argparse

DATASET_DIR = r"C:\Users\Nil\PycharmProjects\stylegan2-ada-pytorch\data\Alternate_realities_2400"
DATA_SAVE_DIR = r"C:\Users\Nil\PycharmProjects\stylegan2-ada-pytorch\data\Alternate_realities_2400\1024"


def main(dataset_dir, data_save_dir, resolution, inter=cv2.INTER_AREA):
    """ Find shortest side and crop largest square from center of that image without going out of bounds. """
    os.makedirs(str(data_save_dir), exist_ok=True)
    for filename in tqdm(os.listdir(dataset_dir)):
        img = cv2.imread(str(dataset_dir / filename))
        if img is not None:
            assert img.shape[0] > 3
            (h, w) = img.shape[:2]
            if h < w:
                scale_factor = resolution / h
                dim = (int(np.ceil(w * scale_factor)), resolution)
            else:
                scale_factor = resolution / w
                dim = (resolution, int(np.ceil(h * scale_factor)))
            resized_img = cv2.resize(img, dim, interpolation=inter)
            crop_origin = ((resized_img.shape[0] - resolution) // 2, (resized_img.shape[1] - resolution) // 2)
            cropped_im = resized_img[crop_origin[0]:crop_origin[0]+resolution,
                                     crop_origin[1]:crop_origin[1]+resolution]
            assert cropped_im.shape == (resolution, resolution, 3),\
                f"Expected {(resolution, resolution, 3)}, received {cropped_im.shape}"
            cv2.imwrite(str(data_save_dir / filename), cropped_im)


def main_padded(dataset_dir, data_save_dir, resolution, im_zoom, inter=cv2.INTER_AREA):
    """ Find largest side and padd other side such that image is square. Zoom into image to obtain center-crop"""
    os.makedirs(str(data_save_dir), exist_ok=True)
    for filename in tqdm(os.listdir(dataset_dir)):
        img = cv2.imread(str(dataset_dir / filename))
        if img is not None:
            assert img.shape[0] > 3
            (h, w) = img.shape[:2]
            if h < w:
                padded_im = cv2.copyMakeBorder(img, (w-h)//2, (w-h)//2, 0, 0, cv2.BORDER_CONSTANT, 0)
            else:
                padded_im = cv2.copyMakeBorder(img, 0, 0, (h-w)//2, (h-w)//2, cv2.BORDER_CONSTANT, 0)

            assert padded_im.shape[0] == padded_im.shape[1], f'Received: {padded_im.shape}'
            resized_img = cv2.resize(padded_im, (int(resolution*im_zoom), int(resolution*im_zoom)), interpolation=inter)

            crop_origin = ((resized_img.shape[0] - resolution) // 2, (resized_img.shape[1] - resolution) // 2)
            cropped_im = resized_img[crop_origin[0]:crop_origin[0]+resolution, crop_origin[1]:crop_origin[1]+resolution]
            assert cropped_im.shape == (resolution, resolution, 3),\
                f"Expected {(resolution, resolution, 3)}, received {cropped_im.shape}"
            cv2.imwrite(str(data_save_dir / filename), cropped_im)


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to video", type=str, default=DATASET_DIR)
    a.add_argument("--pathOut", help="path to images", type=str, default=DATA_SAVE_DIR)
    a.add_argument("--resolution", help="resolution of final image's side", type=int, default=1024)
    a.add_argument("--imZoom", help="Zero-Pad image into square and zoom in. 1.0 keeps original image max width/height",
                   type=float, default=-1)
    args = a.parse_args()
    if args.imZoom > 0:
        main_padded(Path(args.pathIn), Path(args.pathOut), args.resolution, args.imZoom)
    else:
        main(Path(args.pathIn), Path(args.pathOut), args.resolution)
