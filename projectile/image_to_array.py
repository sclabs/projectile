import argparse
import os

import numpy as np
from PIL import Image
from PIL.ImageOps import invert


def image_to_array(image_file, mode='RGB'):
    return np.asarray(invert(Image.open(image_file).convert(mode)))


def run(image_file, mode='RGB'):
    np.save(os.path.splitext(image_file)[0],
            image_to_array(image_file, mode=mode))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('-m', '--mode', choices=['RGB', 'L'], default='RGB')
    args = parser.parse_args()

    run(args.image, mode=args.mode)


if __name__ == '__main__':
    main()
