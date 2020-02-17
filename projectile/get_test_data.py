from six.moves.urllib.request import urlretrieve
import os

from projectile.image_to_array import run as image_to_array


def main():
    if not os.path.exists('images'):
        os.makedirs('images')
    urlretrieve(
        'http://sipi.usc.edu/database/download.php?vol=aerials&img=2.2.17',
        'images/sanfran.tiff')
    urlretrieve(
        'http://sipi.usc.edu/database/download.php?vol=aerials&img=3.2.25',
        'images/pentagon.tiff')
    image_to_array('images/sanfran.tiff', mode='RGB')
    image_to_array('images/pentagon.tiff', mode='L')


if __name__ == '__main__':
    main()
