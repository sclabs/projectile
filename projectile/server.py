import argparse
from six.moves import cStringIO as StringIO
import os

from tornado import ioloop, web
from PIL import Image
import numpy as np

import projectile
from projectile.image_to_array import image_to_array
from projectile.get_test_data import main as get_test_data


class TileHandler(web.RequestHandler):
    def initialize(self, array):
        """
        Initializes the request handler.

        Parameters
        ----------
        array : np.ndarray
            The array this handler will serve.
        """
        self.array = array

    def get(self, z, x, y):
        """
        Responds to the request with a specific image tile (as PNG).

        Parameters
        ----------
        z, x, y : int
            The zoom level, x index, and y index of the requested tile.
        """
        image = self.make_image(self.get_slices(z, x, y))
        stream = StringIO()
        image.save(stream, format='png')
        for line in stream.getvalue():
            self.write(line)
        self.set_header('Content-type', 'image/png')

    def get_slices(self, z, x, y):
        """
        Function which computes the slice objects appropriate for slicing out
        the desired tile from the complete array.

        Parameters
        ----------
        z, x, y : int
            The zoom level, x index, and y index of the requested tile.

        Returns
        -------
        (slice, slice)
            The row and column slices appropriate for slicing out the desired
            tile from the complete array.
        """
        z, x, y = map(int, (z, x, y))
        max_z = int(np.ceil(np.log2(max(self.array.shape))))
        size = 2 ** (max_z - z)
        return slice(y * size, (y+1) * size), slice(x * size, (x+1) * size)

    def make_image(self, slices):
        """
        Slices the array and puts it into a PIL image.

        Parameters
        ----------
        slices : (slice, slice)
            The row and column slices appropriate for slicing out the desired
            tile from the complete array.

        Returns
        -------
        PIL.Image
            The sliced image.
        """
        return Image.fromarray((self.array[slices]*255).astype(np.uint8))


def make_app(array, client=None, debug=False):
    if client is None:
        client = '%s/client.html' % os.path.abspath(os.path.dirname(__file__))
    return web.Application([
        (r'/()$', web.StaticFileHandler, {'path': client}),
        (r'/([0-9]+)/([0-9]+)/([0-9]+).png', TileHandler, dict(array=array))
    ], debug=debug)


def run(array, client=None, port=8000, debug=False):
    """
    Start a Tornado server serving a numpy array.

    Parameters
    ----------
    array : np.ndarray
        The array to serve.
    client : str, optional
        Path to a client HTML file to serve at the root URL. Pass None to use
        the included demo client.
    port : int
        Port to start the server on.
    debug : bool
        Pass True to start the server in debug mode.
    """
    app = make_app(array, client=client, debug=debug)
    app.listen(port)
    ioloop.IOLoop.instance().start()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input', help='''Image file or .npy file to serve. Pass 'sanfran' or
        'pentagon' to serve test images.''')
    parser.add_argument(
        '-m', '--mode', choices=['RGB', 'L'], default='RGB', help='''If input is
        an image, specify whether to read it in RGB or grayscale (L) mode.
        Default is RGB.''')
    parser.add_argument(
        '-c', '--client', help='''Specify a custom client HTML file to
        serve.''')
    parser.add_argument(
        '-p', '--port', type=int, default=8000, help='''The port to start the
        server on. Default is 8000.''')
    parser.add_argument(
        '-D', '--debug', action='store_true', help='''Pass this flag to start
        the server in debug mode.''')
    parser.add_argument(
        '-v', '--version', action='version',
        version='projectile %s' % projectile.__version__, help='''Show
        version information and exit.''')
    args = parser.parse_args()

    print('loading input')
    root, ext = os.path.splitext(args.input)
    if root in ['pentagon', 'sanfran']:
        if not os.path.exists('images/%s.npy' % root):
            print('downloading test images')
            get_test_data()
        array = np.load('images/%s.npy' % root)
    elif ext == '.npy':
        print('array detected')
        array = np.load(args.input)
    else:
        print('image detected')
        array = image_to_array(args.input, mode=args.mode)

    print('starting app')
    run(array, client=args.client, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
