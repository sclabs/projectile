import argparse
from six.moves import cStringIO as StringIO
import os

from tornado import ioloop, web
from PIL import Image
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm

import projectile
from projectile.image_to_array import image_to_array
from projectile.get_test_data import main as get_test_data


class TileHandler(web.RequestHandler):
    def initialize(self, array, tile_size=8):
        """
        Initializes the request handler.

        Parameters
        ----------
        array : np.ndarray
            The array this handler will serve.
        tile_size : int
            The base 2 log of the maximum tile size to use, in pixels.
        """
        self.array = array
        self.max_z = int(np.ceil(np.log2(max(self.array.shape))))
        self.tile_size = tile_size

    def get(self, z, x, y):
        """
        Responds to the request with a specific image tile (as PNG).

        Parameters
        ----------
        z, x, y : str
            The zoom level, x index, and y index of the requested tile.
        """
        z, x, y = map(int, (z, x, y))
        self.send_image(self.make_image(self.get_array_slice(
            self.get_slices(z, x, y), z, tile_size=self.tile_size)))

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
        (slice, slice) or None
            The row and column slices appropriate for slicing out the desired
            tile from the complete array. If the slice requested is
            out-of-bounds then None is returned.
        """
        size = 2 ** (self.max_z - z)
        max_x = self.array.shape[1] / size
        max_y = self.array.shape[0] / size
        if x < 0 or x > max_x or y < 0 or y > max_y:
            return None
        return slice(y * size, (y+1) * size), slice(x * size, (x+1) * size)

    def get_array_slice(self, slices, z, tile_size=8):
        """
        Slices out the part of the array specified by slices, and shrinks it to
        a specified maximum tile size if necessary.

        Parameters
        ----------
        slices : (slice, slice)
            The row and column slices appropriate for slicing out the
            desired
            tile from the complete array.
        z : int
            The requested zoom level (determines how much shrinkage
            to apply).
        tile_size : int
            The base 2 log of the maximum tile size to use, in pixels.

        Returns
        -------
        np.ndarray
            The sliced and shrunken array.
        """
        if slices is None:
            return np.full((1, 1), 255, dtype=np.uint8)
        size = slices[0].stop - slices[0].start
        shrinkage_factor = 2 ** (self.max_z - z - tile_size)
        sliced_array = self.array[slices]
        pad = [(0, 0)] * len(sliced_array.shape)
        if sliced_array.shape[0] < size:
            pad[0] = (0, size - sliced_array.shape[0])
        if sliced_array.shape[1] < size:
            pad[1] = (0, size - sliced_array.shape[1])
        sliced_array = np.pad(sliced_array, pad, mode='constant',
                              constant_values=1)
        return self.shrink_array(sliced_array, shrinkage_factor)

    @staticmethod
    def shrink_array(array, factor):
        """
        Shrinks an array by a factor using max-pooling.

        Parameters
        ----------
        array : np.ndarray
            The array to shrink. Can have either 2 or 3 dimensions.
        factor : int
            The factor to shrink by.

        Returns
        -------
        np.ndarray
            The shrunken array.
        """
        if factor < 2:
            return array
        shape = [array.shape[0] / factor, array.shape[1] / factor,
                 factor, factor]
        strides = [array.strides[0] * factor, array.strides[1] * factor,
                   array.strides[0], array.strides[1]]
        if len(array.shape) == 3:
            shape.append(array.shape[2])
            strides.append(array.strides[2])
        return np.lib.stride_tricks.as_strided(array, shape, strides)\
            .max(axis=(2, 3))

    @staticmethod
    def make_image(array_slice):
        """
        Creates a PIL Image from an array slice.

        Parameters
        ----------
        array_slice : np.ndarray
            The array slice to make the Image from.

        Returns
        -------
        PIL.Image
            The Image.
        """
        return Image.fromarray((array_slice * 255).astype(np.uint8))

    def send_image(self, image):
        """
        Sends a PIL Image as the response to this request.

        Parameters
        ----------
        image : PIL.Image
            The Image to respond with.
        """
        stream = StringIO()
        image.save(stream, format='png')
        for line in stream.getvalue():
            self.write(line)
        self.set_header('Content-type', 'image/png')


class CmapTileHandler(TileHandler):
    def get(self, z, x, y, cmap, vmin, vmax):
        """
        Responds to the request with a specific image tile (as PNG), after
        applying a matplotlib colormap to the image.

        Parameters
        ----------
        z, x, y : str
            The zoom level, x index, and y index of the requested tile.
        cmap : str
            Reference to a matplotlib colormap to use.
        vmin, vmax : str
            The minimum and maximum values of the colorscale to use.
        """
        z, x, y, vmin, vmax = map(int, (z, x, y, vmin, vmax))
        self.send_image(self.make_image(self.get_array_slice(
            self.get_slices(z, x, y), z, tile_size=self.tile_size),
            cmap, vmin, vmax))

    def make_image(self, array_slice, cmap, vmin, vmax):
        """
        Creates a PIL Image from an array slice, after applying a matplotlib
        colormap to the image.

        Parameters
        ----------
        array_slice : np.ndarray
            The array slice to make the Image from.
        cmap : str
            Reference to a matplotlib colormap to use.
        vmin, vmax : int
            The minimum and maximum values of the colorscale to use.

        Returns
        -------
        PIL.Image
            The Image.
        """
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        m = cm.ScalarMappable(norm=norm, cmap=getattr(cm, cmap))
        rgb = m.to_rgba(array_slice)
        return super(CmapTileHandler, self).make_image(rgb)


class ClientHandler(web.RequestHandler):
    def initialize(self, client, cmap=None):
        self.client = client
        self.cmap = cmap

    def get(self):
        cmap_string = '/%s/0/255' % self.cmap if self.cmap else ''
        self.render(self.client, cmap_string=cmap_string)


def make_app(array, cmap=None, tile_size=8, client=None, debug=False):
    if client is None:
        client = '%s/client.html' % os.path.abspath(os.path.dirname(__file__))
    return web.Application([
        (r'/', ClientHandler, {'client': client, 'cmap': cmap}),
        (r'/()$', web.StaticFileHandler, {'path': client}),
        (r'/([0-9]+)/([0-9]+)/([0-9]+).png', TileHandler,
         {'array': array, 'tile_size': tile_size}),
        (r'/([0-9]+)/([0-9]+)/([0-9]+)/(\w+)/([0-9]+)/([0-9]+).png',
         CmapTileHandler, {'array': array, 'tile_size': tile_size})
    ], debug=debug)


def run(array, cmap=None, tile_size=8, client=None, port=8000, debug=False):
    """
    Start a Tornado server serving a numpy array.

    Parameters
    ----------
    array : np.ndarray
        The array to serve.
    cmap : matplotlib colormap, optional
        Pass the name of a matplotlib colormap to colorize `array` if it is
        grayscale.
    tile_size : int
        The base 2 log of the maximum tile size to use, in pixels.
    client : str, optional
        Path to a client HTML file to serve at the root URL. Pass None to use
        the included demo client.
    port : int
        Port to start the server on.
    debug : bool
        Pass True to start the server in debug mode.
    """
    app = make_app(array, cmap=cmap, tile_size=tile_size, client=client,
                   debug=debug)
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
        '-c', '--cmap', help='''If input is a .npy file or an image read in
        grayscale (L) mode, specify the name of a matplotlib colormap to use to
        color the tiles.''')
    parser.add_argument(
        '-t', '--tile_size', type=int, default=8, help='''The maximum size of
        image tiles to serve, on a log base 2 scale. The default is 8 for
        256 x 256 pixel image tiles.''')
    parser.add_argument(
        '--client', help='''Specify a custom client HTML file to serve.''')
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

    print('starting server at http://localhost:%s/' % args.port)
    run(array, cmap=args.cmap, tile_size=args.tile_size, client=args.client,
        port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
