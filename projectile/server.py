import argparse
import os
from io import BytesIO

from tornado import ioloop, web
from PIL import Image
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm

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
        self.max_z = int(np.ceil(np.log2(max(self.array.shape))))

    def get(self, z, x, y, s):
        """
        Responds to the request with a specific image tile (as PNG).

        Parameters
        ----------
        z, x, y, s : str
            The zoom level, x index, y index, and size (in px) of the requested
            tile. `s` must be a power of 2.
        """
        z, x, y, s = map(int, (z, x, y, s))
        self.send_image(self.make_image(self.resize_array(self.get_array_slice(
            self.get_slices(z, x, y)), s)))

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

    def get_array_slice(self, slices):
        """
        Slices out the part of the array specified by slices, padding it to
        match the expected size of the slice.

        Parameters
        ----------
        slices : (slice, slice)
            The row and column slices appropriate for slicing out the desired
            tile from the complete array.

        Returns
        -------
        np.ndarray
            The sliced and shrunken array.
        """
        if slices is None:
            return np.full((1, 1), 1, dtype=np.uint8)
        size = slices[0].stop - slices[0].start
        sliced_array = self.array[slices]
        pad = [(0, 0)] * len(sliced_array.shape)
        if sliced_array.shape[0] < size:
            pad[0] = (0, size - sliced_array.shape[0])
        if sliced_array.shape[1] < size:
            pad[1] = (0, size - sliced_array.shape[1])
        return np.pad(sliced_array, pad, mode='constant', constant_values=1)

    @staticmethod
    def resize_array(array, s):
        """
        Resizes an array to a target size using tiling or max-pooling.

        Parameters
        ----------
        array : np.ndarray
            The array to resize. Can have either 2 or 3 dimensions. First two
            dimensions must be equal and powers of 2.
        s : int
            The size to resize to.

        Returns
        -------
        np.ndarray
            The resized array.
        """
        log_scale_factor = int(np.log2(array.shape[0])) - int(np.log2(s))
        if log_scale_factor < 0:  # array smaller than target, tile it
            factor = 2 ** -log_scale_factor
            shape = [array.shape[0], factor, array.shape[1], factor]
            strides = [array.strides[0], 0, array.strides[1], 0]
            final_shape = [s, s]
            if len(array.shape) == 3:
                shape.append(array.shape[2])
                strides.append(array.strides[2])
                final_shape.append(array.shape[2])
            return np.lib.stride_tricks.as_strided(array, shape, strides)\
                .reshape(final_shape)
        elif log_scale_factor > 0:  # array bigger than target, shrink it
            factor = 2 ** log_scale_factor
            shape = [array.shape[0] / factor, array.shape[1] / factor,
                     factor, factor]
            strides = [array.strides[0] * factor, array.strides[1] * factor,
                       array.strides[0], array.strides[1]]
            if len(array.shape) == 3:
                shape.append(array.shape[2])
                strides.append(array.strides[2])
            return np.lib.stride_tricks.as_strided(array, shape, strides)\
                .max(axis=(2, 3))
        else:
            return array

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
        stream = BytesIO()
        image.save(stream, format='png')
        self.write(stream.getvalue())
        self.set_header('Content-type', 'image/png')


class CmapTileHandler(TileHandler):
    def get(self, z, x, y, s, cmap, vmin, vmax):
        """
        Responds to the request with a specific image tile (as PNG), after
        applying a matplotlib colormap to the image.

        Parameters
        ----------
        z, x, y, s : str
            The zoom level, x index, y index, and size (in px) of the requested
            tile. `s` must be a power of 2.
        cmap : str
            Reference to a matplotlib colormap to use.
        vmin, vmax : str
            The minimum and maximum values of the colorscale to use.
        """
        z, x, y, s = map(int, (z, x, y, s))
        vmin, vmax = map(float, (vmin, vmax))
        self.send_image(self.make_image(self.resize_array(self.get_array_slice(
            self.get_slices(z, x, y)), s), cmap, vmin, vmax))

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
    def initialize(self, client, tile_size=256, cmap=None):
        self.client = client
        self.tile_size = tile_size
        self.cmap = cmap

    def get(self):
        cmap_string = '/%s/0/255' % self.cmap if self.cmap else ''
        self.render(self.client, tile_size=self.tile_size,
                    cmap_string=cmap_string)


def make_app(array, cmap=None, tile_size=256, client=None, debug=False):
    if client is None:
        client = '%s/client.html' % os.path.abspath(os.path.dirname(__file__))
    return web.Application([
        (r'/', ClientHandler,
         {'client': client, 'tile_size': tile_size, 'cmap': cmap}),
        (r'/()$', web.StaticFileHandler, {'path': client}),
        (r'/([0-9]+)/([0-9]+)/([0-9]+)/([0-9]+).png', TileHandler,
         {'array': array}),
        (r'/([0-9]+)/([0-9]+)/([0-9]+)/([0-9]+)/(\w+)/([^/]+)/([^/]+).png',
         CmapTileHandler, {'array': array})
    ], debug=debug)


def run(array, cmap=None, tile_size=256, client=None, port=8000, debug=False):
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
        The resolution of image tiles to serve, in px. Must be a power of 2.
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
        '-t', '--tile_size', type=int, default=256, help='''The resolution of
        image tiles to serve, in pixels. Must be a power of 2. The default is
        256 for 256 x 256 pixel image tiles.''')
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
