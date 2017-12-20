projectile
==========

A tile-on-demand tile server built with PIL and Tornado.

Motivation
----------

We want to store a high-resolution image in memory on the server as a numpy
array. Then when a client requests a particular tile, we can make the PNG of the
requested tile by slicing the numpy array and using PIL to write the resulting
PNG back to the client through a StringIO stream.

This is primarily intended for building interactive visualizations in research
settings where we might want to skip the time- and/or disk-intensive tile
generation step required by typical tile servers.

Demo
----

Install projectile

    $ pip install projectile

Serve a test image from the [USC-SIPI Image Database](http://sipi.usc.edu/database/):

    $ projectile sanfran

Manually request a particular tile by navigating to <http://localhost:8000/2/1/2.png>.

Try zooming and panning in the demo client by navigating to <http://localhost:8000/>.

Serve one of your own images with

    $ projectile some_image.tiff

or, if you have data in a numpy `.npy` file,

    $ projectile some_image.npy

Stress testing
--------------

To test the performance limits of on-demand tiling, download this
[high resolution map of Great Britain](https://commons.wikimedia.org/wiki/File:A_new_map_of_Great_Britain_according_to_the_newest_and_most_exact_observations_(8342715024).jpg)
(8,150 × 13,086 pixels, file size: 102.74 MB) from Wikimedia Commons.

Grayscale performance test:

    $ projectile britain.jpg -m L

Reducing tile resolution when running in color:

    $ projectile britain.jpg --tile_size 7

Dependencies
------------

 - `numpy>=1.13.3`
 - `Pillow>=4.3.0`
 - `tornado>=4.5.2`
 - `matplotlib>=2.1.0`

API
---

### Using a custom client

If you like the projectile backend but just want to use a simple custom client
contained in a single HTML file `custom_client.html`, you can run

    $ projectile array.npy --client custom_client.html

to make projectile serve your client instead of the included demo client.

### Using projectile in your existing Tornado web application

The core functionality is exposed in the `TileHandler` class defined in
[server.py](projectile/server.py), which you can use in your own Tornado web
applications:

```python
from tornado import web
from projectile.server import TileHandler

...

app = web.Application([
    (r'/([0-9]+)/([0-9]+)/([0-9]+).png', TileHandler, dict(array=array)),
    ...
])

...
```

### Launching projectile from your own Python code

You can also launch a server from your own Python code with the `run()` function
defined in [server.py](projectile/server.py):

```python
from projectile.server import run

run(array)
```

Credits
-------

The demo client is lifted from <http://bl.ocks.org/mbostock/5914438>, with the
addition of a small filtering check to prevent the client from requesting tiles
which lie beyond the image boundaries.
