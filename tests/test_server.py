import numpy as np
from tornado.testing import AsyncHTTPTestCase

from projectile.server import make_app, get_test_data


class TestServer(AsyncHTTPTestCase):
    def get_app(self):
        get_test_data()
        return make_app(np.load('images/sanfran.npy'))

    def test_client(self):
        response = self.fetch('/')
        self.assertEqual(response.code, 200)

    def test_tile(self):
        response = self.fetch('/3/2/3/256.png')
        self.assertEqual(response.code, 200)
