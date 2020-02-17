from setuptools import setup, find_packages
from io import open
from os import path

import versioneer


here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='projectile',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='A tile-on-demand tile server built with PIL and Tornado',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/sclabs/projectile',
    author='Thomas Gilgenast',
    author_email='thomasgilgenast@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Operating System :: OS Independent',
        'Topic :: Multimedia :: Graphics :: Viewers',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    keywords='tileserver tile server on-demand tiling',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.13.3',
        'Pillow>=4.3.0',
        'tornado>=4.5.2',
        'six>=1.11.0',
        'matplotlib>=2.1.1',
    ],
    package_data={
        'projectile': ['client.html'],
    },
    entry_points={
        'console_scripts': [
            'projectile=projectile.server:main',
        ],
    },
)
