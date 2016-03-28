#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Download modules for built-in datasets.
Download functions accept two arguments:
* `save_directory` : Where to save the downloaded files
* `clear` : If `True`, clear the downloaded files. Defaults to `False`.
"""
# my_fuel/downloaders/iris_downloader.py
from fuel.downloaders.base import default_downloader

def fill_subparser(subparser):
    subparser.set_defaults(
        urls=['https://archive.ics.uci.edu/ml/machine-learning-databases/'
              'iris/iris.data'],
        filenames=['irisx.data'])
    return default_downloader