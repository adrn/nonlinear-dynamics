# coding: utf-8

""" Misc. utilities """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u

__all__ = ['_parse_grid_spec']

def _parse_grid_spec(p):
    name = str(p[0])
    num = int(p[1])

    if len(p) > 2:
        _min = u.Quantity.from_string(p[2])
        _max = u.Quantity.from_string(p[3])
    else:
        _min,_max = default_bounds[name]

    return name, np.linspace(_min, _max, num)