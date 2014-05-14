# coding: utf-8

""" LM10 stuff """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
from astropy import log as logger
import astropy.units as u

from streams.potential._lm10_acceleration import lm10_acceleration, lm10_variational_acceleration

default_bounds = dict(q1=(0.7,2.0),
                      qz=(0.7,2.0),
                      phi=(0.79,2.36),
                      v_halo=(0.1,0.2),
                      q2=(0.7,2.0),
                      r_halo=(5,20))

def _parse_grid_spec(p):
    name = str(p[0])
    num = int(p[1])

    if len(p) > 2:
        _min = u.Quantity.from_string(p[2])
        _max = u.Quantity.from_string(p[3])
    else:
        _min,_max = default_bounds[name]

    return name, np.linspace(_min, _max, num)

# hamiltons equations
# def F(t, X, *args):
#     # args order should be: q1, qz, phi, v_halo, q2, R_halo
#     x,y,z,px,py,pz = X.T
#     nparticles = x.size
#     acc = np.zeros((nparticles,3))
#     dH_dq = lm10_acceleration(X, nparticles, acc, *args)
#     return np.hstack((np.array([px, py, pz]).T, dH_dq))

def F(t, X, *args):
    # args order should be: q1, qz, phi, v_halo, q2, R_halo
    x,y,z,px,py,pz = X[...,:6].T
    dx,dy,dz,dpx,dpy,dpz = X[...,6:].T

    nparticles = x.size
    acc = np.zeros((nparticles,6))
    acc = lm10_variational_acceleration(X, nparticles, acc, *args)

    term1 = np.array([px, py, pz]).T
    term2 = acc[:,:3]
    term3 = np.array([dpx,dpy,dpz]).T
    term4 = acc[:,3:]

    return np.hstack((term1,term2,term3,term4))