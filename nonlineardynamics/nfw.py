# coding: utf-8

""" Triaxial NFW stuff """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
from astropy import log as logger
import astropy.units as u
from astropy.constants import G

# fit to R < 150kpc
# note: in Ana's fit, phi = 90, so I swapped q1 and q2 and elimnated phi
vl2_params = dict(v_h=(433.057*u.km/u.s).to(u.kpc/u.Myr).value,
                  q1=1.,
                  q2=0.865017,
                  q3=1.17202,
                  r_h=19.8061)

usys = [u.kpc, u.Myr, u.radian, u.M_sun]

# def qs_to_Cs(q1, q2, q3, phi):
#     C_1 = (np.cos(phi)/q1)**2 + (np.sin(phi)/q2)**2
#     C_2 = (np.cos(phi)/q2)**2 + (np.sin(phi)/q1)**2
#     C_3 = 2*np.cos(phi)*np.sin(phi)*(q1**-2 + q2**-2)
#     return (C_1,C_2,C_3)

def potential(t, w, v_h, q1, q2, q3, r_h):
    x,y,z = w[...,:3].T
    r = (x/q1)**2 + (y/q2)**2 + (z/q3)**2 + r_h**2
    return -v_h**2 * r_h/r * np.log(1+r/r_h)

def acceleration(t, w, v_h, q1, q2, q3, r_h):
    x,y,z = w[...,:3].T
    r = (x/q1)**2 + (y/q2)**2 + (z/q3)**2 + r_h**2

    dPhi_dr = r_h*v_h**2*(-r + (r + r_h)*np.log((r + r_h)/r_h))/(r**2*(r + r_h))
    dPhi_dx = dPhi_dr * 2*x/q1**2
    dPhi_dy = dPhi_dr * 2*y/q2**2
    dPhi_dz = dPhi_dr * 2*z/q3**2

    return -np.array([dPhi_dx,dPhi_dy,dPhi_dz]).T

def variational_acceleration(t, w, v_h, q1, q2, q3, r_h):
    x,y,z = w[...,:3].T
    dx,dy,dz = w[...,6:9].T
    r = (x/q1)**2 + (y/q2)**2 + (z/q3)**2 + r_h**2

    d2Phi_dx2 = 2*r_h*v_h**2*(-q1**2*r**2*(r + r_h) + q1**2*r*(r + r_h)**2*np.log((r + r_h)/r_h) + 2*r**2*x**2 + 4*r*x**2*(r + r_h) - 4*x**2*(r + r_h)**2*np.log((r + r_h)/r_h))/(q1**4*r**3*(r + r_h)**2)
    d2Phi_dy2 = 2*r_h*v_h**2*(-q2**2*r**2*(r + r_h) + q2**2*r*(r + r_h)**2*np.log((r + r_h)/r_h) + 2*r**2*y**2 + 4*r*y**2*(r + r_h) - 4*y**2*(r + r_h)**2*np.log((r + r_h)/r_h))/(q2**4*r**3*(r + r_h)**2)
    d2Phi_dz2 = 2*r_h*v_h**2*(-q3**2*r**2*(r + r_h) + q3**2*r*(r + r_h)**2*np.log((r + r_h)/r_h) + 2*r**2*z**2 + 4*r*z**2*(r + r_h) - 4*z**2*(r + r_h)**2*np.log((r + r_h)/r_h))/(q3**4*r**3*(r + r_h)**2)
    d2Phi_dxdy = 4*r_h*v_h**2*x*y*(r**2 + 2*r*(r + r_h) - 2*(r + r_h)**2*np.log((r + r_h)/r_h))/(q1**2*q2**2*r**3*(r + r_h)**2)
    d2Phi_dxdz = 4*r_h*v_h**2*x*z*(r**2 + 2*r*(r + r_h) - 2*(r + r_h)**2*np.log((r + r_h)/r_h))/(q1**2*q3**2*r**3*(r + r_h)**2)
    d2Phi_dydz = 4*r_h*v_h**2*y*z*(r**2 + 2*r*(r + r_h) - 2*(r + r_h)**2*np.log((r + r_h)/r_h))/(q2**2*q3**2*r**3*(r + r_h)**2)

    return -np.array([d2Phi_dx2*dx + d2Phi_dxdy*dy + d2Phi_dxdz*dz,
                      d2Phi_dxdy*dx + d2Phi_dy2*dy + d2Phi_dydz*dz,
                      d2Phi_dxdz*dx + d2Phi_dydz*dy + d2Phi_dz2*dz]).T

def F(t, w, *args):
    x,y,z,px,py,pz = w[...,:6].T
    dx,dy,dz,dpx,dpy,dpz = w[...,6:].T

    acc = acceleration(t, w, **dict(args))
    var_acc = variational_acceleration(t, w, **dict(args))

    term1 = np.array([px, py, pz]).T
    term3 = np.array([dpx,dpy,dpz]).T

    return np.hstack((term1,acc,term3,var_acc))