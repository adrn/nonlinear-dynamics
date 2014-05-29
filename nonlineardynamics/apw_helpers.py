# coding: utf-8

""" Helpers for our 3-component potential. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
from astropy import log as logger
import astropy.units as u
from astropy.constants import G

usys = [u.kpc, u.Myr, u.radian, u.M_sun]
G = G.decompose(usys).value

def potential(X, Md, a, b, Mn, c, Mh, Rs, q):
    R,z = X[...,:2].T

    # Miyamoto-Nagai disk
    z_term = a + np.sqrt(z**2 + b**2)
    disk = -G*Md / np.sqrt(R**2 + z_term**2)

    # Hernquist nucleus
    nucl = -G*Mn / np.sqrt(R**2 + z**2 + c**2)

    # Flattened NFW halo
    r = np.sqrt(R**2 + (z/q)**2)
    halo = -G*Mh * np.log(1 + r/Rs) / r

    return disk + nucl + halo

def effective_potential(X, Lz, Md, a, b, Mn, c, Mh, Rs, q):
    R,z = X[...,:2].T

    # Angular momentum
    angm = Lz**2/(2*R**2)

    return potential(X, Md, a, b, Mn, c, Mh, Rs, q) + angm

def acceleration(X, Md, a, b, Mn, c, Mh, Rs, q):
    R,z = X[...,:2].T

    # Miyamoto-Nagai disk
    z_term = a + np.sqrt(z**2 + b**2)
    _tmp = G*Md / (R**2 + z_term**2)**1.5
    disk_dR = _tmp * R
    disk_dz = _tmp * z_term / np.sqrt(z**2 + b**2)

    # Hernquist nucleus
    _tmp = G*Mn / (R**2 + z**2 + c**2)**1.5
    nucl_dR = _tmp * R
    nucl_dz = _tmp * z

    # Flattened NFW halo
    r = np.sqrt(R**2 + (z/q)**2)
    _tmp = G*Mh*(-r + (r+Rs)*np.log(1+r/Rs)) / (r**2*(r+Rs))
    halo_dR = _tmp * R / r
    halo_dz = _tmp * z / r / q**2

    dV_dR = disk_dR + nucl_dR + halo_dR
    dV_dz = disk_dz + nucl_dz + halo_dz

    return -np.array([dV_dR, dV_dz]).T

def effective_acceleration(X, Lz, Md, a, b, Mn, c, Mh, Rs, q):
    R,z = X[...,:2].T

    # Angular momentum
    angm_dR = -Lz**2/R**3
    angm_dz = 0.

    acc = acceleration(X, Md, a, b, Mn, c, Mh, Rs, q)

    return -np.array([angm_dR, np.zeros_like(angm_dR)]).T + acc

def variational_acceleration(X, Lz, Md, a, b, Mn, c, Mh, Rs, q):
    R,z = X[...,:2].T
    dR,dz = X[...,4:6].T

    # Miyamoto-Nagai disk
    d2V_dR2 = -3*G*Md*R**2/(R**2 + (a + np.sqrt(b**2 + z**2))**2)**2.5 + G*Md/(R**2 + (a + np.sqrt(b**2 + z**2))**2)**1.5 - 3*G*Mn*R**2/(R**2 + c**2 + z**2)**2.5 + G*Mn/(R**2 + c**2 + z**2)**1.5

    d2V_dRdz = -3*G*Md*R*z*(a + np.sqrt(b**2 + z**2))/((R**2 + (a + np.sqrt(b**2 + z**2))**2)**2.5*np.sqrt(b**2 + z**2)) - 3*G*Mn*R*z/(R**2 + c**2 + z**2)**2.5

    d2V_dz2 = -G*Md*z**2*(a + np.sqrt(b**2 + z**2))/((R**2 + (a + np.sqrt(b**2 + z**2))**2)**1.5*(b**2 + z**2)**1.5) + G*Md*z**2/((R**2 + (a + np.sqrt(b**2 + z**2))**2)**1.5*(b**2 + z**2)) - 3*G*Md*z**2*(a + np.sqrt(b**2 + z**2))**2/((R**2 + (a + np.sqrt(b**2 + z**2))**2)**2.5*(b**2 + z**2)) + G*Md*(a + np.sqrt(b**2 + z**2))/((R**2 + (a + np.sqrt(b**2 + z**2))**2)**1.5*np.sqrt(b**2 + z**2)) - 3*G*Mn*z**2/(R**2 + c**2 + z**2)**2.5 + G*Mn/(R**2 + c**2 + z**2)**1.5

    ddR = (d2V_dR2 + 3*Lz**2/R**4)*dR + d2V_dRdz*dz
    ddz = d2V_dRdz*dR + d2V_dz2*dz

    return -np.array([ddR, ddz]).T

def rotation_curve(R, Md, a, b, Mn, c, Mh, Rs, q):
    z = np.zeros_like(R)
    X = np.vstack((R,z)).T
    acc = acceleration(X, Md, a, b, Mn, c, Mh, Rs, q)

    return np.sqrt(R*np.abs(acc[:,0]))

def F(t, X, *args):
    R,z,pR,pz = X[...,:4].T

    acc = effective_acceleration(X, **dict(args))

    term1 = np.array([pR, pz]).T
    term2 = acc[...,:2]
    return np.hstack((term1,term2))

def F_var(t, X, *args):
    R,z,pR,pz = X[...,:4].T
    dR,dz,dpR,dpz = X[...,4:].T

    term1 = np.array([pR, pz]).T
    term2 = effective_acceleration(X, **dict(args))
    term3 = np.array([dpR,dpz]).T
    term4 = variational_acceleration(X, **dict(args))

    return np.hstack((term1,term2,term3,term4))