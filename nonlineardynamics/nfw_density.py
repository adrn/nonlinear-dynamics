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

from numpy import sqrt,log

# vaguely similar to Ana's fit to VL2
vl2_params = dict(vh=0.11072316551541589*2,
                  R0=4.428926620616636,
                  a=1., b=0.7, c=1.5)

usys = [u.kpc, u.Myr, u.radian, u.M_sun]

def potential(t, w, vh, R0, a, b, c):
    x,y,z = w[...,:3].T
    r = sqrt(x*x + y*y + z*z)
    return -vh**2*(48*R0*a**2*r**4*log(sqrt(r/R0) + sqrt((R0 + r)/R0)) - 3*R0*(15*R0**2*log(sqrt(r/R0) + sqrt((R0 + r)/R0)) - sqrt(r/(R0 + r))*(15*R0**2 + 5*R0*r - 2*r**2))*(y**2*(a**2 - b**2) + z**2*(a**2 - c**2)) - 48*a**2*r**5*(sqrt((R0 + r)/r) - 1) + r**2*(-2*a**2 + b**2 + c**2)*(-3*R0*(5*R0**2 - 8*r**2)*log(sqrt(r/R0) + sqrt((R0 + r)/R0)) + 12*r**3 + sqrt(r/(R0 + r))*(15*R0**3 + 5*R0**2*r - 26*R0*r**2 - 12*r**3)))/(24*a**2*r**5)

def acceleration(t, w, vh, R0, a, b, c):
    x,y,z = w[...,:3].T
    r = sqrt(x*x + y*y + z*z)

    tmp6 = (R0 + r)/R0
    tmp = sqrt(tmp6)
    tmp3 = (sqrt(r/R0) + tmp)
    tmp2 = log(tmp3)
    tmp4 = R0*sqrt(r/R0)*tmp + r
    tmp5 = sqrt(r/(R0 + r))

    dPhi_dx = -vh**2*x*(12*R0**2*(tmp6)**(1.5)*tmp3*(4*a**2*r**5*sqrt((R0 + r)/r) - 8*a**2*r**4*(R0 + r)*tmp2 + (R0 + r)*(15*R0**2*tmp2 - tmp5*(15*R0**2 + 5*R0*r - 2*r**2))*(y**2*(a**2 - b**2) + z**2*(a**2 - c**2))) + 3*R0*(R0 + r)*(y**2*(a**2 - b**2) + z**2*(a**2 - c**2))*(90*R0**2*tmp*(R0 + r)*tmp3*tmp2 + R0*tmp*tmp5*tmp3*(15*R0**2 + 5*R0*r - 2*r**2) - 15*R0*(R0 + r)*(tmp4) - 2*tmp*tmp5*(R0 + r)*tmp3*(45*R0**2 + 10*R0*r - 2*r**2)) + 48*a**2*r**4*(R0 + r)**2*(tmp4) - r**2*(R0 + r)*(-2*a**2 + b**2 + c**2)*(2*R0*tmp*tmp5*(R0 + r)*tmp3*(45*R0**2 + 10*R0*r - 26*r**2) - R0*tmp*tmp5*tmp3*(15*R0**3 + 5*R0**2*r - 26*R0*r**2 - 12*r**3) - 6*R0*tmp*(R0 + r)*(15*R0**2 - 8*r**2)*tmp3*tmp2 + 3*(R0 + r)*(5*R0**2 - 8*r**2)*(tmp4)))/(48*R0*a**2*r**7*(tmp6)**(1.5)*(R0 + r)*tmp3)
    dPhi_dy = -vh**2*y*(12*R0**2*(tmp6)**(1.5)*tmp3*(4*a**2*r**5*sqrt((R0 + r)/r) - 8*a**2*r**4*(R0 + r)*tmp2 + (R0 + r)*(15*R0**2*tmp2 - tmp5*(15*R0**2 + 5*R0*r - 2*r**2))*(-r**2*(a**2 - b**2) + y**2*(a**2 - b**2) + z**2*(a**2 - c**2))) + 3*R0*(R0 + r)*(y**2*(a**2 - b**2) + z**2*(a**2 - c**2))*(90*R0**2*tmp*(R0 + r)*tmp3*tmp2 + R0*tmp*tmp5*tmp3*(15*R0**2 + 5*R0*r - 2*r**2) - 15*R0*(R0 + r)*(tmp4) - 2*tmp*tmp5*(R0 + r)*tmp3*(45*R0**2 + 10*R0*r - 2*r**2)) + 48*a**2*r**4*(R0 + r)**2*(tmp4) - r**2*(R0 + r)*(-2*a**2 + b**2 + c**2)*(2*R0*tmp*tmp5*(R0 + r)*tmp3*(45*R0**2 + 10*R0*r - 26*r**2) - R0*tmp*tmp5*tmp3*(15*R0**3 + 5*R0**2*r - 26*R0*r**2 - 12*r**3) - 6*R0*tmp*(R0 + r)*(15*R0**2 - 8*r**2)*tmp3*tmp2 + 3*(R0 + r)*(5*R0**2 - 8*r**2)*(tmp4)))/(48*R0*a**2*r**7*(tmp6)**(1.5)*(R0 + r)*tmp3)
    dPhi_dz = -vh**2*z*(12*R0**2*(tmp6)**(1.5)*tmp3*(4*a**2*r**5*sqrt((R0 + r)/r) - 8*a**2*r**4*(R0 + r)*tmp2 + (R0 + r)*(15*R0**2*tmp2 - tmp5*(15*R0**2 + 5*R0*r - 2*r**2))*(-r**2*(a**2 - c**2) + y**2*(a**2 - b**2) + z**2*(a**2 - c**2))) + 3*R0*(R0 + r)*(y**2*(a**2 - b**2) + z**2*(a**2 - c**2))*(90*R0**2*tmp*(R0 + r)*tmp3*tmp2 + R0*tmp*tmp5*tmp3*(15*R0**2 + 5*R0*r - 2*r**2) - 15*R0*(R0 + r)*(tmp4) - 2*tmp*tmp5*(R0 + r)*tmp3*(45*R0**2 + 10*R0*r - 2*r**2)) + 48*a**2*r**4*(R0 + r)**2*(tmp4) - r**2*(R0 + r)*(-2*a**2 + b**2 + c**2)*(2*R0*tmp*tmp5*(R0 + r)*tmp3*(45*R0**2 + 10*R0*r - 26*r**2) - R0*tmp*tmp5*tmp3*(15*R0**3 + 5*R0**2*r - 26*R0*r**2 - 12*r**3) - 6*R0*tmp*(R0 + r)*(15*R0**2 - 8*r**2)*tmp3*tmp2 + 3*(R0 + r)*(5*R0**2 - 8*r**2)*(tmp4)))/(48*R0*a**2*r**7*(tmp6)**(1.5)*(R0 + r)*tmp3)

    return -np.array([dPhi_dx,dPhi_dy,dPhi_dz]).T

def variational_acceleration(t, w, vh, R0, a, b, c):
    x,y,z = w[...,:3].T
    dx,dy,dz = w[...,6:9].T
    r = sqrt(x*x + y*y + z*z)
    C = vh**2

    tmp6 = (R0 + r)/R0
    tmp = sqrt((R0 + r)/R0)
    tmp3 = (sqrt(r/R0) + tmp)
    tmp2 = log(sqrt(r/R0) + tmp)
    tmp4 = R0*sqrt(r/R0)*tmp + r
    tmp5 = sqrt(r/(R0 + r))
    tmp7 = (15*R0**3 + 5*R0**2*r - 26*R0*r**2 - 12*r**3)
    tmp8 = (y**2*(a**2 - b**2) + z**2*(a**2 - c**2))

    d2Phi_dx2 = C*(R0**3*(tmp6)**(3.5)*(48*R0**2*a**2*r**3*x**2*sqrt((R0 + r)/r)*(R0 + r)*tmp3**2*(x**2 + y**2 + z**2) + 96*R0*a**2*r**6*x**2*sqrt((R0 + r)/r)*(R0 + r)*tmp3**2 + 192*R0*a**2*r**6*(R0 + r)**3*tmp3**2*tmp2 - 96*R0*a**2*r**5*x**2*sqrt((R0 + r)/r)*(R0 + r)**2*tmp3**2 - 576*R0*a**2*r**4*x**2*(R0 + r)**3*tmp3**2*tmp2 - 24*R0*(R0 + r)**3*tmp3**2*(15*R0**2*tmp2 + tmp5*(-15*R0**2 - 5*R0*r + 2*r**2))*(r**2*tmp8 + 4*x**2*(y**2*(-a**2 + b**2) + z**2*(-a**2 + c**2))) - 3*R0*tmp8*(180*R0**2*r**2*(R0 + r)**3*tmp3**2*tmp2 + R0**2*x**2*tmp5*(R0 + r)*tmp3**2*(15*R0**2 + 5*R0*r - 2*r**2) - 900*R0**2*x**2*(R0 + r)**3*tmp3**2*tmp2 + 15*R0**2*tmp*tmp3*(R0**3*sqrt(r/R0)*(tmp6)**(2.5)*(-2*r**2 + 3*x**2) + r**2*x**2*(R0 + r) + 2*r*(R0 + r)**2*(-r**2 + x**2) + 12*x**2*(R0 + r)**2*(tmp4)) + 2*R0*r*x**2*tmp5*(R0 + r)*tmp3**2*(15*R0**2 + 5*R0*r - 2*r**2) - 2*R0*x**2*tmp5*(R0 + r)**2*tmp3**2*(15*R0**2 + 5*R0*r - 2*r**2) - 4*R0*x**2*tmp5*(R0 + r)**2*tmp3**2*(45*R0**2 + 10*R0*r - 2*r**2) + 15*R0*x**2*(R0 + r)**2*(tmp4)**2 + 4*tmp5*(R0 + r)**3*tmp3**2*(225*R0**2*x**2 - 10*R0*r**3 + 40*R0*r*x**2 + 2*r**4 - 3*r**2*(15*R0**2 + 2*x**2)) - 2*tmp5*(R0 + r)*tmp3**2*(15*R0**2 + 5*R0*r - 2*r**2)*(r**3*(R0 + r) - 2*r**2*x**2 - r**2*(R0 + r)**2 + r*x**2*(R0 + r) + x**2*(R0 + r)**2)) + 96*a**2*r**5*sqrt((R0 + r)/r)*(R0 + r)**2*tmp3**2*(r**3 - r**2*(R0 + r) - 3*r*x**2 + 3*x**2*(R0 + r)) + 48*a**2*r**4*x**2*(R0 + r)**2*(tmp4)**2 - r**2*(-2*a**2 + b**2 + c**2)*(-4*R0**2*x**2*tmp5*(R0 + r)**2*tmp3**2*(45*R0**2 + 10*R0*r - 26*r**2) + R0**2*x**2*tmp5*(R0 + r)*tmp3**2*tmp7 + 2*R0*r*x**2*tmp5*(R0 + r)*tmp3**2*tmp7 - 2*R0*x**2*tmp5*(R0 + r)**2*tmp3**2*tmp7 + 3*R0*tmp*tmp3*(4*x**2*(R0 + r)**2*(15*R0**2 - 8*r**2)*(tmp4) + (5*R0**2 - 8*r**2)*(R0**3*sqrt(r/R0)*(tmp6)**(2.5)*(-2*r**2 + 3*x**2) + r**2*x**2*(R0 + r) + 2*r*(R0 + r)**2*(-r**2 + x**2))) + 4*R0*tmp5*(R0 + r)**3*tmp3**2*(225*R0**2*x**2 - 10*R0*r**3 + 40*R0*r*x**2 + 26*r**4 - 3*r**2*(15*R0**2 + 26*x**2)) + 12*R0*(R0 + r)**3*tmp3**2*(-75*R0**2*x**2 - 8*r**4 + 3*r**2*(5*R0**2 + 8*x**2))*tmp2 + 3*x**2*(R0 + r)**2*(5*R0**2 - 8*r**2)*(tmp4)**2 - 2*tmp5*(R0 + r)*tmp3**2*tmp7*(r**3*(R0 + r) - 2*r**2*x**2 - r**2*(R0 + r)**2 + r*x**2*(R0 + r) + x**2*(R0 + r)**2))) + 48*a**2*r**4*(R0 + r)**4*tmp3*(R0**3*sqrt(r/R0)*(tmp6)**(2.5)*(-2*r**2 + 3*x**2) + r**2*x**2*(R0 + r) + 2*r*(R0 + r)**2*(-r**2 + x**2)) + 24*x**2*(R0 + r)**5*tmp3*(-R0*tmp8*(-90*R0**2*tmp*(R0 + r)*tmp3*tmp2 - R0*tmp*tmp5*tmp3*(15*R0**2 + 5*R0*r - 2*r**2) + 15*R0*(R0 + r)*(tmp4) + 2*tmp*tmp5*(R0 + r)*tmp3*(45*R0**2 + 10*R0*r - 2*r**2)) + 8*a**2*r**4*(R0 + r)*(tmp4)))/(96*R0**3*a**2*r**9*(tmp6)**(3.5)*(R0 + r)**3*tmp3**2)
    d2Phi_dy2 = C*(R0**3*(tmp6)**(3.5)*(48*R0**2*a**2*r**3*y**2*sqrt((R0 + r)/r)*(R0 + r)*tmp3**2*(x**2 + y**2 + z**2) + 96*R0*a**2*r**6*y**2*sqrt((R0 + r)/r)*(R0 + r)*tmp3**2 + 192*R0*a**2*r**6*(R0 + r)**3*tmp3**2*tmp2 - 96*R0*a**2*r**5*y**2*sqrt((R0 + r)/r)*(R0 + r)**2*tmp3**2 - 576*R0*a**2*r**4*y**2*(R0 + r)**3*tmp3**2*tmp2 + 24*R0*(R0 + r)**3*tmp3**2*(15*R0**2*tmp2 + tmp5*(-15*R0**2 - 5*R0*r + 2*r**2))*(r**4*(a**2 - b**2) + r**2*(5*y**2*(-a**2 + b**2) + z**2*(-a**2 + c**2)) + 4*y**2*tmp8) - 3*R0*tmp8*(180*R0**2*r**2*(R0 + r)**3*tmp3**2*tmp2 + R0**2*y**2*tmp5*(R0 + r)*tmp3**2*(15*R0**2 + 5*R0*r - 2*r**2) - 900*R0**2*y**2*(R0 + r)**3*tmp3**2*tmp2 + 15*R0**2*tmp*tmp3*(R0**3*sqrt(r/R0)*(tmp6)**(2.5)*(-2*r**2 + 3*y**2) + r**2*y**2*(R0 + r) + 2*r*(R0 + r)**2*(-r**2 + y**2) + 12*y**2*(R0 + r)**2*(tmp4)) + 2*R0*r*y**2*tmp5*(R0 + r)*tmp3**2*(15*R0**2 + 5*R0*r - 2*r**2) - 2*R0*y**2*tmp5*(R0 + r)**2*tmp3**2*(15*R0**2 + 5*R0*r - 2*r**2) - 4*R0*y**2*tmp5*(R0 + r)**2*tmp3**2*(45*R0**2 + 10*R0*r - 2*r**2) + 15*R0*y**2*(R0 + r)**2*(tmp4)**2 + 4*tmp5*(R0 + r)**3*tmp3**2*(225*R0**2*y**2 - 10*R0*r**3 + 40*R0*r*y**2 + 2*r**4 - 3*r**2*(15*R0**2 + 2*y**2)) - 2*tmp5*(R0 + r)*tmp3**2*(15*R0**2 + 5*R0*r - 2*r**2)*(r**3*(R0 + r) - 2*r**2*y**2 - r**2*(R0 + r)**2 + r*y**2*(R0 + r) + y**2*(R0 + r)**2)) + 96*a**2*r**5*sqrt((R0 + r)/r)*(R0 + r)**2*tmp3**2*(r**3 - r**2*(R0 + r) - 3*r*y**2 + 3*y**2*(R0 + r)) + 48*a**2*r**4*y**2*(R0 + r)**2*(tmp4)**2 - r**2*(-2*a**2 + b**2 + c**2)*(-4*R0**2*y**2*tmp5*(R0 + r)**2*tmp3**2*(45*R0**2 + 10*R0*r - 26*r**2) + R0**2*y**2*tmp5*(R0 + r)*tmp3**2*tmp7 + 2*R0*r*y**2*tmp5*(R0 + r)*tmp3**2*tmp7 - 2*R0*y**2*tmp5*(R0 + r)**2*tmp3**2*tmp7 + 3*R0*tmp*tmp3*(4*y**2*(R0 + r)**2*(15*R0**2 - 8*r**2)*(tmp4) + (5*R0**2 - 8*r**2)*(R0**3*sqrt(r/R0)*(tmp6)**(2.5)*(-2*r**2 + 3*y**2) + r**2*y**2*(R0 + r) + 2*r*(R0 + r)**2*(-r**2 + y**2))) + 4*R0*tmp5*(R0 + r)**3*tmp3**2*(225*R0**2*y**2 - 10*R0*r**3 + 40*R0*r*y**2 + 26*r**4 - 3*r**2*(15*R0**2 + 26*y**2)) + 12*R0*(R0 + r)**3*tmp3**2*(-75*R0**2*y**2 - 8*r**4 + 3*r**2*(5*R0**2 + 8*y**2))*tmp2 + 3*y**2*(R0 + r)**2*(5*R0**2 - 8*r**2)*(tmp4)**2 - 2*tmp5*(R0 + r)*tmp3**2*tmp7*(r**3*(R0 + r) - 2*r**2*y**2 - r**2*(R0 + r)**2 + r*y**2*(R0 + r) + y**2*(R0 + r)**2))) + 48*a**2*r**4*(R0 + r)**4*tmp3*(R0**3*sqrt(r/R0)*(tmp6)**(2.5)*(-2*r**2 + 3*y**2) + r**2*y**2*(R0 + r) + 2*r*(R0 + r)**2*(-r**2 + y**2)) + 24*y**2*(R0 + r)**5*tmp3*(-R0*(-r**2*(a**2 - b**2) + y**2*(a**2 - b**2) + z**2*(a**2 - c**2))*(-90*R0**2*tmp*(R0 + r)*tmp3*tmp2 - R0*tmp*tmp5*tmp3*(15*R0**2 + 5*R0*r - 2*r**2) + 15*R0*(R0 + r)*(tmp4) + 2*tmp*tmp5*(R0 + r)*tmp3*(45*R0**2 + 10*R0*r - 2*r**2)) + 8*a**2*r**4*(R0 + r)*(tmp4)))/(96*R0**3*a**2*r**9*(tmp6)**(3.5)*(R0 + r)**3*tmp3**2)
    d2Phi_dz2 = C*(R0**3*(tmp6)**(3.5)*(48*R0**2*a**2*r**3*z**2*sqrt((R0 + r)/r)*(R0 + r)*tmp3**2*(x**2 + y**2 + z**2) + 96*R0*a**2*r**6*z**2*sqrt((R0 + r)/r)*(R0 + r)*tmp3**2 + 192*R0*a**2*r**6*(R0 + r)**3*tmp3**2*tmp2 - 96*R0*a**2*r**5*z**2*sqrt((R0 + r)/r)*(R0 + r)**2*tmp3**2 - 576*R0*a**2*r**4*z**2*(R0 + r)**3*tmp3**2*tmp2 + 24*R0*(R0 + r)**3*tmp3**2*(15*R0**2*tmp2 + tmp5*(-15*R0**2 - 5*R0*r + 2*r**2))*(r**4*(a**2 - c**2) + r**2*(y**2*(-a**2 + b**2) + 5*z**2*(-a**2 + c**2)) + 4*z**2*tmp8) - 3*R0*tmp8*(180*R0**2*r**2*(R0 + r)**3*tmp3**2*tmp2 + R0**2*z**2*tmp5*(R0 + r)*tmp3**2*(15*R0**2 + 5*R0*r - 2*r**2) - 900*R0**2*z**2*(R0 + r)**3*tmp3**2*tmp2 + 15*R0**2*tmp*tmp3*(R0**3*sqrt(r/R0)*(tmp6)**(2.5)*(-2*r**2 + 3*z**2) + r**2*z**2*(R0 + r) + 2*r*(R0 + r)**2*(-r**2 + z**2) + 12*z**2*(R0 + r)**2*(tmp4)) + 2*R0*r*z**2*tmp5*(R0 + r)*tmp3**2*(15*R0**2 + 5*R0*r - 2*r**2) - 2*R0*z**2*tmp5*(R0 + r)**2*tmp3**2*(15*R0**2 + 5*R0*r - 2*r**2) - 4*R0*z**2*tmp5*(R0 + r)**2*tmp3**2*(45*R0**2 + 10*R0*r - 2*r**2) + 15*R0*z**2*(R0 + r)**2*(tmp4)**2 + 4*tmp5*(R0 + r)**3*tmp3**2*(225*R0**2*z**2 - 10*R0*r**3 + 40*R0*r*z**2 + 2*r**4 - 3*r**2*(15*R0**2 + 2*z**2)) - 2*tmp5*(R0 + r)*tmp3**2*(15*R0**2 + 5*R0*r - 2*r**2)*(r**3*(R0 + r) - 2*r**2*z**2 - r**2*(R0 + r)**2 + r*z**2*(R0 + r) + z**2*(R0 + r)**2)) + 96*a**2*r**5*sqrt((R0 + r)/r)*(R0 + r)**2*tmp3**2*(r**3 - r**2*(R0 + r) - 3*r*z**2 + 3*z**2*(R0 + r)) + 48*a**2*r**4*z**2*(R0 + r)**2*(tmp4)**2 - r**2*(-2*a**2 + b**2 + c**2)*(-4*R0**2*z**2*tmp5*(R0 + r)**2*tmp3**2*(45*R0**2 + 10*R0*r - 26*r**2) + R0**2*z**2*tmp5*(R0 + r)*tmp3**2*tmp7 + 2*R0*r*z**2*tmp5*(R0 + r)*tmp3**2*tmp7 - 2*R0*z**2*tmp5*(R0 + r)**2*tmp3**2*tmp7 + 3*R0*tmp*tmp3*(4*z**2*(R0 + r)**2*(15*R0**2 - 8*r**2)*(tmp4) + (5*R0**2 - 8*r**2)*(R0**3*sqrt(r/R0)*(tmp6)**(2.5)*(-2*r**2 + 3*z**2) + r**2*z**2*(R0 + r) + 2*r*(R0 + r)**2*(-r**2 + z**2))) + 4*R0*tmp5*(R0 + r)**3*tmp3**2*(225*R0**2*z**2 - 10*R0*r**3 + 40*R0*r*z**2 + 26*r**4 - 3*r**2*(15*R0**2 + 26*z**2)) + 12*R0*(R0 + r)**3*tmp3**2*(-75*R0**2*z**2 - 8*r**4 + 3*r**2*(5*R0**2 + 8*z**2))*tmp2 + 3*z**2*(R0 + r)**2*(5*R0**2 - 8*r**2)*(tmp4)**2 - 2*tmp5*(R0 + r)*tmp3**2*tmp7*(r**3*(R0 + r) - 2*r**2*z**2 - r**2*(R0 + r)**2 + r*z**2*(R0 + r) + z**2*(R0 + r)**2))) + 48*a**2*r**4*(R0 + r)**4*tmp3*(R0**3*sqrt(r/R0)*(tmp6)**(2.5)*(-2*r**2 + 3*z**2) + r**2*z**2*(R0 + r) + 2*r*(R0 + r)**2*(-r**2 + z**2)) + 24*z**2*(R0 + r)**5*tmp3*(-R0*(-r**2*(a**2 - c**2) + y**2*(a**2 - b**2) + z**2*(a**2 - c**2))*(-90*R0**2*tmp*(R0 + r)*tmp3*tmp2 - R0*tmp*tmp5*tmp3*(15*R0**2 + 5*R0*r - 2*r**2) + 15*R0*(R0 + r)*(tmp4) + 2*tmp*tmp5*(R0 + r)*tmp3*(45*R0**2 + 10*R0*r - 2*r**2)) + 8*a**2*r**4*(R0 + r)*(tmp4)))/(96*R0**3*a**2*r**9*(tmp6)**(3.5)*(R0 + r)**3*tmp3**2)
    d2Phi_dxdy = C*x*y*(48*R0**2*(tmp6)**(2.5)*(R0**2*a**2*r**3*sqrt((R0 + r)/r)*tmp3**2*(x**2 + y**2 + z**2) + 2*R0*a**2*r**6*sqrt((R0 + r)/r)*tmp3**2 + 4*R0*a**2*r**5*sqrt((R0 + r)/r)*(R0 + r)*tmp3**2 - 12*R0*a**2*r**4*(R0 + r)**2*tmp3**2*tmp2 + R0*a**2*r**4*tmp3*(r*tmp*(2*R0 + 3*r) + 3*sqrt(r/R0)*(R0 + r)**2) + R0*(R0 + r)**2*tmp3**2*(15*R0**2*tmp2 + tmp5*(-15*R0**2 - 5*R0*r + 2*r**2))*(-r**2*(a**2 - b**2) + 2*y**2*(a**2 - b**2) + 2*z**2*(a**2 - c**2)) + a**2*r**4*(R0 + r)*(tmp4)**2) - 12*R0*(R0 + r)**3*tmp3*tmp8*(-90*R0**2*tmp*(R0 + r)*tmp3*tmp2 - R0*tmp*tmp5*tmp3*(15*R0**2 + 5*R0*r - 2*r**2) + 15*R0*(R0 + r)*(tmp4) + 2*tmp*tmp5*(R0 + r)*tmp3*(45*R0**2 + 10*R0*r - 2*r**2)) - 12*R0*(R0 + r)**3*tmp3*(-r**2*(a**2 - b**2) + y**2*(a**2 - b**2) + z**2*(a**2 - c**2))*(-90*R0**2*tmp*(R0 + r)*tmp3*tmp2 - R0*tmp*tmp5*tmp3*(15*R0**2 + 5*R0*r - 2*r**2) + 15*R0*(R0 + r)*(tmp4) + 2*tmp*tmp5*(R0 + r)*tmp3*(45*R0**2 + 10*R0*r - 2*r**2)) - 3*R0*(R0 + r)**2*tmp8*(R0**2*tmp*tmp5*tmp3**2*(15*R0**2 + 5*R0*r - 2*r**2) - 900*R0**2*tmp*(R0 + r)**2*tmp3**2*tmp2 + 15*R0**2*tmp*tmp3*(r*tmp*(2*R0 + 3*r) + 3*sqrt(r/R0)*(R0 + r)**2) + 2*R0*r*tmp*tmp5*tmp3**2*(15*R0**2 + 5*R0*r - 2*r**2) - 2*R0*tmp*tmp5*(R0 + r)*tmp3**2*(15*R0**2 + 5*R0*r - 2*r**2) - 4*R0*tmp*tmp5*(R0 + r)*tmp3**2*(45*R0**2 + 10*R0*r - 2*r**2) + 15*R0*tmp*(R0 + r)*(tmp4)**2 + 180*R0*(R0 + r)**2*tmp3*(tmp4) + 4*tmp*tmp5*(R0 + r)**2*tmp3**2*(225*R0**2 + 40*R0*r - 6*r**2) - 2*tmp*tmp5*tmp3**2*(15*R0**2 + 5*R0*r - 2*r**2)*(-2*r**2 + r*(R0 + r) + (R0 + r)**2)) + 192*a**2*r**4*(R0 + r)**4*tmp3*(tmp4) - r**2*(R0 + r)**2*(-2*a**2 + b**2 + c**2)*(-4*R0**2*tmp*tmp5*(R0 + r)*tmp3**2*(45*R0**2 + 10*R0*r - 26*r**2) + R0**2*tmp*tmp5*tmp3**2*tmp7 + 2*R0*r*tmp*tmp5*tmp3**2*tmp7 + 4*R0*tmp*tmp5*(R0 + r)**2*tmp3**2*(225*R0**2 + 40*R0*r - 78*r**2) - 2*R0*tmp*tmp5*(R0 + r)*tmp3**2*tmp7 - 36*R0*tmp*(R0 + r)**2*(25*R0**2 - 8*r**2)*tmp3**2*tmp2 + 3*R0*tmp*(5*R0**2 - 8*r**2)*tmp3*(r*tmp*(2*R0 + 3*r) + 3*sqrt(r/R0)*(R0 + r)**2) - 2*tmp*tmp5*tmp3**2*(-2*r**2 + r*(R0 + r) + (R0 + r)**2)*tmp7 + 3*tmp*(R0 + r)*(5*R0**2 - 8*r**2)*(tmp4)**2 + 12*(R0 + r)**2*(15*R0**2 - 8*r**2)*tmp3*(tmp4)))/(96*R0**2*a**2*r**9*(tmp6)**(2.5)*(R0 + r)**2*tmp3**2)
    d2Phi_dxdz = C*x*z*(48*R0**2*(tmp6)**(2.5)*(R0**2*a**2*r**3*sqrt((R0 + r)/r)*tmp3**2*(x**2 + y**2 + z**2) + 2*R0*a**2*r**6*sqrt((R0 + r)/r)*tmp3**2 + 4*R0*a**2*r**5*sqrt((R0 + r)/r)*(R0 + r)*tmp3**2 - 12*R0*a**2*r**4*(R0 + r)**2*tmp3**2*tmp2 + R0*a**2*r**4*tmp3*(r*tmp*(2*R0 + 3*r) + 3*sqrt(r/R0)*(R0 + r)**2) + R0*(R0 + r)**2*tmp3**2*(15*R0**2*tmp2 + tmp5*(-15*R0**2 - 5*R0*r + 2*r**2))*(-r**2*(a**2 - c**2) + 2*y**2*(a**2 - b**2) + 2*z**2*(a**2 - c**2)) + a**2*r**4*(R0 + r)*(tmp4)**2) - 12*R0*(R0 + r)**3*tmp3*tmp8*(-90*R0**2*tmp*(R0 + r)*tmp3*tmp2 - R0*tmp*tmp5*tmp3*(15*R0**2 + 5*R0*r - 2*r**2) + 15*R0*(R0 + r)*(tmp4) + 2*tmp*tmp5*(R0 + r)*tmp3*(45*R0**2 + 10*R0*r - 2*r**2)) - 12*R0*(R0 + r)**3*tmp3*(-r**2*(a**2 - c**2) + y**2*(a**2 - b**2) + z**2*(a**2 - c**2))*(-90*R0**2*tmp*(R0 + r)*tmp3*tmp2 - R0*tmp*tmp5*tmp3*(15*R0**2 + 5*R0*r - 2*r**2) + 15*R0*(R0 + r)*(tmp4) + 2*tmp*tmp5*(R0 + r)*tmp3*(45*R0**2 + 10*R0*r - 2*r**2)) - 3*R0*(R0 + r)**2*tmp8*(R0**2*tmp*tmp5*tmp3**2*(15*R0**2 + 5*R0*r - 2*r**2) - 900*R0**2*tmp*(R0 + r)**2*tmp3**2*tmp2 + 15*R0**2*tmp*tmp3*(r*tmp*(2*R0 + 3*r) + 3*sqrt(r/R0)*(R0 + r)**2) + 2*R0*r*tmp*tmp5*tmp3**2*(15*R0**2 + 5*R0*r - 2*r**2) - 2*R0*tmp*tmp5*(R0 + r)*tmp3**2*(15*R0**2 + 5*R0*r - 2*r**2) - 4*R0*tmp*tmp5*(R0 + r)*tmp3**2*(45*R0**2 + 10*R0*r - 2*r**2) + 15*R0*tmp*(R0 + r)*(tmp4)**2 + 180*R0*(R0 + r)**2*tmp3*(tmp4) + 4*tmp*tmp5*(R0 + r)**2*tmp3**2*(225*R0**2 + 40*R0*r - 6*r**2) - 2*tmp*tmp5*tmp3**2*(15*R0**2 + 5*R0*r - 2*r**2)*(-2*r**2 + r*(R0 + r) + (R0 + r)**2)) + 192*a**2*r**4*(R0 + r)**4*tmp3*(tmp4) - r**2*(R0 + r)**2*(-2*a**2 + b**2 + c**2)*(-4*R0**2*tmp*tmp5*(R0 + r)*tmp3**2*(45*R0**2 + 10*R0*r - 26*r**2) + R0**2*tmp*tmp5*tmp3**2*tmp7 + 2*R0*r*tmp*tmp5*tmp3**2*tmp7 + 4*R0*tmp*tmp5*(R0 + r)**2*tmp3**2*(225*R0**2 + 40*R0*r - 78*r**2) - 2*R0*tmp*tmp5*(R0 + r)*tmp3**2*tmp7 - 36*R0*tmp*(R0 + r)**2*(25*R0**2 - 8*r**2)*tmp3**2*tmp2 + 3*R0*tmp*(5*R0**2 - 8*r**2)*tmp3*(r*tmp*(2*R0 + 3*r) + 3*sqrt(r/R0)*(R0 + r)**2) - 2*tmp*tmp5*tmp3**2*(-2*r**2 + r*(R0 + r) + (R0 + r)**2)*tmp7 + 3*tmp*(R0 + r)*(5*R0**2 - 8*r**2)*(tmp4)**2 + 12*(R0 + r)**2*(15*R0**2 - 8*r**2)*tmp3*(tmp4)))/(96*R0**2*a**2*r**9*(tmp6)**(2.5)*(R0 + r)**2*tmp3**2)
    d2Phi_dydz = C*y*z*(48*R0**2*a**2*r**3*sqrt((R0 + r)/r)*(R0 + r)**2*tmp3**2*(x**2 + y**2 + z**2) + R0**2*(tmp6)**(1.5)*(12*R0*(R0 + r)*tmp3*(r**2*(a**2 - b**2) + r**2*(a**2 - c**2) - 2*y**2*(a**2 - b**2) - 2*z**2*(a**2 - c**2))*(-90*R0**2*tmp*(R0 + r)*tmp3*tmp2 - R0*tmp*tmp5*tmp3*(15*R0**2 + 5*R0*r - 2*r**2) + 15*R0*(R0 + r)*(tmp4) + 2*tmp*tmp5*(R0 + r)*tmp3*(45*R0**2 + 10*R0*r - 2*r**2)) - 3*R0*tmp8*(R0**2*tmp*tmp5*tmp3**2*(15*R0**2 + 5*R0*r - 2*r**2) - 900*R0**2*tmp*(R0 + r)**2*tmp3**2*tmp2 + 15*R0**2*tmp*tmp3*(r*tmp*(2*R0 + 3*r) + 3*sqrt(r/R0)*(R0 + r)**2) + 2*R0*r*tmp*tmp5*tmp3**2*(15*R0**2 + 5*R0*r - 2*r**2) - 2*R0*tmp*tmp5*(R0 + r)*tmp3**2*(15*R0**2 + 5*R0*r - 2*r**2) - 4*R0*tmp*tmp5*(R0 + r)*tmp3**2*(45*R0**2 + 10*R0*r - 2*r**2) + 15*R0*tmp*(R0 + r)*(tmp4)**2 + 180*R0*(R0 + r)**2*tmp3*(tmp4) + 4*tmp*tmp5*(R0 + r)**2*tmp3**2*(225*R0**2 + 40*R0*r - 6*r**2) - 2*tmp*tmp5*tmp3**2*(15*R0**2 + 5*R0*r - 2*r**2)*(-2*r**2 + r*(R0 + r) + (R0 + r)**2)) + 192*a**2*r**4*(R0 + r)**2*tmp3*(tmp4) - r**2*(-2*a**2 + b**2 + c**2)*(-4*R0**2*tmp*tmp5*(R0 + r)*tmp3**2*(45*R0**2 + 10*R0*r - 26*r**2) + R0**2*tmp*tmp5*tmp3**2*tmp7 + 2*R0*r*tmp*tmp5*tmp3**2*tmp7 + 4*R0*tmp*tmp5*(R0 + r)**2*tmp3**2*(225*R0**2 + 40*R0*r - 78*r**2) - 2*R0*tmp*tmp5*(R0 + r)*tmp3**2*tmp7 - 36*R0*tmp*(R0 + r)**2*(25*R0**2 - 8*r**2)*tmp3**2*tmp2 + 3*R0*tmp*(5*R0**2 - 8*r**2)*tmp3*(r*tmp*(2*R0 + 3*r) + 3*sqrt(r/R0)*(R0 + r)**2) - 2*tmp*tmp5*tmp3**2*(-2*r**2 + r*(R0 + r) + (R0 + r)**2)*tmp7 + 3*tmp*(R0 + r)*(5*R0**2 - 8*r**2)*(tmp4)**2 + 12*(R0 + r)**2*(15*R0**2 - 8*r**2)*tmp3*(tmp4))) + 96*R0*a**2*r**6*sqrt((R0 + r)/r)*(R0 + r)**2*tmp3**2 + 192*R0*a**2*r**5*sqrt((R0 + r)/r)*(R0 + r)**3*tmp3**2 - 576*R0*a**2*r**4*(R0 + r)**4*tmp3**2*tmp2 + 48*R0*a**2*r**4*(R0 + r)**2*tmp3*(r*tmp*(2*R0 + 3*r) + 3*sqrt(r/R0)*(R0 + r)**2) - 48*R0*(R0 + r)**4*tmp3**2*(15*R0**2*tmp2 + tmp5*(-15*R0**2 - 5*R0*r + 2*r**2))*(r**2*(2*a**2 - b**2 - c**2) - 2*y**2*(a**2 - b**2) - 2*z**2*(a**2 - c**2)) + 48*a**2*r**4*(R0 + r)**3*(tmp4)**2)/(96*a**2*r**9*(R0 + r)**4*tmp3**2)

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