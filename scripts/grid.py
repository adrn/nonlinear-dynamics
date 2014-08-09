# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import gc
import random
random.seed(42)
import time

# Third-party
from astropy import log as logger
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

# Project

from streamteam.potential.lm10 import LM10Potential
from streamteam.potential.apw import PW14Potential
import streamteam.integrate as si
import streamteam.dynamics as sd
from streamteam.util import get_pool

usys = (u.kpc, u.Myr, u.radian, u.Msun)
# plot_path = "output/planes"
# plot_path = "/hotfoot/astrostats/astro/users/amp2217/planes"
plot_path = "/vega/astro/users/amp2217/projects/nonlinear-dynamics/output/planes"

def filter_grid(E, r, r_dot, phi, phi_dot, theta, potential):

    x = r*np.cos(phi)*np.sin(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(theta)

    xyz = np.array([x,y,z]).T.copy()
    theta_dot = np.sqrt(2*(E - potential.value(xyz))/r**2 - (r_dot/r)**2)
    r_dot_max = np.sqrt(2*(E - potential.value(xyz)))
    r_dot_min = -np.sqrt(2*(E - potential.value(xyz)))

    ix = (r_dot > r_dot_min) & (r_dot < r_dot_max) & (~np.isnan(theta_dot))

    return r[ix], r_dot[ix], theta_dot[ix]

def grid_to_ics(r, r_dot, phi, phi_dot, theta, theta_dot):
    # convert back to cartesian
    ics = np.zeros((len(r),6))*np.nan
    ics[:,0] = r*np.cos(phi)*np.sin(theta)
    ics[:,1] = r*np.sin(phi)*np.sin(theta)
    ics[:,2] = r*np.cos(theta)

    # xdot, ydot, zdot
    sint,cost = np.sin(theta),np.cos(theta)
    sinp,cosp = np.sin(phi),np.cos(phi)
    ics[:,3] = sint*cosp*r_dot + r*cost*cosp*theta_dot - r*sint*sinp*phi_dot
    ics[:,4] = sint*sinp*r_dot + r*cost*sinp*theta_dot + r*sint*cosp*phi_dot
    ics[:,5] = cost*r_dot - r*sint*theta_dot

    return ics

def bork(angles):
    phi,theta = angles

    fn = os.path.join(plot_path, "phi{}_theta{}.npy".format(phi,theta))

    if not os.path.exists(fn):

        # take energy from Sgr orbit
        sgr_w = np.array([19.0,2.7,-6.9,0.2352238,-0.03579493,0.19942887])
        sgr_pot = LM10Potential()
        E = sgr_pot.value(sgr_w[:3]) + 0.5*np.sum(sgr_w[3:]**2)
        E = -0.11150041

        # arbitrarily set phi_dot = 0
        phi_dot = 0.

        # make a grid in r, r_dot
        _r = np.arange(10., 200., 0.5)
        _r_dot = (np.arange(-600., 600., 15.)*u.km/u.s).decompose(usys).value
        r,r_dot = np.meshgrid(_r,_r_dot)

        # potential = LM10Potential(q1=1.4, q2=1., q3=0.8, phi=0.)
        potential = PW14Potential()
        r,r_dot,theta_dot = filter_grid(E, r.ravel(), r_dot.ravel(), phi, phi_dot, theta, potential)

        # plot grid of ICs
        # fig,ax = plt.subplots(1,1,figsize=(10,10))
        # cax = ax.scatter(r, r_dot, marker='o', c=np.log(theta_dot))
        # fig.colorbar(cax)
        # fig.savefig(os.path.join(plot_path, "ic_grid_phi{}_theta{}.png".format(phi,theta)))

        # turn the grid into an array of initial conditions
        w0 = grid_to_ics(r, r_dot, phi, phi_dot, theta, theta_dot)
        logger.debug("Shape of ICs: {}".format(w0.shape))

        # integrate all the ICs
        integrator = si.LeapfrogIntegrator(lambda t, *args: potential.acceleration(*args))

        # define initial conditions for Sgr orbit (x,y,z,vx,vy,vz)
        a = time.time()
        t,ws = integrator.run(w0, dt=1., nsteps=15000)
        logger.debug("Took {} seconds to integrate.".format(time.time() - a))

        np.save(fn, ws)
    else:
        ws = np.load(fn)

    orb = sd.classify_orbit(ws)
    is_loop = np.any(orb, axis=1).astype(bool)
    is_box = np.logical_not(is_loop)
    box_frac = is_box.sum() / float(len(is_loop))
    logger.info("Fraction of box orbits: {}".format(box_frac))

    del ws, orb
    gc.collect()

    return phi,theta,box_frac

    # plot grid of ICs, classified
    # plt.figure(figsize=(12,12))
    # plt.scatter(r[is_box], r_dot[is_box], marker='s', c='#d7191c')
    # plt.scatter(r[is_loop], r_dot[is_loop], marker='o',
    #             facecolors='none', edgecolors='#1a9641')
    # plt.xlim(r.min(),r.max())
    # plt.ylim(r_dot.min(),r_dot.max())
    # plt.savefig(os.path.join(plot_path, "phi{}_theta{}.png".format(phi,theta)))

def main(mpi=False):
    pool = get_pool(mpi=mpi)

    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    theta = np.arccos(1. - np.linspace(0.01,1.,15))
    phi = np.linspace(0.01, 0.99*np.pi/2., 15)

    t,p = np.meshgrid(theta, phi)
    theta = t.ravel()
    phi = p.ravel()

    angles = []
    for t,p in zip(theta,phi):
        #fn = os.path.join(plot_path, "phi{}_theta{}.npy".format(p,t))
        #if os.path.exists(fn):
        #    logger.debug("'{}' exists...skipping".format(fn))
        #    continue
        angles.append((p,t))

    box_fracs = pool.map(bork, angles)
    pool.close()

    fn = os.path.join(plot_path, "box_fracs.npy")
    np.save(fn, box_fracs)

    sys.exit(0)

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("--mpi", dest="mpi", action="store_true", default=False,
                        help="Run with MPI.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(mpi=args.mpi)
