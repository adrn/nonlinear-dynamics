# coding: utf-8

"""
Experiment 1
============

In a simple, axisymmetric NFW potential -- no disk and bulge -- what
kind of orbits do you get? What does stream formation look like as a
function of Lyapunov time?

"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
import shutil

# Third-party
import cubehelix
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root, minimize
import astropy.units as u
from astropy.constants import G
from astropy import log as logger
logger.setLevel(logging.INFO)

# Project
from streamteam.util import get_pool
from streamteam.integrate import DOPRI853Integrator
from streamteam.dynamics import lyapunov_spectrum
from nonlineardynamics.apw_helpers import *
from nonlineardynamics import LyapunovMap

def R_Rdot_grid(E, potential_params, nR=10, nRdot=10):
    z = 0.

    # find the min and max allowed R values
    f = lambda R: E - effective_potential(np.array([R,0.,0.,0.]),**potential_params)
    res1 = root(f, x0=1.)
    res2 = root(f, x0=100.)

    if not (res1.success and res2.success):
        raise ValueError("R Root finding failed.")

    minR = res1.x[0] + res1.x[0]/100.
    maxR = res2.x[0] - res2.x[0]/100.
    R_grid = np.linspace(minR, maxR, nR)
    logger.info("Min, max R for grid: {},{}".format(minR, maxR))

    # now find the max allowed Rdot
    zvc = lambda R: -np.sqrt(2*(E - effective_potential(np.array([R,0.,0.,0.]), **potential_params)))
    res = minimize(zvc, x0=20., options={'xtol': 1e-8},  method='nelder-mead')
    if not res.success:
        raise ValueError("Rdot optimization failed.")

    maxRdot = -res.fun + res.fun/100.
    Rdot_grid = np.linspace(0., maxRdot, nRdot)

    # create a grid of initial conditions
    R,Rdot = map(np.ravel, np.meshgrid(R_grid, Rdot_grid))

    # now throw out any point where Rdot > ZVC
    ix = Rdot < -zvc(R)

    return R[ix], Rdot[ix]

def w0_from_grid(E, potential_params, R, Rdot):
    z = np.zeros_like(R)
    V = effective_potential(np.vstack((R,z)).T, **potential_params)
    zdot = np.sqrt(2*(E-V) - Rdot**2)
    return np.vstack((R,z,Rdot,zdot)).T

def plot_rotation_curve(potential_params, plot_path=""):
    # rotation curve stuff
    logger.info((rotation_curve(8., **potential_params)*u.kpc/u.Myr).to(u.km/u.s).value)

    R = np.linspace(0.01, 50, 400)
    V = rotation_curve(R, **potential_params)
    plt.plot(R, (V*u.kpc/u.Myr).to(u.km/u.s).value, marker=None, lw=2.)
    plt.xlabel("R [kpc]")
    plt.ylabel("V [km/s]")
    plt.savefig(os.path.join(plot_path, "rotation_curve.png"))

def main(pool, ngrid, nsteps=5000, dt=10., overwrite=False):
    # set the name here
    name = "nfw"

    project_path = os.path.split(os.environ['STREAMSPATH'])[0]
    output_path = os.path.join(project_path, "nonlinear-dynamics", "output")
    path = os.path.join(output_path, name)
    if os.path.exists(path) and overwrite:
        shutil.rmtree(path)

    plot_path = os.path.join(path, "plots")
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Most parameters taken from:
    #   http://www.aanda.org/articles/aa/pdf/2013/01/aa20540-12.pdf
    # Rs estimated from Rvir=267 / c=12. (Xue (Beers) et al. paper)
    potential_params = dict()

    # disk
    potential_params['Md'] = 7.E10 # Msun
    potential_params['a'] = 3.3
    potential_params['b'] = 0.29

    # nucleus
    potential_params['Mn'] = 10E9 # Msun
    potential_params['c'] = 0.24

    # halo
    potential_params['Mh'] = 2.5E12 # Msun
    potential_params['Rs'] = 40.
    potential_params['q'] = 0.8

    # plot the rotation curve for this potential
    plot_rotation_curve(potential_params, plot_path=plot_path)

    # energy, angular momentum
    E = (-2000*100*(u.km/u.s)**2).decompose(usys).value
    Lz = (150.*10.*u.km*u.kpc/u.s).decompose(usys).value
    potential_params['Lz'] = Lz

    # generate a grid of R, Rdot
    R,Rdot = R_Rdot_grid(E, potential_params, nR=ngrid, nRdot=ngrid)
    w0s = w0_from_grid(E, potential_params, R, Rdot)
    # w0s = np.array([[19.61443808,0.,0.,0.25824497]]) # HACK
    gridsize = len(w0s)

    # grid of IC's
    plt.clf()
    plt.plot(R,Rdot,linestyle='none')
    plt.savefig(os.path.join(plot_path, "ic_grid.png"))

    kwargs = dict(nsteps=nsteps, dt=dt)
    lm = LyapunovMap(name, F_var, lyapunov_kwargs=kwargs,
                     overwrite=overwrite, prefix=output_path)
    lm.potential_pars = potential_params.items()

    # get a pool to use map()
    pool.map(lm, list(zip(np.arange(gridsize),w0s)))
    pool.close()

    fig = plt.figure(figsize=(10,10))
    chaotic = np.zeros(gridsize).astype(bool)
    dm = np.zeros(gridsize)
    for r in lm.iterate_cache():
        lyap, t, w, pp, fname = r
        w0 = w[0]

        # hack to get ii
        ii = w0s.tolist().index(list(w0))

        # estimate lyapunov time from final step
        max_idx = lyap.sum(axis=0).argmax()
        t_lyap = (1./lyap[:,max_idx]*u.Myr).to(u.Gyr)
        # print(t_lyap[-100:].mean())
        # print(t_lyap.min())
        # print()

        # pericenter and apocenter
        r = np.sqrt(w[...,0]**2 + w[...,1]**2)
        logger.debug("Peri: {}, Apo: {}".format(r.min(), r.max()))

        # orbit
        plt.clf()
        plt.title(r"$R$={}, $\dot{{R}}$={}".format(w0[0], w0[2]))
        plt.text(2., -35, "End lyap.: {}".format(np.median(t_lyap[-100])))
        plt.text(2., -40, "Peri: {:.2f}, Apo: {:.2f}".format(r.min(), r.max()))

        plt.plot(w[...,0], w[...,1], marker=None)
        plt.xlim(0., 85.)
        plt.ylim(-45., 45.)
        plt.savefig(os.path.join(plot_path, "{}.png".format(ii)))

        # lyapunov exponents
        plt.clf()
        plt.loglog(t, lyap, marker=None)
        plt.xlim(t[1], t[-1])
        plt.ylim(1E-5, 1.)
        plt.savefig(os.path.join(plot_path, "lyap_{}.png".format(ii)))

    return

    plt.plot(ws[...,0], ws[...,1], marker=None)
    plt.show()

    # plt.plot(R_zvc, Rdot_zvc, marker=None)
    # plt.plot(ws[...,0], ws[...,2], marker=None)
    # plt.show()


if __name__ == "__main__":
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite",
                        default=False, help="Overwrite cached files.")

    parser.add_argument("--ngrid", dest="ngrid", required=True,
                        help="Number of grid points along R and Rdot.")

    # threading
    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Run with MPI.")

    # pass to lyapunov
    parser.add_argument("--nsteps", dest="nsteps", default=100000, type=int,
                        help="Number of steps.")
    parser.add_argument("--dt", dest="dt", default=1., type=float,
                        help="Timestep.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)

    pool = get_pool(mpi=args.mpi)
    main(pool=pool, overwrite=args.overwrite, ngrid=args.ngrid,
         nsteps=args.nsteps, dt=args.dt)
