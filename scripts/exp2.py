# coding: utf-8

"""
Experiment 2
============

In a triaxial NFW potential -- no disk and bulge -- what does the Sgr orbit
look like at different orientations relative to the halo potential.

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
import scipy.optimize as so
import astropy.units as u
from astropy.constants import G
from astropy import log as logger
logger.setLevel(logging.INFO)
from astropy.coordinates.angles import rotation_matrix

# Project
from streamteam import integrate
from streamteam.util import get_pool
from streamteam.dynamics import lyapunov_spectrum
from nonlineardynamics import nfw
from nonlineardynamics import LyapunovMap

# phase-space position of Sgr today in the MW
sgr_w = np.array([19.0,2.7,-6.9,0.2352238,-0.03579493,0.19942887])

def main(pool, overwrite=False, nsteps=None, dt=None, ngrid=None):
    # set the name here
    name = "exp2"

    # ----------------------------------------------------------------------
    # Don't remove or change this stuff
    project_path = os.path.split(os.environ['STREAMSPATH'])[0]
    output_path = os.path.join(project_path, "nonlinear-dynamics", "output")
    path = os.path.join(output_path, name)
    if os.path.exists(path) and overwrite:
        shutil.rmtree(path)

    plot_path = os.path.join(path, "plots")
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    # ----------------------------------------------------------------------

    # print(nfw.potential(0., sgr_w, **nfw.vl2_params))
    # print(nfw.acceleration(0., sgr_w, **nfw.vl2_params))

    # similar to Sgr
    r = np.array([20.,0,0])
    v = np.array([0.075,0.,0.125])

    # # altitude and azimuth
    # alt = 0.*u.deg
    # azi = 45.*u.deg

    # generate initial conditions
    w0s = []
    azis = np.linspace(0,180,ngrid)*u.deg
    for alt,azi in zip(np.zeros(N)*u.deg,azis):
        R_alt = rotation_matrix(-alt, "y")
        R_azi = rotation_matrix(azi, "z")
        r_rot = np.array(r.dot(R_alt).dot(R_azi))
        v_rot = np.array(v.dot(R_alt.T).dot(R_azi.T))

        w0 = np.hstack((r_rot, v_rot)).squeeze()
        w0s.append(w0.tolist())
    w0s = np.array(w0s)

    kwargs = dict(nsteps=nsteps, dt=dt)
    lm = LyapunovMap(name, nfw.F, lyapunov_kwargs=kwargs,
                     overwrite=overwrite, prefix=output_path,
                     Integrator=integrate.RK5Integrator)
    lm.potential_pars = nfw.vl2_params.items()

    # get a pool to use map()
    pool.map(lm, list(zip(np.arange(ngrid),w0s)))
    pool.close()

    fig = plt.figure(figsize=(10,10))
    #chaotic = np.zeros(ngrid).astype(bool)
    #end_lyaps = np.zeros(ngrid)
    for r in lm.iterate_cache():
        lyap, t, w, pp, fname = r
        w0 = w[0]

        # hack to get ii
        ii = w0s.tolist().index(list(w0))

        # estimate lyapunov time from final step
        max_idx = lyap.sum(axis=0).argmax()
        t_lyap = (1./lyap[:,max_idx]*u.Myr).to(u.Gyr)
        logger.debug("t_lyap = {}".format(t_lyap))
        #end_lyaps[ii] = np.median(t_lyap[-100:])
        # print(t_lyap[-100:].mean())
        # print(t_lyap.min())
        # print()

        # pericenter and apocenter
        r = np.sqrt(np.sum(w[...,:3]**2,axis=-1))
        logger.debug("Peri: {}, Apo: {}".format(r.min(), r.max()))

        # orbit
        # plt.clf()
        # plt.text(2., -35, "End lyap.: {}".format(np.median(t_lyap[-100])))
        # plt.text(2., -40, "Peri: {:.2f}, Apo: {:.2f}".format(r.min(), r.max()))

        fig,axes = plt.subplots(2,2,figsize=(10,10),sharex=True,sharey=True)
        axes[0,0].plot(w[:,0], w[:,1], marker=None)
        axes[0,1].plot(w[:,1], w[:,1], marker=None)
        axes[0,1].set_visible(False)
        axes[1,0].plot(w[:,0], w[:,2], marker=None)
        axes[1,1].plot(w[:,1], w[:,2], marker=None)
        axes[1,1].set_xlim(-50,50)
        axes[1,1].set_ylim(-50,50)
        fig.suptitle(r"$\phi$={:.2f}".format(azis[ii].to(u.degree).value))
        fig.savefig(os.path.join(plot_path,'orbit_{}.png'.format(ii)))
        plt.close('all')

        # lyapunov exponents
        plt.clf()
        plt.loglog(t, lyap, marker=None)
        plt.xlim(t[1], t[-1])
        plt.ylim(1E-5, 1.)
        plt.savefig(os.path.join(plot_path, "lyap_{}.png".format(ii)))


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
    main(pool=pool, overwrite=args.overwrite, nsteps=args.nsteps,
         dt=args.dt, ngrid=args.ngrid)
