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
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.constants import G
from astropy import log as logger
logger.setLevel(logging.INFO)
from astropy.coordinates.angles import rotation_matrix

# Project
from streamteam import integrate
from streamteam.util import get_pool
from streamteam.dynamics import lyapunov_spectrum
from nonlineardynamics import nfw_density as nfw
from nonlineardynamics import LyapunovMap

# phase-space position of Sgr today in the MW
sgr_w = np.array([19.0,2.7,-6.9,0.2352238,-0.03579493,0.19942887])

def is_box(L):
    boxy = True
    for ii in range(3):
        l = L[:,ii]
        sign = np.sign(l[0])
        if sign > 0:
            box = (l < 0).any()
        else:
            box = (l > 0).any()

        boxy = boxy & box
    return boxy

def rotate_w(w, phi, theta):
    w = np.atleast_1d(w)
    if w.ndim > 1:
        raise ValueError("w must be 1D")

    R_alt = rotation_matrix(theta-90.*u.deg, "y")
    R_azi = rotation_matrix(phi, "z")

    r = w[:w.size//2]
    v = w[w.size//2:]
    r_rot = np.array(r.dot(R_alt).dot(R_azi))
    v_rot = np.array(v.dot(R_alt.T).dot(R_azi.T))
    return np.hstack((r_rot, v_rot)).squeeze()

def main(pool, name="exp2", overwrite=False, nsteps=None, dt=None, ngrid=None):
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
    w0 = np.append(r,v)

    # generate initial conditions, uniform over 2 angles
    w0s = []
    phis = np.linspace(0,90.,ngrid)*u.deg
    thetas = np.arccos(np.linspace(0.,1.,ngrid))*u.rad
    phis,thetas = np.meshgrid(phis.to(u.deg), thetas.to(u.deg))
    phis = phis.ravel()
    thetas = thetas.ravel()
    for phi,theta in zip(phis,thetas):
        w0s.append(rotate_w(w0, phi, theta).squeeze().tolist())
    w0s = np.array(w0s)

    kwargs = dict(nsteps=nsteps, dt=dt)
    lm = LyapunovMap(name, nfw.F, lyapunov_kwargs=kwargs,
                     overwrite=overwrite, prefix=output_path,
                     Integrator=integrate.RK5Integrator)
    lm.potential_pars = nfw.vl2_params.items()

    # get a pool to use map()
    pool.map(lm, list(zip(np.arange(ngrid*ngrid),w0s)))
    pool.close()

    fig = plt.figure(figsize=(10,10))
    end_lyaps = np.zeros(ngrid*ngrid)
    box = np.zeros(ngrid*ngrid).astype(bool)
    for r in lm.iterate_cache():
        lyap, t, w, pp, fname = r
        w0 = w[0]

        # hack to get ii
        ii = w0s.tolist().index(list(w0))

        # estimate lyapunov time from final step
        max_idx = lyap.sum(axis=0).argmax()
        t_lyap = (1./lyap[:,max_idx]*u.Myr).to(u.Gyr)
        logger.debug("t_lyap = {}".format(t_lyap))
        end_lyaps[ii] = np.median(t_lyap[-100:].value)

        # see if angular momentum changes sign
        w = np.squeeze(w)
        r = w[:,:3]
        v = w[:,3:]
        L = np.cross(r,v)
        box[ii] = is_box(L)

        # pericenter and apocenter
        r = np.sqrt(np.sum(w[...,:3]**2,axis=-1))
        logger.debug("Peri: {}, Apo: {}".format(r.min(), r.max()))

        # orbit
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
        fig.suptitle(r"$\phi$={:.2f}, $\theta$={:.2f}".format(phis[ii].to(u.degree).value,
                                                              thetas[ii].to(u.degree).value))
        fig.savefig(os.path.join(plot_path,'orbit_{:03d}.png'.format(ii)))
        plt.close('all')

        # lyapunov exponents
        plt.clf()
        plt.loglog(t, lyap, marker=None)
        plt.xlim(t[1], t[-1])
        plt.ylim(1E-5, 1.)
        plt.savefig(os.path.join(plot_path, "lyap_{:03d}.png".format(ii)))

    # grid of IC's
    fig,ax = plt.subplots(1,1,figsize=(10,10))

    # colored chaotic points
    ix = end_lyaps < 10. # Gyr
    s = ax.scatter(phis.to(u.deg).value[ix], thetas.to(u.deg).value[ix],
                   c=end_lyaps[ix], s=50, cmap=cm.RdYlBu, edgecolor='#444444',
                   linewidth=1., marker="^")
    cbar = fig.colorbar(s, ax=ax)
    cbar.set_label(r'$t_{\rm lyap}$ [Gyr]')

    # boxy orbits
    s = ax.scatter(phis.to(u.deg).value[box], thetas.to(u.deg).value[box],
                   c='k', s=50, marker='s')

    # loop orbits
    s = ax.scatter(phis.to(u.deg).value[~box], thetas.to(u.deg).value[~box],
                   c='k', s=50, marker='o')

    ax.set_xlabel(r"$\phi$ [deg]")
    ax.set_ylabel(r"$\theta$ [deg]")
    fig.savefig(os.path.join(plot_path, "0_grid.png"))

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

    parser.add_argument("--ngrid", dest="ngrid", required=True, type=int,
                        help="Number of grid points along R and Rdot.")
    parser.add_argument("--name", dest="name", type=str, default="exp2")

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
    main(pool=pool, name=args.name, overwrite=args.overwrite,
         nsteps=args.nsteps, dt=args.dt, ngrid=args.ngrid)
