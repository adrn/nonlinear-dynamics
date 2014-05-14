# coding: utf-8

""" Chaos in the Law & Majewski 2010 potential? """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
from functools import partial
import logging

# Third-party
import cubehelix
from matplotlib import animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import astropy.units as u
from astropy import log as logger
logger.setLevel(logging.INFO)

# Project
from streams.potential._lm10_acceleration import lm10_acceleration, lm10_variational_acceleration
from streamteam.util import get_pool
from nonlineardynamics import LyapunovMap

# phase-space position of Sgr today
pal5_w = np.array([8.161671207, 0.244760075, 16.962073974, -0.04566825,-0.12354177,-0.01531983])

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
def F(t, X, *args):
    # args order should be: q1, qz, phi, v_halo, q2, R_halo
    x,y,z,px,py,pz = X.T
    nparticles = x.size
    acc = np.zeros((nparticles,3))
    dH_dq = lm10_acceleration(X, nparticles, acc, *args)
    return np.hstack((np.array([px, py, pz]).T, dH_dq))

def F_sali(t, X, *args):
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

    # path
    parser.add_argument("--prefix", type=str, dest="prefix", default="",
                        help="Path prefix.")

    parser.add_argument("--plot-indicators", action="store_true", dest="plot_indicators",
                        default=False)
    parser.add_argument("--plot-orbits", action="store_true", dest="plot_orbits",
                        default=False)

    # threading
    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Run with MPI.")
    parser.add_argument("--threads", dest="threads", default=None, type=int,
                        help="Number of multiprocessing threads to run on.")

    # potential parameters
    parser.add_argument("--xparam", dest="xparam", required=True, nargs="+",
                        help="Grid axis 1 specification. <param name> <num. values> "
                             "[min val, max val] ")
    parser.add_argument("--yparam", dest="yparam", required=True, nargs="+",
                        help="Grid axis 2 specification. <param name> <num. values> "
                             "[min val, max val] ")
    parser.add_argument("--name", dest="name", default=None,
                        help="Name for this experiment.")

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

    # parse parameter grid
    xname,xgrid = _parse_grid_spec(args.xparam)
    yname,ygrid = _parse_grid_spec(args.yparam)
    X,Y = map(np.ravel, np.meshgrid(xgrid, ygrid))
    gridsize = X.size

    logger.debug("{}: {}".format(xname, xgrid))
    logger.debug("{}: {}".format(yname, ygrid))

    if args.name is None:
        name = "{}_{}".format(xname, yname)
    else:
        name = args.name

    # default parameter values
    par_names = ["q1","qz","phi","v_halo","q2","r_halo"]
    default_pars = (1.38, 1.36, 1.692969, 0.12462565900, 1., 12.)

    ppars = np.zeros((gridsize, len(default_pars)))
    for ii,p in enumerate(default_pars):
        ppars[:,ii] = p

    ppars[:,par_names.index(xname)] = X
    ppars[:,par_names.index(yname)] = Y

    kwargs = dict(nsteps=args.nsteps, dt=args.dt)
    lm = LyapunovMap(name, F_sali, lyapunov_kwargs=kwargs,
                     overwrite=args.overwrite,
                     prefix=args.prefix)
    lm.w0 = pal5_w

    # get a pool to use map()
    pool = get_pool(mpi=args.mpi, threads=args.threads)
    pool.map(lm, list(zip(np.arange(gridsize),ppars)))
    pool.close()

    fig = plt.figure(figsize=(10,10))
    chaotic = np.zeros(gridsize).astype(bool)
    dm = np.zeros(gridsize)
    for r in lm.iterate_cache():
        lyap, t, w, pp, fname = r

        # prune filename
        fn = os.path.splitext(os.path.split(fname)[1])[0]

        # hack to get ii
        ii = ppars.tolist().index(list(pp))
        # chaotic[ii] = lm.is_chaotic(t, lyap)
        chaotic[ii] = True
        dm[ii] = lm.slope_diff(t, lyap)

        title = "{}={}, {}={}".format(xname, pp[par_names.index(xname)],
                                      yname, pp[par_names.index(yname)])

        if args.plot_indicators:
            plt.clf()
            plt.loglog(t,lyap,marker=None)
            plt.title(title)
            plt.xlim(t[1], t[-1])
            plt.ylim(1E-5, 1.)
            plt.savefig(os.path.join(lm.output_path, "lyap_{}.png".format(fn)))

        if args.plot_orbits:
            plt.clf()
            plt.plot(w[...,0], w[...,2], marker=None)
            plt.title(title)
            plt.xlim(-40,40)
            plt.ylim(-40,40)
            plt.savefig(os.path.join(lm.output_path, "orbit_{}.png".format(fn)))

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.set_axis_bgcolor("#eeeeee")

    #cax = ax.scatter(X[chaotic], Y[chaotic], c='k', s=75,
    cax = ax.scatter(X[chaotic], Y[chaotic], c=dm, s=75,
                     edgecolor='#666666', marker='o', cmap=cubehelix.cmap())
    # ax.plot(X[~chaotic], Y[~chaotic], color='k', markeredgewidth=1.,
    #             markeredgecolor='k', marker='x', linestyle='none', markersize=10)
    fig.colorbar(cax)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    fig.savefig(os.path.join(lm.output_path, "grid.png"))

"""

# YETI
mpiexec -n 4 /vega/astro/users/amp2217/anaconda/bin/python /vega/astro/users/amp2217/p\
rojects/nonlinear-dynamics/scripts/pal5.py -v --xparam q1 5 0.7 1.8 --yparam qz 5 0.7 \
1.8 --nsteps=10000 --mpi --prefix=/vega/astro/users/amp2217/projects/nonlinear-dynamics/output/pal5

python /vega/astro/users/amp2217/projects/nonlinear-dynamics/scripts/pal5.py -v --xparam phi 16 0. 1.7 --yparam r_halo 8 --nsteps=10000 --dt=5. --prefix=/vega/astro/users/amp2217/projects/nonlinear-dynamics/output/pal5 --plot-orbits --plot-indicators

python /vega/astro/users/amp2217/projects/nonlinear-dynamics/scripts/pal5.py -v --xparam phi 21 0. 1.7 --yparam q1 7 --nsteps=10000 --dt=5. --prefix=/vega/astro/users/amp2217/projects/nonlinear-dynamics/output/pal5 --plot-orbits --plot-indicators

mpiexec -n 4 /vega/astro/users/amp2217/anaconda/bin/python /vega/astro/users/amp2217/projects/nonlinear-dynamics/scripts/pal5.py -v --xparam q1 15 --yparam qz 15 --nsteps=10000 --dt=5. --prefix=/vega/astro/users/amp2217/projects/nonlinear-dynamics/output/pal5 --plot-orbits --plot-indicators --mpi

# LAPTOP
python scripts/pal5.py -v --xparam q1 2 0.7 1.8 --yparam qz 2 0.7 1.8 --nsteps=1000 --dt=5. --prefix=/Users/adrian/projects/nonlinear-dynamics/output/pal5 --plot-orbits --plot-indicators

# DEIMOS
python scripts/pal5.py -v --xparam q1 5 0.7 1.8 --yparam qz 5 0.7 1.8 --nsteps=1000 --dt=1. --prefix=/home/adrian/projects/nonlinear-dynamics/output/pal5 --plot-orbits

mpiexec -n 4 python /home/adrian/projects/nonlinear-dynamics/scripts/pal5.py -v --xparam q1 5 0.7 1.8 --yparam qz 5 0.7 1.8 --nsteps=1000 --mpi --prefix=/home/adrian/projects/nonlinear-dynamics/output/pal5
"""