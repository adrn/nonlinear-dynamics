# coding: utf-8

""" Chaos in the Law & Majewski 2010 potential? """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
from functools import partial
import logging
import base64

# Third-party
import h5py
import cubehelix
from matplotlib import animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import astropy.units as u
from astropy import log as logger
logger.setLevel(logging.INFO)

from streamteam.util import get_pool
from streamteam.integrate import DOPRI853Integrator
from streamteam.dynamics import lyapunov
from streams.potential._lm10_acceleration import lm10_acceleration

# phase-space position of Sgr today
sgr_w = np.array([19.0,2.7,-6.9,0.2352238,-0.03579493,0.19942887])

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
    # dH_dq = potential._acceleration_at(X, nparticles, acc)
    return np.hstack((np.array([px, py, pz]).T, dH_dq))

class LyapunovMap(object):

    def __init__(self, name, func, func_args=tuple(),
                 lyapunov_kwargs=dict(), Integrator=DOPRI853Integrator,
                 output_file=None, overwrite=False, prefix=""):
        """ TODO

            Parameters
            ----------
            name : str
            func : callable
                Must accept a single argument - the potential to run in.
            func_args : sequence
                Extra arguments passed to the equations of motion function.
                These get *prepended* to parameter arguments passed later.
            lyapunov_kwargs : keyword arguments
                Other arguments passed to `lyapunov()`. Things like the
                number of steps, timestep, etc.
            Integrator : streamteam.Integrator (optional)
            cache_data : bool (optional)
                Cache output data.
            make_plots : bool (optional)
                Plot orbits and Lyapunov exponents/
            overwrite : bool (optional)
                Overwrite cached data files.
            prefix : str (optional)
                Prefix to the path.
        """

        # path to save data and plots
        self.output_path = os.path.join(prefix, "output/{}".format(name))
        self.cache_path = os.path.join(self.output_path, "cache")
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        self.F = func
        self._F_args = tuple(func_args)

        # class to use for integration
        self.Integrator = Integrator
        self.lyapunov_kwargs = lyapunov_kwargs

        self.overwrite = bool(overwrite)

        self.w0 = None
        self.potential_pars = None

    def _map_helper(self, w0, potential_pars, filename=None):
        """ """

        if filename is None:
            hashstr = "".join((str(w0) + str(potential_pars)).split())
            hashed = base64.b64encode(hashstr)
            fn = os.path.join(self.cache_path, "{}.hdf5".format(hashed))
        else:
            fn = os.path.join(self.cache_path, filename)

        if os.path.exists(fn) and self.overwrite:
            os.remove(fn)

        if not os.path.exists(fn):
            args = self._F_args + tuple(potential_pars)
            integrator = self.Integrator(F, func_args=args)
            LE,t,w = lyapunov(w0, integrator, **self.lyapunov_kwargs)

            with h5py.File(fn, "w") as f:
                f["lambda_k"] = LE
                f["t"] = t
                f["w"] = w
                f["potential_pars"] = potential_pars

        else:
            with h5py.File(fn, "r") as f:
                LE = f["lambda_k"].value
                t = f["t"].value
                w = f["w"].value

        return LE,t,w

    def __call__(self, arg):

        if self.w0 is None and self.potential_pars is not None:
            # assume arg is (index, w0)
            index,w0 = arg
            return self._map_helper(w0, self.potential_pars,
                                    filename="{}.hdf5".format(index))

        elif self.potential_pars is None and self.w0 is not None:
            # assume arg is (index, potential_pars)
            index,potential_pars = arg
            return self._map_helper(self.w0, potential_pars,
                                    filename="{}.hdf5".format(index))

        else:
            raise ValueError("Must set either initial conditions or "
                             "potential parameters.")

    def plot_lambda(self, lambda_k):
        # TODO: how to get info about parameters in here?
        ax.cla()
        ax.set_title("q1={}, qz={}".format(q1,qz))
        ax.semilogy(LEs, marker=None)
        fig.savefig(os.path.join(plot_path,'le_{}.png'.format(ii)))

    def classify_chaotic(self, lambda_k):

        lambda_k = np.mean(lambda_k, axis=1)

        # take only the 2nd half
        y = lambda_k[len(lambda_k)//2:]

        window = 10
        niter = len(y) // window
        mn = []

        for ii in range(niter):
            mn.append(np.mean(y[ii*window:ii*window+window]))
        mn = np.array(mn)

        # take only the 2nd half
        x = np.arange(0,len(mn))
        A = np.vstack([x, np.ones(len(x))]).T
        m,b = np.linalg.lstsq(A, mn)[0]

        # plt.figure()
        # if m > (-1.8E-7):
        #     plt.title("Chaotic")
        # plt.semilogy(lambda_k, marker=None)

        # plt.figure()
        # plt.plot(mn, marker=None)
        # plt.plot(x, m*x+b, marker=None)
        # plt.show()

        return m


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

    lyapunov_kwargs = dict(nsteps=args.nsteps, dt=args.dt, noffset=4)
    lm = LyapunovMap(name, F, lyapunov_kwargs=lyapunov_kwargs,
                     output_file=None, overwrite=args.overwrite,
                     prefix=args.prefix)
    lm.w0 = sgr_w

    # get a pool to use map()
    pool = get_pool(mpi=args.mpi, threads=args.threads)
    results = pool.map(lm, zip(np.arange(gridsize),ppars))

    ms = np.zeros(len(results))
    for ii,r in enumerate(results):
        ms[ii] = lm.classify_chaotic(r[0])

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)

    # neg_ms = np.abs(ms[ms < 0.])
    # bins = np.logspace(np.log10(neg_ms.min()),np.log10(neg_ms.max()),10)
    # print(neg_ms, bins)
    # ax.hist(neg_ms, bins=bins)
    # ax.set_xscale('log')
    # fig.savefig(os.path.join(lm.output_path, "ms.png"))

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.set_axis_bgcolor("#eeeeee")

    chaotic = (ms > 0.) | (ms > (-1.8E-7))
    # cax = ax.scatter(X[chaotic], Y[chaotic], c=np.log10(ms[chaotic]), s=75,
    #                  edgecolor='#666666', marker='o', cmap=cubehelix.cmap())
    cax = ax.scatter(X[chaotic], Y[chaotic], c='k', s=75,
                     edgecolor='#666666', marker='o')
    ax.plot(X[~chaotic], Y[~chaotic], color='k', markeredgewidth=1.,
                markeredgecolor='k', marker='x', linestyle='none', markersize=10)
    # fig.colorbar(cax)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    fig.savefig(os.path.join(lm.output_path, "grid.png"))

    sys.exit(0)

    # t,w = orbit(sgr_w.reshape((1,6)), potential_params)

    # plt.figure(figsize=(10,10))
    # plt.plot(w[:,0,0], w[:,0,2], marker=None, linestyle='-')
    # plt.savefig(os.path.join(plot_path, "sgr.png"))

def ic_grid(dphi=10*u.deg, drdot=10*u.km/u.s):
    # spacing between IC's in Phi and Rdot
    dphi = dphi.decompose(potential.units).value
    drdot = drdot.decompose(potential.units).value
    max_rdot = (400*u.km/u.s).decompose(potential.units).value
    max_phi = (360*u.deg).decompose(potential.units).value

    # find the energy of Sgr orbit
    pot = np.zeros((1,))
    T = 0.5*np.sum(sgr_w[3:]**2)
    V = lm10_potential(sgr_w[:3].reshape((1,3)), 1, pot, *potential_params)[0]
    E = T + V

    r = 20. # kpc
    phidot = 0. # rad/Myr
    theta = (90*u.deg).decompose(potential.units).value # rad

    # T = 0.5*(rdot**2 + thetadot**2/r**2 + phidot**2/(r*np.sin(theta))**2)

    w0s = []
    for rdot in np.arange(0, max_rdot, drdot):
        for phi in np.arange(0, max_phi, dphi):
            X = r*np.cos(phi)*np.sin(theta)
            Y = r*np.sin(phi)*np.sin(theta)
            Z = r*np.cos(theta)
            V = lm10_potential(np.vstack((X,Y,Z)).T, 1, pot, *potential_params)[0]

            # solve for theta dot from energy
            ptheta = r*np.sqrt(2*(E - V) - rdot**2 - phidot**2/(r*np.sin(theta))**2)
            thetadot = ptheta / r**2

            vx = rdot*np.sin(theta)*np.cos(phi) + r*np.cos(theta)*np.cos(phi)*thetadot - r*np.sin(theta)*np.sin(phi)*phidot
            vy = rdot*np.sin(theta)*np.sin(phi) + r*np.cos(theta)*np.sin(phi)*thetadot + r*np.sin(theta)*np.cos(phi)*phidot
            vz = rdot*np.cos(theta) - r*np.sin(theta)*thetadot

            w0 = [X,Y,Z,vx,vy,vz]
            if np.any(np.isnan(w0)): continue
            w0s.append(w0)

    w0s = np.array(w0s)
    return w0s
