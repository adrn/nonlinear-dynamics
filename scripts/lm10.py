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

try:
    x,v = os.path.split(__file__)
    if x != "scripts":
        raise IOError()
except:
    logger.error("You must run scripts from the top level in the cloned "
                 "nonlinear-dynamics directory.")
    sys.exit(1)

# plot_path = "plots/lm10"
# if not os.path.exists(plot_path):
#     os.makedirs(plot_path)

# cache_path = "cache/lm10"
# if not os.path.exists(cache_path):
#     os.makedirs(cache_path)

# phase-space position of Sgr today
sgr_w = np.array([19.0,2.7,-6.9,0.2352238,-0.03579493,0.19942887])

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
                 output_file=None, overwrite=False):
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
        """

        # path to save data and plots
        self.output_path = "output/{}".format(name)
        self.cache_path = os.path.join(self.output_path, "cache")
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        if output_file is None:
            output_file = "lyapunov.hdf5"
        self.output_filename = os.path.join(self.cache_path, output_file)

        self.F = func
        self._F_args = tuple(func_args)

        # class to use for integration
        self.Integrator = Integrator
        self.lyapunov_kwargs = lyapunov_kwargs


        overwrite = bool(overwrite)
        if os.path.exists(self.output_filename) and overwrite:
            logger.debug("Overwriting (removing) file {}".format(self.output_filename))
            os.remove(self.output_filename)

        if not os.path.exists(self.output_filename):
            logger.debug("File {} does not exist".format(self.output_filename))
            self.file = h5py.File(self.output_filename,'w')
        else:
            logger.debug("File {} already exists".format(self.output_filename))
            self.file = h5py.File(self.output_filename,'r+')

        self.w0 = None
        self.potential_pars = None

    def _map_helper(self, w0, potential_pars):
        """ """
        hashstr = "".join((str(w0) + str(potential_pars)).split())
        hashed = base64.b64encode(hashstr)

        try:
            grp = self.file[hashed]
            LE,t,w = grp["lambda_k"].value,grp["t"].value,grp["w"].value
        except KeyError:
            grp = self.file.create_group(hashed)

            args = self._F_args + tuple(potential_pars)
            integrator = self.Integrator(F, func_args=args)
            LE,t,w = lyapunov(w0, integrator, **self.lyapunov_kwargs)

            dset1 = grp.create_dataset("lambda_k", data=LE)
            dset2 = grp.create_dataset("t", data=t)
            dset3 = grp.create_dataset("w", data=w)

        return LE,t,w

    def __call__(self, arg):

        if self.w0 is None and self.potential_pars is not None:
            # assume arg is (index, w0)
            return self._map_helper(arg, self.potential_pars)

        elif self.potential_pars is None and self.w0 is not None:
            # assume arg is (index, potential_pars)
            return self._map_helper(self.w0, arg)

        else:
            raise ValueError("Must set either initial conditions or "
                             "potential parameters.")

def potential_grid(x, y, **kwargs):
    if len(x) > 1 or len(y) > 1:
        raise ValueError("Only 1D TODO")

    grids = []
    name1,v1 = x.items()[0]
    name2,v2 = y.items()[0]

    _min,_max,n = v1
    if hasattr(_min, "unit"):
        _min = _min.decompose(potential.units).value
        _max = _max.decompose(potential.units).value
    vals1 = np.linspace(_min,_max,n)

    _min,_max,n = v2
    if hasattr(_min, "unit"):
        _min = _min.decompose(potential.units).value
        _max = _max.decompose(potential.units).value
    vals2 = np.linspace(_min,_max,n)

    param_dict_list = []
    for elem1 in vals1:
        for elem2 in vals2:
            g = dict()
            g[name1] = elem1
            g[name2] = elem2
            param_dict_list.append(dict(g.items()+kwargs.items()))

    return param_dict_list

def main(pool, grid1, grid2, **lyapunov_kwargs):

    # grid of potential parameter dicts
    grid = potential_grid(grid1, grid2)

    func = partial(lyapunov_map_potential, sgr_w, path, **lyapunov_kwargs)
    LEs = pool.map(func, enumerate(grid))

    return

    r = []
    p1,p2,p3 = False,False,False
    for ii,LE in enumerate(LEs):
        # take only the 2nd half
        y = LE[len(LE)//2:]
        x = np.arange(0,len(y))
        A = np.vstack([x, np.ones(len(x))]).T
        m,b = np.linalg.lstsq(A, y)[0]
        r.append([m,b])

        if m < -1E-7 and b > 0.0005 and not p1:
            plot_that(LE, grid[ii][grid1.keys()[0]],
                      grid[ii][grid2.keys()[0]], ii)
            print("p1: {}".format(ii))
            p1 = True

    k1,k2 = grid_kwargs.keys()
    names = [k1,k2,"m","b"]
    return np.array([(g[k1],g[k2],row[0],row[1]) for g,row in zip(grid,r)],
                    dtype=[(n,float) for n in names])

def plot_that(LE, q1, qz, ii):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    # fig2d = plt.figure(figsize=(10,10))
    # ax2d = fig2d.add_subplot(111)

    ax.cla()
    ax.set_title("q1={}, qz={}".format(q1,qz))
    ax.semilogy(LE, marker=None)
    fig.savefig(os.path.join(plot_path,'le_{}.png'.format(ii)))

    # ax2d.cla()
    # ax2d.set_title("q1={}, qz={}".format(q1,qz))
    # ax2d.plot(ws[:,0], ws[:,2], marker=None)
    # fig2d.savefig(os.path.join(plot_path,'2d_orbit_{}.png'.format(ii)))

if __name__ == "__main__":
    def _parse_grid(tup):
        pname = tup[0]
        n = tup[-1]
        _min,_max = tup[1:3]

        return {pname : (u.Quantity.from_string(_min), u.Quantity.from_string(_max), int(n))}

    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite",
                        default=False, help="Overwrite cached files.")

    # potential parameter grid
    # parser.add_argument("--grid1", nargs=4, dest="grid1", required=True,
    #                     help="Grid axis 1. Should be <param name> <min. val> "
    #                     "<max. val> <num. vals>")
    # parser.add_argument("--grid2", nargs=4, dest="grid2", required=True,
    #                     help="Grid axis 2. Should be <param name> <min. val> "
    #                     "<max. val> <num. vals>")

    # threading
    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Run with MPI.")
    parser.add_argument("--threads", dest="threads", default=None, type=int,
                        help="Number of multiprocessing threads to run on.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)

    pool = get_pool(mpi=args.mpi, threads=args.threads)

    lyapunov_kwargs = dict(nsteps=1000, dt=1., noffset=4)
    test = LyapunovMap("test", F, lyapunov_kwargs=lyapunov_kwargs,
                       output_file=None, overwrite=args.overwrite)

    test.w0 = sgr_w
    args = np.zeros((10,6))
    args[:,0] = 1.38
    args[:,1] = np.linspace(1.,1.7,args.shape[0])
    args[:,2] = 1.692969
    args[:,3] = 0.12462565900
    args[:,4] = 1.
    args[:,5] = 12.

    r = pool.map(test, args)

    test.file.close()

    sys.exit(0)

    g1 = _parse_grid(args.grid1)
    g2 = _parse_grid(args.grid2)
    g = dict(g1.items() + g2.items())

    name = "_".join(sorted(g.keys()))
    cache_name = os.path.join(cache_path, "{}.txt".format(name))
    plot_name = os.path.join(plot_path, "{}.png".format(name))

    if os.path.exists(cache_name) and args.overwrite:
        os.remove(cache_name)

    if not os.path.exists(cache_name):
        pool = get_pool(mpi=args.mpi, threads=args.threads)
        lya_exps = main(pool, g1, g2, dt=10., nsteps=10000, noffset=4)
        pool.close()

        header = ",".join(lya_exps.dtype.names)
        np.savetxt(cache_name, lya_exps, header=header,
                   delimiter=",", fmt=['%.4f','%.4f','%e','%e'])

    lya_exps = np.genfromtxt(cache_name, delimiter=",", names=True)

    x1,x2 = lya_exps['m'].min(),lya_exps['m'].max()
    dx = x2-x1
    y1,y2 = lya_exps['b'].min(),lya_exps['b'].max()
    dy = y2-y1

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.set_axis_bgcolor("#eeeeee")
    ax.scatter(lya_exps['m'], lya_exps['b'], c='k', s=50,
               edgecolor='#666666', marker='o')#, cmap=cubehelix.cmap())
    #ax.set_yscale('log')
    ax.set_xlim(x1-dx/10,x2+dx/10)
    ax.set_ylim(y1-dy/10,y2+dy/10)
    fig.savefig(plot_name)

    sys.exit(0)

    # # # Vary orbit parameters
    # # fig = plt.figure(figsize=(10,10))
    # # ax = fig.add_subplot(111)
    # # fig3d = plt.figure(figsize=(10,10))
    # # ax3d = fig3d.add_subplot(111, projection='3d')
    # # for ii,w0 in enumerate(w0s):
    # #     LEs,ws = compute_lyapunov(w0, nsteps=100000)

    # #     print("Lyapunov exponent computed")
    # #     ax.cla()
    # #     ax.semilogy(LEs, marker=None)
    # #     fig.savefig(os.path.join(plot_path,'le_{}.png'.format(ii)))

    # #     ax3d.cla()
    # #     ax3d.plot(ws[:,0], ws[:,1], ws[:,2], marker=None)
    # #     fig3d.savefig(os.path.join(plot_path,'orbit_{}.png'.format(ii)))

    # # Vary potential parameters
    # nsteps_per_pullback = 10
    # nsteps = 10000
    # dt = 10.

    # d = [] # append potential params and m,b to

    # fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(111)
    # fig2d = plt.figure(figsize=(10,10))
    # ax2d = fig2d.add_subplot(111)
    # fig3d = plt.figure(figsize=(10,10))
    # ax3d = fig3d.add_subplot(111, projection='3d')
    # for ii,(q1,qz) in enumerate(potential_grid(nq1=25,nqz=25)):
    # #for ii,(q1,qz) in enumerate([(0.7,0.866666666),(0.8666666666,1.366666666),(1.2,0.866666666),
    # #                             (1.2,1.533333333),(1.53333333,1.03333333)]):
    #     pparams = list(potential_params)
    #     pparams[0] = q1
    #     pparams[1] = qz
    #     LEs,ws = compute_lyapunov(sgr_w, nsteps=nsteps, dt=dt,
    #                               nsteps_per_pullback=nsteps_per_pullback,
    #                               potential_params=tuple(pparams))

    #     # take only the 2nd half
    #     slc = nsteps//2//nsteps_per_pullback
    #     y = LEs[slc:]
    #     x = np.arange(0,len(y))
    #     A = np.vstack([x, np.ones(len(x))]).T
    #     m,b = np.linalg.lstsq(A, y)[0]
    #     d.append([q1,qz,m,b])

    #     print("Lyapunov exponent computed")
    #     ax.cla()
    #     ax.set_title("q1={}, qz={}".format(q1,qz))
    #     ax.semilogy(LEs, marker=None)
    #     fig.savefig(os.path.join(plot_path,'le_{}.png'.format(ii)))

    #     ax2d.cla()
    #     ax2d.set_title("q1={}, qz={}".format(q1,qz))
    #     ax2d.plot(ws[:,0], ws[:,2], marker=None)
    #     fig2d.savefig(os.path.join(plot_path,'2d_orbit_{}.png'.format(ii)))

    #     ax3d.cla()
    #     ax3d.set_title("q1={}, qz={}".format(q1,qz))
    #     ax3d.plot(ws[:,0], ws[:,1], ws[:,2], marker=None)
    #     fig3d.savefig(os.path.join(plot_path,'3d_orbit_{}.png'.format(ii)))

    # d = np.array(d)
    # np.savetxt("lm10.txt", np.array(d), fmt=['%.2f','%.2f','%e','%e'])

    # d = np.loadtxt("lm10.txt")
    # fig = plt.figure(figsize=(8,8))
    # ax = fig.add_subplot(111)
    # ax.set_axis_bgcolor("#eeeeee")

    # chaotic = (d[:,2] > 0.) | ((d[:,2] < 0.) & (np.log10(np.abs(d[:,2])) < -8.))
    # cax = ax.scatter(d[chaotic,0], d[chaotic,1], c=np.log10(d[chaotic,3]), s=75,
    #                  edgecolor='#666666', marker='o', cmap=cubehelix.cmap())
    # ax.plot(d[~chaotic,0], d[~chaotic,1], color='k', markeredgewidth=1.,
    #             markeredgecolor='k', marker='x', linestyle='none', markersize=10)
    # fig.colorbar(cax)
    # ax.set_xlabel("$q_1$")
    # ax.set_ylabel("$q_z$")
    # fig.savefig(os.path.join(plot_path, "q1_qz_grid.png"))
    # sys.exit(0)

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