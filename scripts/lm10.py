# coding: utf-8

""" Chaos in the Law & Majewski 2010 potential? """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
from itertools import repeat
import logging

# Third-party
import cubehelix
from matplotlib import animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import astropy.units as u

from streamteam.util import get_pool
from streamteam.integrate import DOPRI853Integrator
from streamteam.dynamics import lyapunov
from streams.potential.lm10 import LawMajewski2010

# Create logger
logger = logging.getLogger(__name__)

plot_path = "plots/lm10"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# standard LM10 values:
potential = LawMajewski2010()

# phase-space position of Sgr today
sgr_w = np.array([19.0,2.7,-6.9,0.2352238,-0.03579493,0.19942887])

# hamiltons equations
def F(t, X, potential):
    x,y,z,px,py,pz = X.T
    nparticles = x.size
    acc = np.zeros((nparticles,3))
    dH_dq = potential._acceleration_at(X, nparticles, acc)
    return np.hstack((np.array([px, py, pz]).T, dH_dq))

def lyapunov_map(w0, potential_params, lyapunov_kwargs):
    potential = LawMajewski2010(**potential_params)
    integrator = DOPRI853Integrator(F, func_args=(potential,))
    LE,w = lyapunov(w0, integrator, **lyapunov_kwargs)
    return LE.shape

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

def potential_grid(**kwargs):
    if len(kwargs) > 2:
        raise ValueError("Max of 2D grid.")

    grids = []
    for k,v in kwargs.items():
        _min,_max,n = v
        if hasattr(_min, "unit"):
            _min = _min.decompose(potential.units).value
            _max = _max.decompose(potential.units).value

        grids.append((k,np.linspace(_min,_max,n)))

    param_dict_list = []
    for elem1 in grids[0][1]:
        for elem2 in grids[1][1]:
            g = dict()
            g[grids[0][0]] = elem1
            g[grids[1][0]] = elem2
            param_dict_list.append(g)

    return param_dict_list

if __name__ == "__main__":
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")

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
    else:
        logger.setLevel(logging.INFO)

    pool = get_pool(mpi=args.mpi, threads=args.threads)
    if pool is not None: map = pool.map

    # used to be orbit()
    # integrator = DOPRI853Integrator(F, func_args=(potential,))
    # ts,ws = integrator.run(w0, dt=dt, nsteps=nsteps)

    # grid of potential parameter dicts
    g = potential_grid(q1=(0.7,2.0,5),
                       phi=(45*u.deg,135*u.deg,5))
    ngrid = len(g)

    lyapunov_kwargs = dict(dt=1., nsteps=1000, noffset=4)
    print(map(lyapunov_map,
              repeat(sgr_w,ngrid)),
              [potential._parameter_dict for ii in range(ngrid)],
              repeat(lyapunov_kwargs,ngrid))

    if pool is not None:
        pool.close()

    sys.exit(0)

    # # Vary orbit parameters
    # fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(111)
    # fig3d = plt.figure(figsize=(10,10))
    # ax3d = fig3d.add_subplot(111, projection='3d')
    # for ii,w0 in enumerate(w0s):
    #     LEs,ws = compute_lyapunov(w0, nsteps=100000)

    #     print("Lyapunov exponent computed")
    #     ax.cla()
    #     ax.semilogy(LEs, marker=None)
    #     fig.savefig(os.path.join(plot_path,'le_{}.png'.format(ii)))

    #     ax3d.cla()
    #     ax3d.plot(ws[:,0], ws[:,1], ws[:,2], marker=None)
    #     fig3d.savefig(os.path.join(plot_path,'orbit_{}.png'.format(ii)))

    # Vary potential parameters
    nsteps_per_pullback = 10
    nsteps = 10000
    dt = 10.

    d = [] # append potential params and m,b to

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    fig2d = plt.figure(figsize=(10,10))
    ax2d = fig2d.add_subplot(111)
    fig3d = plt.figure(figsize=(10,10))
    ax3d = fig3d.add_subplot(111, projection='3d')
    for ii,(q1,qz) in enumerate(potential_grid(nq1=25,nqz=25)):
    #for ii,(q1,qz) in enumerate([(0.7,0.866666666),(0.8666666666,1.366666666),(1.2,0.866666666),
    #                             (1.2,1.533333333),(1.53333333,1.03333333)]):
        pparams = list(potential_params)
        pparams[0] = q1
        pparams[1] = qz
        LEs,ws = compute_lyapunov(sgr_w, nsteps=nsteps, dt=dt,
                                  nsteps_per_pullback=nsteps_per_pullback,
                                  potential_params=tuple(pparams))

        # take only the 2nd half
        slc = nsteps//2//nsteps_per_pullback
        y = LEs[slc:]
        x = np.arange(0,len(y))
        A = np.vstack([x, np.ones(len(x))]).T
        m,b = np.linalg.lstsq(A, y)[0]
        d.append([q1,qz,m,b])

        print("Lyapunov exponent computed")
        ax.cla()
        ax.set_title("q1={}, qz={}".format(q1,qz))
        ax.semilogy(LEs, marker=None)
        fig.savefig(os.path.join(plot_path,'le_{}.png'.format(ii)))

        ax2d.cla()
        ax2d.set_title("q1={}, qz={}".format(q1,qz))
        ax2d.plot(ws[:,0], ws[:,2], marker=None)
        fig2d.savefig(os.path.join(plot_path,'2d_orbit_{}.png'.format(ii)))

        ax3d.cla()
        ax3d.set_title("q1={}, qz={}".format(q1,qz))
        ax3d.plot(ws[:,0], ws[:,1], ws[:,2], marker=None)
        fig3d.savefig(os.path.join(plot_path,'3d_orbit_{}.png'.format(ii)))

    d = np.array(d)
    np.savetxt("lm10.txt", np.array(d), fmt=['%.2f','%.2f','%e','%e'])

    d = np.loadtxt("lm10.txt")
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.set_axis_bgcolor("#eeeeee")

    chaotic = (d[:,2] > 0.) | ((d[:,2] < 0.) & (np.log10(np.abs(d[:,2])) < -8.))
    cax = ax.scatter(d[chaotic,0], d[chaotic,1], c=np.log10(d[chaotic,3]), s=75,
                     edgecolor='#666666', marker='o', cmap=cubehelix.cmap())
    ax.plot(d[~chaotic,0], d[~chaotic,1], color='k', markeredgewidth=1.,
                markeredgecolor='k', marker='x', linestyle='none', markersize=10)
    fig.colorbar(cax)
    ax.set_xlabel("$q_1$")
    ax.set_ylabel("$q_z$")
    fig.savefig(os.path.join(plot_path, "q1_qz_grid.png"))
    sys.exit(0)

    t,w = orbit(sgr_w.reshape((1,6)), potential_params)

    plt.figure(figsize=(10,10))
    plt.plot(w[:,0,0], w[:,0,2], marker=None, linestyle='-')
    plt.savefig(os.path.join(plot_path, "sgr.png"))
