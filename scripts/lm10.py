# coding: utf-8

""" Chaos in the Law & Majewski 2010 potential? """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import cubehelix
from matplotlib import animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from astropy.constants import G
import astropy.units as u
from scipy.signal import argrelmin
from streamteam.integrate import DOPRI853Integrator, LeapfrogIntegrator
from streamteam.dynamics import lyapunov
from streams.potential._lm10_acceleration import lm10_acceleration, lm10_potential

# Create logger
logger = logging.getLogger(__name__)

plot_path = "plots/lm10"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# unit system
usys = [u.kpc, u.Myr, u.radian, u.M_sun]
_G = G.decompose(usys).value

# standard prolate:
potential_params = (1.38, 1.36, 1.692969, 0.12462565, 1., 12.)

# phase-space position of Sgr today
#sgr_w = np.array([19.0149,2.64883,-6.8686,0.23543018,-0.03598748, 0.19917575])
sgr_w = np.array([19.0,2.7,-6.9,0.2352238,-0.03579493,0.19942887])

# basically, hamiltons equations
def F(t, X, nparticles, acc, *potential_params):
    x,y,z,px,py,pz = X.T
    dH_dq = lm10_acceleration(X, nparticles, acc, *potential_params)
    return np.hstack((np.array([px, py, pz]).T, dH_dq))

def orbit(w0, potential_params, nsteps=10000, dt=1.):
    nparticles = w0.shape[0]
    acc = np.zeros((nparticles,3))
    integrator = DOPRI853Integrator(F, func_args=(nparticles, acc)+potential_params)
    ts,ws = integrator.run(w0, dt=dt, nsteps=nsteps)
    return ts,ws

def leapfrog_orbit(w0, potential_params, nsteps=10000, dt=1.):
    nparticles = w0.shape[0]
    acc = np.zeros((nparticles,3))
    integrator = LeapfrogIntegrator(lm10_acceleration,
                                    func_args=(nparticles, acc)+potential_params)
    ts,qs,ps = integrator.run(w0[:,:3].copy(), w0[:,3:].copy(),
                           dt=dt, nsteps=nsteps)
    return ts,np.hstack((qs,ps))

def compute_lyapunov(w0, nsteps=10000, dt=1.,
                     nsteps_per_pullback=10, potential_params=potential_params):
    nparticles = 2
    acc = np.zeros((nparticles,3))

    integrator = DOPRI853Integrator(F, func_args=(nparticles, acc)+tuple(potential_params))
    print(nsteps*dt*u.Myr)
    d0 = 1e-5

    LEs, xs = lyapunov(w0, integrator, dt, nsteps,
                       d0=d0, nsteps_per_pullback=nsteps_per_pullback)

    return LEs,xs

def generate_ic_grid(dphi=10*u.deg, drdot=10*u.km/u.s):
    # spacing between IC's in Phi and Rdot
    dphi = dphi.decompose(usys).value
    drdot = drdot.decompose(usys).value
    max_rdot = (400*u.km/u.s).decompose(usys).value
    max_phi = (360*u.deg).decompose(usys).value

    # find the energy of Sgr orbit
    pot = np.zeros((1,))
    T = 0.5*np.sum(sgr_w[3:]**2)
    V = lm10_potential(sgr_w[:3].reshape((1,3)), 1, pot, *potential_params)[0]
    E = T + V

    r = 20. # kpc
    phidot = 0. # rad/Myr
    theta = (90*u.deg).decompose(usys).value # rad

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

def potential_grid(nq1=5,nqz=5):
    pps = []
    for q1 in np.append(np.linspace(0.7,1.7,nq1),1.38):
        for qz in np.append(np.linspace(0.7,1.7,nqz),1.36):
            pps.append([q1,qz])
    return np.array(pps)

if __name__ == "__main__":
    # w0s = generate_ic_grid()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(w0s[:,0],w0s[:,1],w0s[:,2],marker='.',linestyle='none')

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(w0s[:,3],w0s[:,4],w0s[:,5],marker='.',linestyle='none')
    # plt.show()
    # sys.exit(0)

    # Check that orbit looks good
    # t,w = orbit(sgr_w.reshape((1,6)), potential_params,
    #             dt=-100., nsteps=100)
    # t,lw = leapfrog_orbit(sgr_w.reshape((1,6)), potential_params,
    #                       dt=-1., nsteps=10000)
    # fig,ax = plt.subplots(1,1,figsize=(6,6))
    # ax.plot(w[:,0,0], w[:,0,2], marker=None)
    # ax.plot(lw[:,0,0], lw[:,0,2], marker=None, alpha=0.5)
    # fig.savefig("plots/lm10_orbit.png")
    # sys.exit(0)

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

    # # Vary potential parameters
    # nsteps_per_pullback = 10
    # nsteps = 100000
    # d = [] # append potential params and m,b to

    # fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(111)
    # fig2d = plt.figure(figsize=(10,10))
    # ax2d = fig2d.add_subplot(111)
    # fig3d = plt.figure(figsize=(10,10))
    # ax3d = fig3d.add_subplot(111, projection='3d')
    # for ii,(q1,qz) in enumerate(potential_grid(nq1=5,nqz=5)):
    #     pparams = list(potential_params)
    #     pparams[0] = q1
    #     pparams[1] = qz
    #     LEs,ws = compute_lyapunov(sgr_w, nsteps=nsteps,
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

    # np.savetxt("lm10.txt", np.array(d), fmt=['%.2f','%.2f','%e','%e'])

    d = np.loadtxt("lm10.txt")
    plt.figure(figsize=(8,8))

    chaotic = d[:,2] > 0.
    plt.scatter(d[chaotic,0], d[chaotic,1], c=np.log10(d[chaotic,2]), s=75,
                edgecolor='none', marker='o', cmap=cubehelix.cmap())
    plt.plot(d[~chaotic,0], d[~chaotic,1], color='k', markeredgewidth=1.,
                markeredgecolor='k', marker='x', linestyle='none', markersize=10)
    plt.colorbar()
    plt.xlabel("$q_1$")
    plt.ylabel("$q_z$")
    plt.savefig(os.path.join(plot_path, "q1_qz_grid.png"))
    sys.exit(0)

    t,w = orbit(sgr_w.reshape((1,6)), potential_params)

    plt.figure(figsize=(10,10))
    plt.plot(w[:,0,0], w[:,0,2], marker=None, linestyle='-')
    plt.savefig(os.path.join(plot_path, "sgr.png"))
